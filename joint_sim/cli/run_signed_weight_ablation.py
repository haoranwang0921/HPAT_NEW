#!/usr/bin/env python
"""Run LPWM signed-weight MRR mapping ablations on an audited trace.

【中文说明】
这是"负权重符号处理"的消融实验。背景知识：光子微环（MRR）阵列做矩阵乘时，
每个微环的透射率只能取 0~1 的非负值（光强是能量，没有负数），而神经网络的
权重是有正有负的。所以必须设计"符号处理方案"。本实验对比 4 种方案：

- UNSIGNED（无符号）：沿用旧版"只存非负值"的 64×64 tile 基线（实际近似）。
- SIGNED_FIXED_BALANCED（固定面积·平衡方案）：把正、负权重行相邻排布，
  一对正负行共用一个"固定面积"，接收端做平衡差分，不用额外数字减法。
- SIGNED_FIXED_DIGITAL（固定面积·数字减法）：同样固定面积相邻排布正负行，
  但接收端用"双支路 ADC + 数字减法"把负号减回来。
- SIGNED_DUAL_BANK（双库方案）：正负权重各占一块独立阵列（吞吐不变），
  面积与能耗按上界（×2）估算。

每种方案再分两种存储模式（storage_mode）：
  compact   压缩存储：tile 体积减半（2048 字节/块），SRAM 占用更少
  expanded  展开存储：tile 体积不变（4096 字节/块），与旧版一致

实验会依次仿真 5 个配置（UNSIGNED、BALANCED×2 种存储、DIGITAL compact、
DUAL_BANK compact），各自输出端到端时延/能耗/面积等，并写 JSON+CSV 汇总。

运行方式：python joint_sim/cli/run_signed_weight_ablation.py
常用参数：--capacity SRAM 容量（MB，默认 384）、--pprog 编程并行度（默认 96）、
  --pipeline 流水线模式（默认 triple_pipeline）、--digital-sub-energy-pj /
  --digital-sub-latency-ns 给数字减法附加的每输出能耗/每向量时延（用于
  SIGNED_FIXED_DIGITAL 方案的成本修正）、--overwrite 允许覆盖输出目录。
产出：results/signed_weight_ablation/ 下每个配置一个 JSON，
  以及 signed_ablation_summary.json / signed_ablation_summary.csv 汇总。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

# 把项目根目录、SimPhony、LLMCompass 加入搜索路径
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "SimPhony", ROOT / "LLMCompass"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joint_sim.sram_manager import SramManager, LruSramPolicy
from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.op_classifier import classify_operator
from joint_sim.scheduler import ResourcePool, ResourceScheduler
# 负权重适配器：把 SimPhony 的成本/架构按符号方案做"改造"（面积、能耗调整）
from joint_sim.signed_mrr_adapter import (
    SIGNED_DUAL_BANK,
    SIGNED_FIXED_BALANCED,
    SIGNED_FIXED_DIGITAL,
    UNSIGNED,
    adapt_architecture_cost,  # 按方案调整架构成本（面积/器件数）
    adapt_kernel_cost,  # 按方案调整单个内核的成本（含可选的数字减法代价）
)
from joint_sim.simphony_backend import SimPhonyBackend
from joint_sim.tile_mapper import TileMapper
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


@dataclass(frozen=True)
class ModeConfig:
    """一个符号处理方案 + 存储模式的完整配置。

    字段含义：
      mode                  方案代号（UNSIGNED / SIGNED_FIXED_BALANCED / ...）
      storage_mode          存储模式（compact 压缩 / expanded 展开）
      mapping_mode          TileMapper 用的映射模式名
      weight_tile_bytes     每个权重 tile 占多少字节（影响 SRAM 占用）
      program_energy_multiplier  编程能耗乘子（双库方案×2）
      hold_power_multiplier      保持功率乘子（双库方案×2）
      interpretation        方案的一句话解释（写入报告供人阅读）
    """
    mode: str
    storage_mode: str
    mapping_mode: str
    weight_tile_bytes: int
    program_energy_multiplier: float
    hold_power_multiplier: float
    interpretation: str


def build_mode_config(mode: str, storage_mode: str) -> ModeConfig:
    """按 (方案, 存储模式) 构造对应的 ModeConfig。

    中文说明：每种方案的 tile 体积、能耗乘子等硬件参数不同：
    - UNSIGNED：非负基线，tile 4096 字节，能耗/功率不变。
    - SIGNED_FIXED_BALANCED / DIGITAL：固定面积相邻排布正负行，所以 tile
      大小可以"压缩"到 2048 字节（compact）或保持 4096（expanded）；
      能耗不变。
    - SIGNED_DUAL_BANK：正负两块阵列各一份，面积/能耗都按 2 倍上界估算。
    参数校验：storage_mode 只接受 compact/expanded，mode 必须是四种之一，
    否则抛 ValueError。
    """
    if storage_mode not in {"compact", "expanded"}:
        raise ValueError(f"unsupported storage mode: {storage_mode!r}")
    if mode == UNSIGNED:
        return ModeConfig(
            mode, storage_mode, "unsigned_64x64", 4096, 1.0, 1.0,
            "legacy unsigned 64x64 baseline",  # 旧版非负 64×64 基线
        )
    if mode in {SIGNED_FIXED_BALANCED, SIGNED_FIXED_DIGITAL}:
        return ModeConfig(
            mode,
            storage_mode,
            "signed_row_pair_fixed_area",  # 固定面积的正负行成对排布
            # compact 时 tile 减半（2048B），expanded 保持 4096B
            2048 if storage_mode == "compact" else 4096,
            1.0,
            1.0,
            (
                # balanced：正负行相邻，接收端做平衡差分（无数字减法）
                "fixed-area adjacent positive/negative rows; balanced receiver"
                if mode == SIGNED_FIXED_BALANCED
                else # digital：同样相邻排布，但用双支路 ADC + 数字减法还原符号
                "fixed-area adjacent rows; two-branch ADC plus digital subtraction"
            ),
        )
    if mode == SIGNED_DUAL_BANK:
        return ModeConfig(
            mode, storage_mode, "unsigned_64x64", 4096, 2.0, 2.0,
            "fixed-throughput dual-bank area/energy upper bound",  # 双库的面积/能耗上界
        )
    raise ValueError(f"unsupported mode: {mode!r}")


def _percentile(values: List[float], q: float) -> float:
    """计算一组数值的第 q 百分位（线性插值，空列表返回 0.0）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q  # 目标位置（可能是小数）
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    # 落在两个值之间：按比例线性插值
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _step_durations(cumulative: Dict[int, float]) -> List[float]:
    """把"每步的累计结束时刻"转成"每步的持续时长"列表。

    中文说明：调度器给的 per_step_latency_s 是"到第几步为止的累计时延"，
    这里把相邻两步的累计值相减，得到每一步自己的时长。
    """
    durations = []
    previous = 0.0
    for _, end_time in sorted((int(k), float(v)) for k, v in cumulative.items()):
        durations.append(max(0.0, end_time - previous))  # 相减并防负数
        previous = end_time
    return durations


def _eligible(rec: dict) -> bool:
    """判断一个算子是否有资格上光子阵列（复用统一分类器）。"""
    return classify_operator(
        rec.get("op_type", ""),
        rec.get("op_role", "unknown"),
        rec.get("weight_static", False),
    )["eligible"]


def run_mode(
    manifest: dict,
    records: List[dict],
    sim: SimPhonyBackend,
    elec: ElectronicBackend,
    cfg: ModeConfig,
    *,
    capacity: int,
    pprog: int,
    pipeline: str,
    digital_sub_energy_per_output_j: float,
    digital_sub_latency_per_vector_s: float,
) -> dict:
    """按给定符号方案配置完整仿真一次，返回详细结果字典。

    中文说明：
    流程：按 cfg 的映射/存储模式建 TileMapper → 建签名缓存的"光子成本函数"
    （用 adapt_kernel_cost 把 SimPhony 成本按方案改造，可加数字减法代价）→
    建 SRAM 管理器与资源池（注意 tile 体积取 cfg.weight_tile_bytes）→ 调度
    整个算子轨迹 → 汇总时延/能耗/面积等字段。返回的结果字典会被写入报告文件。
    参数：manifest 轨迹清单；records 算子记录；sim/elec 光子/电子后端；
    cfg 方案配置；capacity/pprog/pipeline 是仿真参数；
    digital_sub_energy_per_output_j 与 digital_sub_latency_per_vector_s 是
    数字减法方案的附加单位代价（能量 J/输出、时延 s/向量）。
    """
    # 按 cfg 的映射/存储模式创建 tile 映射器（signed 方案会改变分块方式）
    mapper = TileMapper(
        mapping_mode=cfg.mapping_mode,
        storage_mode=cfg.storage_mode,
    )
    accesses = mapper.build_tile_trace(records)
    tile_stats = mapper.stats()

    kernel_cache: Dict[tuple, dict] = {}

    def photonic_cost(rec: dict) -> dict:
        # 按 (M,K,N,位宽,方案) 签名缓存"改造后的光子成本"
        signature = (
            int(rec["M"]), int(rec["K"]), int(rec["N"]),
            int(rec.get("input_bits", 8)),
            int(rec.get("weight_bits", 8)),
            int(rec.get("output_bits", 8)),
            cfg.mode,
        )
        if signature not in kernel_cache:
            # adapt_kernel_cost 负责把 SimPhony 原始成本按方案改造，
            # 并附上可选的数字减法代价
            kernel_cache[signature] = adapt_kernel_cost(
                sim.kernel_cost,
                rec,
                cfg.mode,
                digital_sub_energy_per_output_j=digital_sub_energy_per_output_j,
                digital_sub_latency_per_vector_s=digital_sub_latency_per_vector_s,
            )
        return kernel_cache[signature]

    # SRAM 管理器：容量 capacity MB，LRU 逐出；tile 大小按方案取
    sram_mgr = SramManager(
        total_bytes=capacity * 1024 * 1024,
        policy=LruSramPolicy(),
        tile_bytes=cfg.weight_tile_bytes,
    )
    pool = ResourcePool(
        num_photonic_cores=1,
        num_dac_channels=1,
        num_adc_channels=1,
        num_program_lanes=pprog,
        weight_tile_bytes=cfg.weight_tile_bytes,  # 编程/DMA 按 tile 字节计费
    )
    scheduler = ResourceScheduler(
        resource_pool=pool,
        electronic_latency_fn=elec.operator_latency_s,
        photonic_latency_fn=photonic_cost,
        sram_manager=sram_mgr,
        pipeline_mode=pipeline,
        # 每个 tile 的编程能耗：按 0.093e-3（编程功率 W）× 1000e-9（时长 s）
        # × 64×64（tile 微环数）估算，再乘方案的能耗乘子
        program_energy_j_per_tile=(
            0.093e-3 * 1000e-9 * 64 * 64 * cfg.program_energy_multiplier
        ),
    )

    # 完整调度一遍算子轨迹，记录真实墙钟耗时
    wall_start = time.perf_counter()
    scheduler.schedule_trace(
        records,
        classifier_fn=classify_operator,
        tile_mapper=mapper,
        tile_accesses=accesses,
    )
    wall_time_s = time.perf_counter() - wall_start
    sched = scheduler.summary()
    # 把 SRAM 统计与流式（编程/DMA）统计合并成一个字典方便取值
    bank = {**sched.get("sram_manager_summary", {}), **sched.get("streaming_stats", {})}
    e2e_s = scheduler.end_to_end_latency_s  # 端到端时延

    # 按方案改造架构成本；保持功率按方案的乘子缩放（双库方案×2）
    architecture = adapt_architecture_cost(sim.architecture_cost(), cfg.mode)
    program = sim.programming_cost()
    hold_power_w = float(program["hold_power_w"]) * cfg.hold_power_multiplier
    laser_power_w = float(architecture["laser_wall_plug_power_w"])

    # --- 能量核算：逐算子累加 ---
    photonic_dynamic_j = 0.0  # 光子动态能耗（扣除内嵌调谐）
    dac_energy_j = 0.0
    adc_energy_j = 0.0
    digital_sub_energy_j = 0.0  # 数字减法附加能耗
    electronic_energy_j = 0.0  # 电子后端能耗
    photonic_calls = 0
    electronic_calls = 0
    for rec in records:
        if _eligible(rec):
            photonic_calls += 1
            cost = photonic_cost(rec)
            # 动态能耗扣掉内嵌的 MRR 调谐（调谐按保持项单独计），避免重复计费
            photonic_dynamic_j += max(
                0.0,
                float(cost.get("dynamic_energy_j", 0.0))
                - float(cost.get("mrr_tuning_energy_j", 0.0)),
            )
            dac_energy_j += float(cost.get("dac_energy_j", 0.0))
            adc_energy_j += float(cost.get("adc_energy_j", 0.0))
            digital_sub_energy_j += float(cost.get("digital_sub_energy_j", 0.0))
        else:
            # 电子算子：时延与能耗来自电子后端
            electronic_calls += 1
            latency = elec.operator_latency_s(rec)
            electronic_energy_j += elec.operator_energy_j(rec, latency)

    # 静态/持续项按"功率 × 端到端时长"积分一次
    programming_energy_j = float(bank.get("programming_energy_j", 0.0))  # 调度编程事件能耗
    hold_energy_j = hold_power_w * e2e_s  # MRR 保持能耗
    laser_energy_j = laser_power_w * e2e_s  # 激光能耗
    # 总能量 = 光子动态 + 编程 + 电子 + 保持 + 激光
    total_energy_j = (
        photonic_dynamic_j
        + programming_energy_j
        + electronic_energy_j
        + hold_energy_j
        + laser_energy_j
    )
    # 每步时长列表与守恒校验
    steps = _step_durations(sched.get("per_step_latency_s", {}))
    conservation = sched.get("conservation_checks", [])

    # 组装完整结果字典（含方案配置、时延、能量、面积、守恒等全部字段）
    result = {
        **asdict(cfg),  # 展开方案配置字段
        "cost_source": "SimPhony analytical model + joint_sim event scheduler",
        "trace_steps": manifest.get("num_steps"),
        "operator_count": len(records),
        "photonic_calls": photonic_calls,
        "electronic_calls": electronic_calls,
        "coverage": photonic_calls / len(records) if records else 0.0,
        "e2e_latency_s": e2e_s,
        "step_count": len(steps),
        "step_latency_p50_s": statistics.median(steps) if steps else 0.0,  # 每步时延中位数
        "step_latency_p95_s": _percentile(steps, 0.95),  # 每步时延第 95 百分位
        "simulation_wall_time_s": wall_time_s,
        "total_events": sched.get("total_events", 0),
        "unique_weights": tile_stats["total_unique_weights"],
        "unique_tiles": tile_stats["total_unique_tiles"],
        "tile_accesses": len(accesses),
        "logical_weight_storage_bytes": tile_stats["total_storage_bytes"],
        "physical_mrr_slots_per_tile": 4096,  # 每个 tile 物理占用 4096 个微环槽位
        "logical_outputs_per_core": architecture["logical_outputs_per_core"],
        "programs": bank.get("programs", 0),  # 编程事件次数
        "evictions": bank.get("evictions", 0),  # SRAM 逐出次数
        "hit_rate": bank.get("hit_rate", 0.0),  # SRAM 命中率
        "dma_bytes": bank.get("dma_bytes", 0),
        "dma_latency_s": bank.get("dma_latency_s", 0.0),
        "programming_latency_s": bank.get("programming_latency_s", 0.0),
        "programming_critical_path_s": sched.get("programming_critical_path_s", 0.0),
        # --- 能量分解 ---
        "photonic_dynamic_energy_j": photonic_dynamic_j,
        "dac_energy_j": dac_energy_j,
        "adc_energy_j": adc_energy_j,
        "digital_sub_energy_j": digital_sub_energy_j,
        "programming_energy_j": programming_energy_j,
        "electronic_energy_j": electronic_energy_j,
        "mrr_hold_power_w": hold_power_w,
        "mrr_hold_energy_j": hold_energy_j,
        "laser_power_w": laser_power_w,
        "laser_energy_j": laser_energy_j,
        "total_energy_j": total_energy_j,
        # --- 架构（面积/器件数）---
        "mrr_count": architecture["mrr_count"],
        "pd_count": architecture["pd_count"],
        "dac_count": architecture["dac_count"],
        "adc_count": architecture["adc_count"],
        "pic_area_um2": architecture["pic_area_um2"],
        "rf_eic_area_um2": architecture["rf_eic_area_um2"],
        "total_area_um2": architecture["total_area_um2"],
        "core_insertion_loss_db": architecture["core_insertion_loss_db"],
        "extra_split_loss_db": architecture["extra_split_loss_db"],  # 符号拆分额外损耗
        "is_upper_bound": architecture["is_upper_bound"],  # 是否为面积上界
        # --- 校验 ---
        "conservation_passed": all(c.get("passed", False) for c in conservation),
        "conservation_checks": conservation,
        "kernel_signatures": len(kernel_cache),  # 缓存里独特内核签名数
    }

    # 校验：固定面积 signed 映射必须保持 MRR/PD 数量守恒（若违反说明适配逻辑有 bug）
    if cfg.mode in {SIGNED_FIXED_BALANCED, SIGNED_FIXED_DIGITAL}:
        if result["mrr_count"] != 393216 or result["pd_count"] != 6144:
            raise AssertionError("fixed-area signed mapping violated MRR/PD conservation")
    # 守恒检查失败直接抛错，防止把不合法结果写进报告
    if not result["conservation_passed"]:
        raise AssertionError(f"resource conservation failed for {cfg.mode}")
    return result


def main() -> None:
    """命令行入口：依次仿真 5 个符号方案配置，写出 JSON 与 CSV 汇总。

    中文说明：
    依次跑 5 个配置：UNSIGNED、SIGNED_FIXED_BALANCED(compact)、
    SIGNED_FIXED_DIGITAL(compact)、SIGNED_FIXED_BALANCED(expanded)、
    SIGNED_DUAL_BANK。每个配置跑完写一个独立 JSON，最后写汇总 JSON 与 CSV。
    命令行参数：--trace 轨迹路径；--output 输出目录；--capacity SRAM 容量；
    --pprog 编程并行度；--pipeline 流水线模式；
    --digital-sub-energy-pj / --digital-sub-latency-ns 给数字减法方案附加的
    单位能耗（皮焦耳）/单位时延（纳秒），运行前统一换算成 J 和 s；
    --overwrite 允许覆盖非空输出目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        default=str(ROOT / "traces" / "lpwm_bair128_checkpoint_horizon30.jsonl"),
    )
    parser.add_argument("--output", default=str(ROOT / "results" / "signed_weight_ablation"))
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--pprog", type=int, default=96)
    parser.add_argument(
        "--pipeline",
        choices=["serial", "dma_program_overlap", "triple_pipeline"],
        default="triple_pipeline",
    )
    parser.add_argument("--digital-sub-energy-pj", type=float, default=0.0)
    parser.add_argument("--digital-sub-latency-ns", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    output = Path(args.output).resolve()
    # 防御：非空输出目录默认拒绝覆盖
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    # 读轨迹、建两个后端
    manifest, records = load_manifest_jsonl(trace_path)
    sim = SimPhonyBackend()
    elec = ElectronicBackend()
    # 待跑的 5 个配置（方案 × 存储模式）
    configs = [
        build_mode_config(UNSIGNED, "compact"),  # 非负基线
        build_mode_config(SIGNED_FIXED_BALANCED, "compact"),  # 平衡方案·压缩
        build_mode_config(SIGNED_FIXED_DIGITAL, "compact"),  # 数字减法·压缩
        build_mode_config(SIGNED_FIXED_BALANCED, "expanded"),  # 平衡方案·展开
        build_mode_config(SIGNED_DUAL_BANK, "compact"),  # 双库上界
    ]

    results = []
    for cfg in configs:
        print(f"Running {cfg.mode} storage={cfg.storage_mode} ...", flush=True)
        # 运行单个配置（注意把皮焦耳/纳秒统一换算成焦耳/秒）
        result = run_mode(
            manifest,
            records,
            sim,
            elec,
            cfg,
            capacity=args.capacity,
            pprog=args.pprog,
            pipeline=args.pipeline,
            digital_sub_energy_per_output_j=args.digital_sub_energy_pj * 1e-12,
            digital_sub_latency_per_vector_s=args.digital_sub_latency_ns * 1e-9,
        )
        results.append(result)
        # 每个配置写一个独立 JSON 文件
        run_name = f"{cfg.mode}__{cfg.storage_mode}"
        (output / f"{run_name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"  E2E={result['e2e_latency_s']*1e3:.3f} ms  "
            f"tiles={result['unique_tiles']} accesses={result['tile_accesses']}  "
            f"energy={result['total_energy_j']:.6f} J",
            flush=True,
        )

    # 汇总 JSON：记录轨迹哈希与实验参数，便于复现与追溯
    json_path = output / "signed_ablation_summary.json"
    json_path.write_text(
        json.dumps({
            "trace": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "capacity": args.capacity,
            "pprog": args.pprog,
            "pipeline": args.pipeline,
            "digital_sub_energy_pj": args.digital_sub_energy_pj,
            "digital_sub_latency_ns": args.digital_sub_latency_ns,
            "results": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # 汇总 CSV：只保留"标量字段"（字典/列表这类复杂值不适合放进 CSV 单元格）
    csv_path = output / "signed_ablation_summary.csv"
    scalar_keys = [
        key for key, value in results[0].items()
        if not isinstance(value, (dict, list))
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in scalar_keys})
    print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
