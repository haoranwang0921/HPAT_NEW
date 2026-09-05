#!/usr/bin/env python
"""
Unified joint-simulation experiment entry point.

Wires the real SimPhony + ElectronicBackend into the tile-granularity
ResourceScheduler + SramManager.  Supports residency-strategy sweeps,
capacity sweeps, and P_prog / pipeline-mode comparisons.

中文阅读提示：这是最适合从头追代码的入口。它负责组装各个模块，不在这里
实现器件公式或调度细节；遇到具体行为时再跳转到对应模块。

【中文说明 · 本文件是整套仿真器的总入口】
功能：读入 LPWM 算子执行轨迹（trace）→ 用 TileMapper 把权重切成 tile（分块）
→ 把真实的 SimPhony 光子成本与 ElectronicBackend 电子成本接入"tile 粒度"
的资源调度器（ResourceScheduler）+ SRAM 管理器（SramManager）→ 统计端到端
时延与总能耗，并输出诊断报告（含能量总账、守恒校验、逐算子成本）。

支持三种运行形态：
  1) 单次运行（默认）：指定策略/容量/编程并行度/流水线模式跑一次，
     生成完整诊断报告（含能量总账 energy_total.csv）。
  2) 参数扫描（--sweep）：
       --sweep capacity   策略 × 容量扫描（对应实验4a）
       --sweep pprog      编程并行度扫描（实验4b）
       --sweep pipeline   三种流水线模式对比
能量总账的构成（这也是 power_breakdown.py 打印的内容）：
  总能量 = 光子动态能耗 + 调度编程事件能耗 + 电子后端能耗
           + 激光功率 × 总耗时 + MRR 保持功率 × 总耗时
其中静态项（激光/MRR 保持）按"功率 × 端到端时长"积分一次，绝不能逐算子
累加（否则重叠执行的算子会被重复计费）；动态项（DAC/ADC/MRR 调谐）是离散
事件，可以逐次累加。

Usage (from repository root):

    # Single run with defaults
    python joint_sim/cli/run_joint_sim.py

    # Strategy sweep across capacities
    python joint_sim/cli/run_joint_sim.py --sweep capacity

    # P_prog sweep
    python joint_sim/cli/run_joint_sim.py --sweep pprog

    # Pipeline mode comparison
    python joint_sim/cli/run_joint_sim.py --sweep pipeline
"""

from __future__ import annotations

import argparse, csv, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 把项目根目录、SimPhony、LLMCompass 三个目录加进搜索路径
ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "SimPhony"), str(ROOT / "LLMCompass")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from joint_sim.sram_manager import make_sram_manager  # 按策略名字创建 SRAM 管理器
from joint_sim.electronic_backend import ElectronicBackend  # 电子后端（时延/能耗模型）
from joint_sim.op_classifier import classify_operator  # 算子分类器（能否上光子阵列）
from joint_sim.report import JointSimReport, SweepReport  # 单次报告 / 扫描报告
from joint_sim.scheduler import ResourcePool, ResourceScheduler, EventType  # 资源池与调度器
from joint_sim.simphony_backend import SimPhonyBackend  # 光子后端（成本模型）
from joint_sim.tile_mapper import TileMapper  # 权重分块映射器
from joint_sim.trace_io import load_manifest_jsonl, sha256_file  # 轨迹读写工具

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 常量区：数据流方式、默认输入输出路径、扫描取值点
DATAFLOW = "weight_stationary"  # 光子矩阵乘数据流：权重驻留
DEFAULT_TRACE = str(ROOT / "traces" / "lpwm_bair128_checkpoint_horizon30.jsonl")  # 默认轨迹
DEFAULT_OUTPUT = str(ROOT / "results" / "joint_sim")  # 默认输出目录

CAPACITY_POINTS = [24, 48, 96, 192, 384]  # SRAM capacity in MB  # 容量扫描取值（MB）
PPROG_POINTS = [1, 2, 4, 8, 16, 32, 64, 96]  # 编程并行度扫描取值
PIPELINE_MODES = ["serial", "dma_program_overlap", "triple_pipeline"]  # 三种流水线模式

STRATEGY_MAP: Dict[str, str] = {  # 驻留策略名 → 策略名（映射表，供 --strategy 选择）
    "lru": "lru",
    "belady_optimal": "belady_optimal",
    "rollout_aware": "rollout_aware",
}


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------
# 轨迹读写区

def load_trace(path: Path) -> Tuple[dict, List[dict]]:
    """统一使用严格的 Trace 读取器，避免实验入口绕开字段校验。

    中文说明：读取"首行清单 + 每行一个算子"的 JSONL 轨迹。
    返回 (清单元信息 dict, 算子记录列表)。
    """
    return load_manifest_jsonl(path)


def kernel_signature(rec: dict) -> tuple:
    """将影响光子成本的字段组成缓存键；相同签名复用一次 SimPhony 查询。

    中文说明：光子阵列跑一个矩阵乘的成本只取决于 (M, K, N, 输入位宽, 权重位宽,
    输出位宽, 数据流方式)。把这些字段拼成元组当缓存键，相同形状的算子只向
    SimPhony 查询一次，避免重复计算。
    """
    return (
        int(rec["M"]), int(rec["K"]), int(rec["N"]),
        int(rec.get("input_bits", 8)),
        int(rec.get("weight_bits", 8)),
        int(rec.get("output_bits", 8)),
        DATAFLOW,
    )


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------
# 后端接线区：创建光子/电子两个成本后端，并包成调度器可直接调用的函数

def build_backends(simphony_subprocess: bool = False):
    """创建两类成本后端，并返回调度器可直接调用的函数。

    中文说明：
    1) SimPhonyBackend：光子后端，提供 kernel_cost（矩阵乘成本）、
       architecture_cost（架构面积/激光功率）、programming_cost（编程开销）。
       use_subprocess=True 时把 SimPhony 放到子进程跑（隔离其调试输出）。
    2) ElectronicBackend：电子后端，提供算子时延/能耗估计。
    返回：5 元组 (sim, elec, photonic_latency_fn, electronic_latency_fn, kernel_cache)。
    photonic_latency_fn(rec) 按签名缓存查光子成本；electronic_latency_fn(rec)
    返回电子时延；kernel_cache 是共享的签名→成本缓存（供能量核算复用）。
    """
    sim = SimPhonyBackend(use_subprocess=simphony_subprocess)
    elec = ElectronicBackend()

    # Pre-compute kernel costs for each unique (M,K,N,input_bits,weight_bits,output_bits)
    # signature in the trace.  This avoids calling SimPhony for every record.
    # 为轨迹里每种独特的 (M,K,N,位宽) 签名预计算内核成本，避免逐算子调用 SimPhony
    kernel_cache: Dict[tuple, dict] = {}

    def photonic_latency_fn(rec: dict) -> dict:
        # 调度器调用它查光子成本：先算签名，缓存没有就现查并存入缓存
        sig = kernel_signature(rec)
        if sig not in kernel_cache:
            m, k, n, ib, wb, ob, _df = sig
            kernel_cache[sig] = sim.kernel_cost(
                M=m, K=k, N=n, input_bits=ib, weight_bits=wb, output_bits=ob,
            )
        return kernel_cache[sig]

    def electronic_latency_fn(rec: dict) -> float:
        # 电子算子的时延直接问电子后端
        return elec.operator_latency_s(rec)

    return sim, elec, photonic_latency_fn, electronic_latency_fn, kernel_cache


# ---------------------------------------------------------------------------
# Single-run API
# ---------------------------------------------------------------------------
# 单次运行 API

def run_one(
    records: List[dict],
    tile_mapper: TileMapper,
    tile_accesses: list,
    strategy: str = "lru",
    total_slots: int = 96,
    program_parallelism: int = 96,
    pipeline_mode: str = "serial",
    photonic_latency_fn: Callable = None,
    electronic_latency_fn: Callable = None,
) -> ResourceScheduler:
    """Run a single simulation configuration.

    中文说明：跑一次完整仿真。流程：按策略建 SRAM 管理器 → 建资源池（1 个光子
    核 + 1 路 DAC/ADC + program_parallelism 条编程通道）→ 建调度器 → 对整个
    算子轨迹调度执行。返回调度器对象（其 end_to_end_latency_s、events、
    summary() 等携带全部仿真结果）。未传成本函数时用占位常数（1e-6/10e-9），
    主要用于独立测试调度逻辑而不依赖真实后端。
    """
    # 按策略名字创建带容量上限的 SRAM 管理器（total_slots 单位 MB）
    sram_mgr = make_sram_manager(total_mb=total_slots, policy=strategy)
    pool = ResourcePool(
        num_photonic_cores=1,  # 光子核数量
        num_dac_channels=1,  # DAC（数模转换）通道
        num_adc_channels=1,  # ADC（模数转换）通道
        num_program_lanes=program_parallelism,  # 编程通道数 = P_prog
    )
    sched = ResourceScheduler(
        resource_pool=pool,
        # 没传成本函数时用占位常数，便于不依赖真实后端做逻辑测试
        electronic_latency_fn=electronic_latency_fn or (lambda r: 1e-6),
        photonic_latency_fn=photonic_latency_fn or (lambda r: 10e-9),
        sram_manager=sram_mgr,
        pipeline_mode=pipeline_mode,
    )
    # 调度整个算子轨迹（classify_operator 决定每个算子去光子还是电子）
    sched.schedule_trace(
        records, classifier_fn=classify_operator,
        tile_mapper=tile_mapper, tile_accesses=tile_accesses,
    )
    return sched


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
# 报告辅助区：把一次运行的所有结果整理成诊断报告文件

def write_run_report(
    output_dir: Path, run_id: str,
    manifest: dict, records: List[dict],
    sim_backend: SimPhonyBackend,
    kernel_cache: Dict[tuple, dict],
    sched: ResourceScheduler,
    tile_mapper: TileMapper,
    electronic_backend: ElectronicBackend,
):
    """Write the full diagnostic report for one run.

    中文说明：
    为一次单次运行写出完整诊断报告，内容包括：架构信息、调度器汇总、SRAM
    驻留汇总、事件时间线、每个算子的成本（光子/电子）、能量总账、守恒校验。
    能量总账是本报告最重要的部分，其核算原则（易错点）：
      - 动态能耗（DAC/ADC/MRR 调谐等离散事件）可跨算子安全累加；
      - 静态能耗（激光器、MRR 保持功率）是"持续消耗"，必须用
        功率 × 端到端总时长一次性积分，不能逐算子累加——否则重叠执行的
        多个算子会重复计费。
    参数：output_dir 报告输出目录；run_id 本次运行标识；其余为仿真各组件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture = sim_backend.architecture_cost()  # 光子架构信息（面积/功率/器件数）

    # 用 JointSimReport 逐块填充报告内容
    report = JointSimReport(output_dir=str(output_dir), run_id=run_id)
    report.set_architecture(architecture)
    report.set_scheduler_summary(sched.summary())
    report.set_residency_summary(sched.summary().get("sram_manager_summary"))
    report.set_timeline(sched.timeline_csv())  # 事件时间线

    # 给每个算子做光子/电子分类，生成 op_id → 是否光子 的映射
    classifications: Dict[str, bool] = {}
    for rec in records:
        verdict = classify_operator(
            rec.get("op_type", ""), rec.get("op_role", "unknown"),
            rec.get("weight_static", False),
        )
        classifications[rec["op_id"]] = verdict["eligible"]

    # Per-operator costs with invocation-counted energy
    # 统计每个"形状签名"在轨迹里出现多少次（同一权重被多次调用 = 多次计算）
    sig_counts: Dict[tuple, int] = defaultdict(int)
    for rec in records:
        if classifications.get(rec["op_id"]):
            sig = kernel_signature(rec)
            sig_counts[sig] += 1

    # 从调度事件里统计每个算子被"编程"（把权重写进微环）的总能耗
    seen_sigs: set = set()
    programming_by_op = defaultdict(float)
    for evt in sched.events:
        if evt.event_type == EventType.WEIGHT_PROGRAM:  # 只关心权重编程事件
            programming_by_op[evt.op_id] += evt.energy_j

    # Accumulate invocation-weighted energy totals
    # 能量总账：先清零所有分项，再逐算子累加
    energy_totals = {
        "dac_energy_j": 0.0, "adc_energy_j": 0.0,
        "laser_energy_j": 0.0, "mrr_tuning_energy_j": 0.0,
        "mrr_hold_energy_j": 0.0, "dynamic_energy_j": 0.0,
        "electronic_energy_j": 0.0,
        "total_compute_time_s": 0.0,
    }

    # 逐算子核算：光子算子取 SimPhony 内核成本（按调用次数加权），
    # 电子算子用电子后端的时延/能耗模型
    for rec in records:
        is_photonic = classifications[rec["op_id"]]
        if is_photonic:
            sig = kernel_signature(rec)
            cost = kernel_cache.get(sig, {})  # 查该形状的光子成本
            inv_count = sig_counts.get(sig, 1)  # 该形状被调用的次数
            if sig not in seen_sigs:  # 每个唯一形状只登记一次内核成本
                seen_sigs.add(sig)
                report.add_kernel_cost(rec["M"], rec["K"], rec["N"], rec["op_id"], cost)
                # Accumulate invocation-weighted energy
                # 能量按"单次成本 × 调用次数"加权累加（同一权重多次计算要算多次能量）
                for key in energy_totals:
                    if key in cost:
                        energy_totals[key] += cost[key] * inv_count
            # 记录该算子的成本（时延、动态能耗、编程能耗）
            report.add_operator_cost(rec, {
                "latency_s": float(cost.get("compute_latency_s", 0)),
                "dynamic_energy_j": cost.get("dynamic_energy_j", 0.0),
                "memory_energy_j": 0.0,
                "programming_energy_j": programming_by_op[rec["op_id"]],
            }, "photonic")
        else:
            # 电子算子：时延与能耗来自电子后端
            lat = electronic_backend.operator_latency_s(rec)
            e_energy = electronic_backend.operator_energy_j(rec, lat)
            energy_totals["electronic_energy_j"] += e_energy
            report.add_operator_cost(rec, {
                "latency_s": lat,
                "dynamic_energy_j": e_energy,
                "memory_energy_j": 0.0,
                "programming_energy_j": 0.0,
            }, "electronic")

    # ---- Static energy correction ----
    # Per-kernel MRR hold and laser values from SimPhony cover the kernel's
    # active window (compute + programming) and CANNOT be summed across
    # overlapping kernels.  Static power is continuous: P_static x E2E_time.
    # 静态能耗修正：SimPhony 给的"每内核 MRR 保持/激光"只覆盖该内核的活动
    # 窗口，跨重叠内核不能累加。静态功率是持续消耗，应按 功率 × 端到端总时长。
    e2e_s = sched.end_to_end_latency_s  # 端到端总时延（关键路径）
    prog_cost = sim_backend.programming_cost()
    hold_power_w = prog_cost.get("hold_power_w", 39.3)  # MRR 保持功率（默认 39.3W）
    laser_w = architecture["laser_wall_plug_power_w"]  # 激光器墙插功率

    # Dynamic energy (discrete events — safe to sum across invocations)
    # 动态能耗（离散事件，跨调用累加是安全的）
    raw_dynamic_j = energy_totals["dynamic_energy_j"]
    embedded_tuning_j = energy_totals["mrr_tuning_energy_j"]
    energy_totals["raw_simphony_dynamic_energy_j"] = raw_dynamic_j  # 保留原始值供核对
    energy_totals["embedded_mrr_tuning_excluded_j"] = embedded_tuning_j  # 被扣除的调谐能量
    # MRR 调谐能量已内嵌在 SimPhony 动态能量里；为避免与下面的保持项重复计费，
    # 从动态项中扣除（保持项单独按功率×时长算）
    energy_totals["dynamic_energy_j"] = max(
        0.0, raw_dynamic_j - embedded_tuning_j
    )
    energy_totals["mrr_tuning_energy_j"] = 0.0
    # 调度编程事件的能耗（把权重写进微环的开销）
    energy_totals["programming_energy_j"] = sched.summary()["streaming_stats"]["programming_energy_j"]

    # Static energy (continuous — P x T, not per-kernel sum)
    # 静态能耗（持续项：功率 × 端到端时长，不是逐内核求和）
    energy_totals["laser_total_j"] = laser_w * e2e_s  # 激光总能耗
    energy_totals["mrr_hold_total_j"] = hold_power_w * e2e_s  # MRR 保持总能耗
    energy_totals["hold_power_w"] = hold_power_w
    energy_totals["laser_power_w"] = laser_w

    # Per-kernel sums kept for reference (indicate they overlap)
    # 逐内核累加值仅作参考保留（注意它们有重叠，不能当总账用）
    energy_totals["laser_per_kernel_sum_j"] = energy_totals["laser_energy_j"]
    energy_totals["mrr_hold_per_kernel_sum_j"] = energy_totals["mrr_hold_energy_j"]

    energy_totals["e2e_latency_s"] = e2e_s
    energy_totals["num_steps"] = int(manifest.get("num_steps") or 1)
    # 最终总能量：动态 + 编程 + 电子 + 激光 + MRR 保持（五项，避免重复计费）
    energy_totals["total_energy_j"] = sum(
        energy_totals[key] for key in (
            "dynamic_energy_j", "programming_energy_j",
            "electronic_energy_j", "laser_total_j", "mrr_hold_total_j",
        )
    )

    report.set_energy_totals(energy_totals)

    # Conservation checks  # 守恒校验：事件数/能量收支是否自洽
    for check in sched.check_conservation():
        report.add_conservation_check(check["check"], check["passed"], check["detail"])

    report.write_all()  # 把所有内容写成报告文件

    # Print summary  # --- 屏幕打印关键结果摘要 ---
    sram_sum = sched.summary().get("sram_manager_summary") or {}
    stream = sched.summary().get("streaming_stats") or {}
    print(f"\n  Architecture: MRR={architecture['mrr_count']}  PD={architecture['pd_count']}  "
          f"Area={architecture['total_area_um2']:.1f} um^2")
    print(f"  E2E latency:  {sched.end_to_end_latency_s:.6e} s")
    print(f"  Total events: {len(sched.events)}")
    print(f"  Residency:    hit_rate={sram_sum.get('hit_rate', 0):.1%}  "
          f"programs={stream.get('programs', 0)}  evictions={sram_sum.get('evictions', 0)}")
    print(f"  DMA:          {sram_sum.get('dma_latency_s', 0):.4e} s  "
          f"{sram_sum.get('dma_bytes', 0)} bytes")
    print(f"  Conservation: {'ALL PASS' if all(c['passed'] for c in sched.check_conservation()) else 'FAILURES'}")

    # Per-step summary
    # 按 rollout 步统计时延，并计算"每步时延增量"（观察自回归越长时延怎么涨）
    per_step = sched.per_step_latency()
    keys = sorted(per_step.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
    if len(keys) >= 2:
        growth = (per_step[keys[-1]] - per_step[keys[0]]) / max(1, int(keys[-1]) - int(keys[0]))
        print(f"  Per-step:     {len(keys)} steps  growth={growth:.4e} s/step")


# ---------------------------------------------------------------------------
# Sweep commands
# ---------------------------------------------------------------------------
# 参数扫描区：三种扫描各是一个函数

def sweep_capacity(
    records, tile_mapper, tile_accesses,
    photonic_latency_fn, electronic_latency_fn,
    output_dir: Path,
):
    """Strategy × capacity sweep (Experiment 4a).

    中文说明：容量扫描（对应实验4a）。在 3 种驻留策略 × 5 档容量的每个组合上
    各跑一次仿真（固定 P_prog=96、串行流水线），把每次运行记入 SweepReport，
    最后写出报告并打印汇总表。观察"SRAM 多大才够、什么策略更好"。
    """
    sweep = SweepReport(str(output_dir))
    N = len(STRATEGY_MAP) * len(CAPACITY_POINTS)  # 总运行次数
    i = 0
    print(f"\nCapacity sweep: {len(STRATEGY_MAP)} strategies × {len(CAPACITY_POINTS)} capacities = {N} runs\n")
    print(f"{'#':>3}  {'Strategy':<16} {'Cap':>5} {'HitRate':>8} {'Hits':>7} {'Misses':>7} {'Evicts':>7} {'E2E(s)':>12} {'Time':>6}")
    print("-" * 90)
    # 双重循环：每种策略 × 每档容量
    for strat_name, _ in STRATEGY_MAP.items():
        for cap in CAPACITY_POINTS:
            i += 1
            t0 = time.perf_counter()
            sched = run_one(
                records, tile_mapper, tile_accesses,
                strategy=strat_name, total_slots=cap,
                program_parallelism=96,  # 固定 P_prog=96
                photonic_latency_fn=photonic_latency_fn,
                electronic_latency_fn=electronic_latency_fn,
            )
            elapsed = time.perf_counter() - t0
            config = {"strategy": strat_name, "total_slots": cap,
                      "program_parallelism": 96, "pipeline_mode": "serial"}
            sweep.add_run(config, sched.summary())  # 记入报告
            sm = sched.summary()["sram_manager_summary"]  # SRAM 命中/缺失/逐出
            # 打印一行结果
            print(f"  [{i:>2}/{N}] {strat_name:<16} {cap:>5} {sm['hit_rate']:>7.1%} "
                  f"{sm['hits']:>7} {sm['misses']:>7} "
                  f"{sm['evictions']:>7} {sched.end_to_end_latency_s:>12.4e} {elapsed:>5.0f}s")
    sweep.write_all()  # 写出全部报告文件
    sweep.print_sweep_table()
    return sweep


def sweep_pprog(
    records, tile_mapper, tile_accesses,
    photonic_latency_fn, electronic_latency_fn,
    output_dir: Path,
):
    """P_prog sweep (Experiment 4b).

    中文说明：编程并行度扫描（对应实验4b）。固定 LRU 策略、384MB 容量、串行
    流水线，逐个扫描 P_prog ∈ {1,2,4,...,96}，观察"同时编程的微环越多，
    编程时延/端到端时延能压多少"。注意：P_prog 再高，编程总工作量不变，
    主要赢在并行度。
    """
    sweep = SweepReport(str(output_dir))
    N = len(PPROG_POINTS)
    print(f"\nP_prog sweep: {N} points (lru, cap=384, serial)\n")
    print(f"{'P_prog':>6} {'HitRate':>8} {'Prog(s)':>12} {'DMA(s)':>12} {'E2E(s)':>12} {'Time':>6}")
    print("-" * 65)
    for p_prog in PPROG_POINTS:
        t0 = time.perf_counter()
        sched = run_one(
            records, tile_mapper, tile_accesses,
            strategy="lru", total_slots=384,
            program_parallelism=p_prog,
            photonic_latency_fn=photonic_latency_fn,
            electronic_latency_fn=electronic_latency_fn,
        )
        elapsed = time.perf_counter() - t0
        config = {"strategy": "lru", "total_slots": 384,
                  "program_parallelism": p_prog, "pipeline_mode": "serial"}
        sweep.add_run(config, sched.summary())
        sm = sched.summary()["sram_manager_summary"]
        stream = sched.summary()["streaming_stats"]
        print(f"  {p_prog:>6} {sm['hit_rate']:>7.1%} "
              f"{stream['programming_latency_s']:>12.4e} {sm['dma_latency_s']:>12.4e} "
              f"{sched.end_to_end_latency_s:>12.4e} {elapsed:>5.0f}s")
    sweep.write_all()
    return sweep


def sweep_pipeline(
    records, tile_mapper, tile_accesses,
    photonic_latency_fn, electronic_latency_fn,
    output_dir: Path,
):
    """Pipeline mode comparison.

    中文说明：流水线模式对比。固定 LRU、384MB、P_prog=96，对比三种流水线：
      serial                完全串行（最朴素，最慢）
      dma_program_overlap   DMA 搬运与权重编程重叠进行
      triple_pipeline       三重流水（搬运/编程/计算三级流水）
    目的是看"让准备工作和计算重叠"能省多少关键路径时延。
    """
    sweep = SweepReport(str(output_dir))
    print(f"\nPipeline mode comparison (lru, cap=384, P_prog=96)\n")
    for mode in PIPELINE_MODES:
        t0 = time.perf_counter()
        sched = run_one(
            records, tile_mapper, tile_accesses,
            strategy="lru", total_slots=384,
            program_parallelism=96, pipeline_mode=mode,
            photonic_latency_fn=photonic_latency_fn,
            electronic_latency_fn=electronic_latency_fn,
        )
        elapsed = time.perf_counter() - t0
        config = {"strategy": "lru", "total_slots": 384,
                  "program_parallelism": 96, "pipeline_mode": mode}
        sweep.add_run(config, sched.summary())
        sm = sched.summary()["sram_manager_summary"]
        print(f"  {mode:<24} E2E={sched.end_to_end_latency_s:.4e}s  "
              f"hit={sm['hit_rate']:.1%}  {elapsed:.0f}s")
    sweep.write_all()
    return sweep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """命令行总入口：解析参数 → 读轨迹 → 建 tile 映射 → 接后端 → 运行/扫描 → 写报告。

    中文说明：
    流程（对应模块 docstring 的用法）：
      1) 解析命令行参数，校验输出目录（非空需 --overwrite）。
      2) 读入轨迹并打印算子数、步数、SHA-256 哈希（保证可复现）。
      3) 用 TileMapper 建立权重分块映射，统计权重/分块/访问次数。
      4) 接线后端：SimPhony（光子）+ ElectronicBackend（电子）。
      5) 按 --sweep 分派：capacity / pprog / pipeline 扫描，或默认单次运行
         （单次运行会调用 write_run_report 生成完整诊断报告）。
      6) 把输入轨迹复制到输出目录（复现依据），打印完成。
    常用参数：--strategy/-s（驻留策略）、--capacity/-c（SRAM 容量 MB）、
      --pprog/-p（编程并行度）、--pipeline（流水线模式）、--sweep（扫描模式）、
      --simphony-subprocess（SimPhony 放子进程）、--overwrite（允许覆盖输出）。
    产出：results/joint_sim/ 下的诊断报告（含能量总账 energy_total.csv、
      时间线、逐算子成本、守恒校验等）；扫描模式则产出 SweepReport 报告。
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default=DEFAULT_TRACE, help="LPWM trace JSONL path")
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output directory")
    ap.add_argument("--strategy", "-s", default="lru",
                    choices=list(STRATEGY_MAP.keys()), help="Residency strategy")
    ap.add_argument("--capacity", "-c", type=float, default=96, help="SRAM capacity in MB")
    ap.add_argument("--pprog", "-p", type=int, default=96, help="Program parallelism (P_prog)")
    ap.add_argument("--pipeline", default="serial", choices=PIPELINE_MODES,
                    help="Pipeline mode")
    ap.add_argument("--sweep", choices=["capacity", "pprog", "pipeline"],
                    help="Run a parameter sweep instead of a single run")
    ap.add_argument("--simphony-subprocess", action="store_true",
                    help="Run SimPhony in a subprocess")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output directory")
    args = ap.parse_args()

    trace_path = Path(args.trace).resolve()
    output_dir = Path(args.output).resolve()

    # 防御：非空输出目录默认拒绝覆盖，必须显式 --overwrite
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        print(f"Output directory {output_dir} is not empty. Use --overwrite to replace.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load trace
    # ------------------------------------------------------------------
    # 读轨迹：打印算子数与步数，并计算文件哈希以便复现对比
    print(f"Loading trace: {trace_path}")
    manifest, records = load_trace(trace_path)
    trace_hash = sha256_file(trace_path)
    print(f"  {len(records)} operators, {manifest.get('num_steps', '?')} steps")
    print(f"  SHA-256: {trace_hash[:16]}...")

    # ------------------------------------------------------------------
    # Build tile map
    # ------------------------------------------------------------------
    # 建 tile 映射：把每个权重切成 64×64 之类的分块，得到访问序列
    print("Building tile map...")
    mapper = TileMapper()
    mapper.build_tile_trace(records)
    stats = mapper.stats()
    print(f"  {stats['total_unique_weights']} weights → {stats['total_unique_tiles']} tiles → "
          f"{len(mapper.tile_accesses)} tile accesses")

    # ------------------------------------------------------------------
    # Wire backends
    # ------------------------------------------------------------------
    # 接线后端：光子 + 电子，拿到调度器用的成本函数与共享缓存
    print("Initializing backends...")
    sim, elec, photonic_fn, electronic_fn, kernel_cache = build_backends(
        simphony_subprocess=args.simphony_subprocess,
    )
    arch = sim.architecture_cost()
    print(f"  SimPhony: MRR={arch['mrr_count']}  Area={arch['total_area_um2']:.0f} um^2")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    # 按 --sweep 分派到对应扫描；否则跑单次并写完整报告
    if args.sweep == "capacity":
        sweep_capacity(records, mapper, mapper.tile_accesses,
                       photonic_fn, electronic_fn, output_dir)
    elif args.sweep == "pprog":
        sweep_pprog(records, mapper, mapper.tile_accesses,
                    photonic_fn, electronic_fn, output_dir)
    elif args.sweep == "pipeline":
        sweep_pipeline(records, mapper, mapper.tile_accesses,
                       photonic_fn, electronic_fn, output_dir)
    else:
        # 单次运行
        print(f"\nSingle run: strategy={args.strategy}  cap={args.capacity}  "
              f"P_prog={args.pprog}  pipeline={args.pipeline}\n")
        t0 = time.perf_counter()
        sched = run_one(
            records, mapper, mapper.tile_accesses,
            strategy=args.strategy, total_slots=args.capacity,
            program_parallelism=args.pprog, pipeline_mode=args.pipeline,
            photonic_latency_fn=photonic_fn, electronic_latency_fn=electronic_fn,
        )
        elapsed = time.perf_counter() - t0
        # 写出完整诊断报告（run_id 含 策略/容量/编程并行度，便于区分）
        write_run_report(
            output_dir, f"joint_{args.strategy}_c{args.capacity}_p{args.pprog}",
            manifest, records, sim, kernel_cache, sched, mapper, elec,
        )
        print(f"\n  Run time: {elapsed:.1f}s")
        print(f"  Report:   {output_dir}")

    print(f"  Simulated photonic kernel signatures: {len(kernel_cache)}")

    # Copy trace for reproducibility
    # 把输入轨迹复制到输出目录，作为本次结果的复现依据
    import shutil
    shutil.copy2(trace_path, output_dir / "input_trace.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
