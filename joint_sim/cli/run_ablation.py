#!/usr/bin/env python
"""Experiment 2 system ablation: A (mapping scope) x B (pipeline/parallelism).

Two orthogonal axes, each isolating exactly one factor (paired design):

A axis - photonic mapping scope (fixed: serial, P_prog=96)
    M0  no mapping (all-electronic anchor)
    M1  Dynamics-only fixed-weight Linear
    M2  Context + Dynamics
    M3  all fixed-weight Linear (Encode/Context/Dynamics/Decode)

B axis - scheduling/parallelism (fixed: M3 mapping)
    B1  parallel programming P_prog in [1, 4, 8, 16, 32, 64, 96]
    B2  pipeline modes serial / dma_program_overlap / triple_pipeline

Weights are assumed resident (infinite SRAM): every tile access programs only,
with no HBM DMA on the critical path.  Weight-supply architecture (SRAM
capacity / policy / MRR mode) is studied separately by run_experiment4.py.

Usage (from repository root)::

    python joint_sim/cli/run_ablation.py
    python joint_sim/cli/run_ablation.py --mapping m3 --no-b

【中文说明】
这是"实验2"的系统级消融实验（消融实验 = 每次去掉一个变量，看它对结果的影响）。
消融沿着两条互相垂直的轴进行，每次只改变一个因素（配对设计，便于归因）：

A 轴——光子映射范围（把多少算子派给光子阵列；固定串行调度、编程并行度 96）：
  M0 完全不映射：一切走电子后端（作为"全电子"的基准锚点）
  M1 只映射 Dynamics（自回归动态预测）阶段里的固定权重 Linear
  M2 映射 Context（上下文）+ Dynamics 两个阶段
  M3 映射所有阶段的固定权重 Linear（编码/上下文/动态/解码全映射）

B 轴——调度与并行方式（固定 M3 全映射）：
  B1 扫描编程并行度 P_prog（1~96，即一次可以同时编程多少个微环）
  B2 对比三种流水线模式：串行 / DMA 与编程重叠 / 三重流水线

重要前提假设：本实验假设所有权重常驻 SRAM（无限容量），所以每次 tile（权重分块）
访问只有"编程开销"，关键路径上没有 HBM 的 DMA（内存搬运）开销。而"SRAM 容量 /
驻留策略 / MRR 复用"这些权重供给架构的问题，由另一个脚本 run_experiment4.py 单独研究。

运行方式（在仓库根目录下）：
    python joint_sim/cli/run_ablation.py                       # 跑全部分组
    python joint_sim/cli/run_ablation.py --mapping m3 --no-b   # 只跑 M3 映射、跳过 B 轴

产出文件（默认写入 results/ablation/）：
- ablation_summary.csv   每次运行的逐行结果（端到端时延、覆盖度、编程次数等）
- ablation_summary.json  带 A 轴瀑布图、B1 编程并行度扫描、B2 流水线模式对比的结构化汇总
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

# 把项目根目录加进搜索路径，方便 import joint_sim 的模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.op_classifier import classify_operator as photonic_classifier
from joint_sim.scheduler import ResourcePool, ResourceScheduler
from joint_sim.tile_mapper import TileMapper
from joint_sim.trace_io import load_manifest_jsonl


DATAFLOW = "weight_stationary"  # 光子矩阵乘的数据流方式：权重驻留（权重不动、数据流动）


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
# 轨迹加载区：读入 LPWM 算子执行轨迹，并按实验分组筛选取舍算子。

def sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 哈希（分块读取，避免大文件占满内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 每次读 1MB 分块喂给哈希器，防止一次读入整个大轨迹文件
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(trace_path: Path):
    """Return (manifest, records) from a manifest-prefixed JSONL trace.

    中文说明：轨迹文件是"首行带清单（manifest 元信息）+ 后续每行一个算子"
    的 JSONL 格式。load_manifest_jsonl 负责解析，返回 (元信息字典, 算子记录列表)。
    """
    return load_manifest_jsonl(trace_path)


def scope_records(records: List[dict], scope: str) -> List[dict]:
    """Return a copy with out-of-scope Linear marked non-static.

    scope: 'm0' | 'm1' | 'm2' | 'm3'
      m0  none                m1  dynamics
      m2  context+dynamics    m3  all phases
    The standard photonic_classifier then naturally excludes them.

    中文说明：这是实现"映射范围"消融的关键函数。它不动原轨迹，而是拷贝一份，
    把"本分组不该上光子阵列"的 Linear 算子的 weight_static 标记改成 False。
    这样分类器（photonic_classifier）看到 weight_static=False 就会自动把这些
    算子排除在光子后端之外，让它们走电子后端。
    参数：scope 是 M0~M3 之一。返回一份筛选后的记录副本。
    """
    out = []
    for r in records:
        r2 = dict(r)  # 浅拷贝，避免污染原始记录
        # 只对"固定权重 Linear"做取舍（这类算子才可能上光子阵列）
        if r2["op_type"] == "Linear" and r2["weight_static"]:
            phase = r2.get("phase", "")
            # 查表决定该算子是否保留光子资格：M0 全排除、M1 仅 dynamics、
            # M2 为 context+dynamics、M3 全部保留
            keep = {"m0": False, "m1": phase == "dynamics",
                    "m2": phase in ("context", "dynamics"), "m3": True}[scope]
            if not keep:
                r2["weight_static"] = False  # 剥夺光子资格 → 交给电子后端
        out.append(r2)
    return out


# ---------------------------------------------------------------------------
# Photonic kernel cost cache (SimPhony, with honest fallback)
# ---------------------------------------------------------------------------
# 光子内核成本缓存区：把"每个 (M,K,N,位宽) 组合在光子阵列上跑一次的
# 时延/能耗"交给真实的 SimPhony 后端算，并按签名缓存，避免重复计算。

def build_photonic_latency_fn(records: List[dict]):
    """Return real SimPhony stage costs; initialization errors are fatal.

    中文说明：
    为轨迹里所有"可上光子阵列的固定权重 Linear"请求 SimPhony 的 kernel_cost
    （即这个矩阵乘在光子阵列上的分段时延/能耗成本），按签名
    (M, K, N, input_bits, weight_bits, output_bits) 缓存，返回一个查缓存用的
    latency_fn 闭包。若初始化 SimPhony 失败则直接抛错（诚实失败，不静默兜底）。
    返回值：(latency_fn, "simphony")，其中 latency_fn(rec) 返回该算子的成本字典。
    """
    from joint_sim.simphony_backend import SimPhonyBackend

    backend = SimPhonyBackend(use_subprocess=False)
    cache: Dict[tuple, dict] = {}
    for rec in records:
        # 只关心固定权重 Linear（这类才会上光子阵列）
        if rec["op_type"] != "Linear" or not rec["weight_static"]:
            continue
        sig = (int(rec["M"]), int(rec["K"]), int(rec["N"]),
               int(rec["input_bits"]), int(rec["weight_bits"]),
               int(rec["output_bits"]))
        if sig not in cache:  # 相同形状只算一次
            cache[sig] = backend.kernel_cost(
                M=sig[0], K=sig[1], N=sig[2],
                input_bits=sig[3], weight_bits=sig[4],
                output_bits=sig[5], dataflow=DATAFLOW,
            )

    def latency_fn(rec: dict) -> dict:
        # 根据算子的形状签名查缓存；查不到说明前面的预计算有遗漏，直接报错
        sig = (int(rec["M"]), int(rec["K"]), int(rec["N"]),
               int(rec["input_bits"]), int(rec["weight_bits"]),
               int(rec["output_bits"]))
        if sig not in cache:
            raise KeyError(f"missing SimPhony kernel cost for signature {sig}")
        return cache[sig]

    return latency_fn, "simphony"


# ---------------------------------------------------------------------------
# Single-config run
# ---------------------------------------------------------------------------
# 单配置运行区：给定一个具体配置（映射范围、编程并行度、流水线模式），
# 通过 DAG 资源调度器完整仿真一遍，返回统计结果字典。

def run_config(
    name: str,
    axis: str,
    records: List[dict],
    tile_mapper: TileMapper,
    photonic_latency_fn: Callable,
    electronic_backend: ElectronicBackend,
    p_prog: int,
    pipeline_mode: str,
) -> dict:
    """Run one config through the DAG resource scheduler.

    records must already be scope-filtered via scope_records().
    Weights are assumed resident (infinite SRAM), so only programming cost is
    modelled; there is no HBM DMA on the critical path.

    中文说明：把一组合格的算子记录（已按映射范围筛选过）喂给 DAG 资源调度器
    仿真执行，返回本配置的端到端结果。前提假设同模块文档：权重常驻 SRAM，
    只建模编程开销，关键路径上无 HBM DMA。
    参数：
      name          本次运行的名字（如 "m1_dynamics"）
      axis          属于哪条轴（"A" / "B1" / "B2"）
      records       已筛选的记录列表
      tile_mapper   tile（权重分块）映射器，决定权重怎么切分、怎么驻留
      photonic_latency_fn  查询光子成本用的函数
      electronic_backend   电子后端（提供算子时延估计）
      p_prog        编程并行度（同时编程的微环数）
      pipeline_mode 流水线模式（serial / dma_program_overlap / triple_pipeline）
    返回值：字典，含端到端时延、编程/ DMA 时延与能耗、光子/电子算子数、覆盖度等。
    """
    # 统计本组记录里有多少算子会被分类为"光子可执行"
    photonic_calls = sum(
        1 for r in records
        if photonic_classifier(r.get("op_type", ""), r.get("op_role", "unknown"),
                               r.get("weight_static", False))["eligible"]
    )
    has_photonic = photonic_calls > 0

    # 构造资源池：若有光子算子就配 1 个光子核 + 1 路 DAC/ADC；编程通道数 = P_prog
    pool = ResourcePool(
        num_photonic_cores=1 if has_photonic else 0,
        num_dac_channels=1 if has_photonic else 0,
        num_adc_channels=1 if has_photonic else 0,
        num_program_lanes=max(p_prog, 1),
    )

    # 创建调度器：电子时延用电子后端算，光子时延用 SimPhony 成本查表
    sched = ResourceScheduler(
        resource_pool=pool,
        electronic_latency_fn=electronic_backend.operator_latency_s,
        photonic_latency_fn=photonic_latency_fn,
        pipeline_mode=pipeline_mode,
    )
    # 对整个算子序列调度执行（tile_accesses 提供每个权重分块的访问序列）
    sched.schedule_trace(
        records,
        classifier_fn=photonic_classifier,
        tile_mapper=tile_mapper,
        tile_accesses=tile_mapper.tile_accesses,
    )

    electronic_calls = len(records) - photonic_calls
    sched_summary = sched.summary()  # 调度器的汇总统计
    ss = sched_summary["streaming_stats"]  # 流式（编程/DMA）统计

    return {
        "name": name,
        "axis": axis,
        "p_prog": p_prog,
        "pipeline_mode": pipeline_mode,
        "e2e_latency_s": sched.end_to_end_latency_s,  # 端到端总时延（核心指标）
        "hit_rate": 0.0,  # 无限 SRAM 假设下没有缓存未命中，命中率恒为 0 的占位
        "programs": ss["programs"],  # 编程事件次数
        "programming_latency_s": ss["programming_latency_s"],  # 总编程时延
        "programming_energy_j": ss["programming_energy_j"],  # 总编程能耗
        "dma_latency_s": ss["dma_latency_s"],  # DMA（搬运权重）时延
        "photonic_calls": photonic_calls,
        "electronic_calls": electronic_calls,
        "coverage": photonic_calls / len(records) if records else 0.0,  # 光子覆盖度
        "event_count": sched_summary["total_events"],  # 调度事件总数
        "conservation_passed": all(c["passed"] for c in sched_summary["conservation_checks"]),
        # 守恒检查：调度器内部校验"事件守恒/能耗守恒"是否全部通过
    }


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
# 实验分组定义区。

A_SCOPES = [
    # A 轴四个映射范围分组：名字 / 代号 / 说明
    ("m0_electronic", "m0", "no mapping (all-electronic anchor)"),
    ("m1_dynamics", "m1", "Dynamics-only fixed-weight Linear"),
    ("m2_context_dynamics", "m2", "Context + Dynamics"),
    ("m3_all", "m3", "all fixed-weight Linear"),
]

B1_P_PROG = [1, 4, 8, 16, 32, 64, 96]  # B1 轴要扫描的编程并行度取值
B2_PIPELINES = ["serial", "dma_program_overlap", "triple_pipeline"]  # B2 轴三种流水线模式


def main() -> None:
    """命令行入口：依次执行 A 轴与 B 轴的每组消融，并写出汇总结果。

    中文说明：
    流程：解析参数 → 校验轨迹/输出目录 → 读轨迹 → 建 tile 映射 →
    跑 A 轴（映射范围）→ 跑 B 轴（编程并行度 + 流水线模式）→ 写 CSV/JSON。
    命令行参数：
      --trace    轨迹文件路径（默认 traces/lpwm_bair128_checkpoint_horizon30.jsonl）
      --output   输出目录（默认 results/ablation）
      --mapping  只跑某个 A 分组（m0/m1/m2/m3），默认 all 跑全 4 组
      --no-b     跳过 B 轴
      --overwrite 允许覆盖非空输出目录（默认拒绝，防止误覆盖）
    产出：ablation_summary.csv（每行一组结果）+ ablation_summary.json（结构化汇总）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", default="traces/lpwm_bair128_checkpoint_horizon30.jsonl",
        help="Manifest-prefixed LPWM Trace JSONL path",
    )
    parser.add_argument(
        "--output", "-o", default="results/ablation",
        help="Output directory",
    )
    parser.add_argument(
        "--mapping", choices=["m0", "m1", "m2", "m3", "all"], default="all",
        help="Run a single A-scope, or 'all' for the full A sweep (default all)",
    )
    parser.add_argument(
        "--no-b", action="store_true",
        help="Skip the B axis (P_prog + pipeline modes)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing files in an existing output directory",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    output = Path(args.output).resolve()
    # 轨迹文件必须存在
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    # 防御：非空输出目录默认拒绝覆盖，必须显式 --overwrite 才允许
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}. "
            "Use --overwrite for a deliberate replacement."
        )
    output.mkdir(parents=True, exist_ok=True)

    # 读取轨迹；建好电子后端与光子成本查询函数
    manifest, records = load_records(trace_path)
    electronic_backend = ElectronicBackend()
    photonic_latency_fn, cost_source = build_photonic_latency_fn(records)

    print(f"Trace: {trace_path}  ({len(records)} operators, "
          f"steps={manifest.get('num_steps', '?')})")
    print(f"Photonic cost source: {cost_source}")

    # 建立 tile 映射：把每个固定权重切分成 tile（分块），并统计各阶段的块数
    mapper = TileMapper()
    mapper.build_tile_trace(records)
    stats = mapper.stats()
    print(f"Tiles: {stats['total_unique_weights']} weights -> "
          f"{stats['total_unique_tiles']} tiles")
    # 分阶段打印 tile/权重数量，以及按每块 4096 字节估算的 SRAM 占用（KB）
    for phase in ["encode", "context", "dynamics", "decode"]:
        ps = stats["by_phase"].get(phase, {})
        if ps:
            print(f"  {phase}: {ps.get('unique_tiles', 0)} tiles, "
                  f"{ps.get('unique_weights', 0)} weights, "
                  f"{ps['unique_tiles'] * 4096 / 1024:.0f} KB"
                  if ps.get("unique_tiles") else "")

    runs: List[dict] = []  # 收集所有分组的运行结果

    # ---- A axis: mapping scope ----
    # --- A 轴：映射范围 ---
    # 如果 --mapping 指定了某个分组就只跑那一个，否则跑全部 4 组
    scopes = A_SCOPES if args.mapping == "all" else \
        [s for s in A_SCOPES if s[1] == args.mapping]
    for name, scope, note in scopes:
        scoped = scope_records(records, scope)  # 按分组筛选可上光子的算子
        # A 轴固定串行调度、编程并行度 96
        r = run_config(name, "A", scoped, mapper,
                       photonic_latency_fn, electronic_backend,
                       p_prog=96, pipeline_mode="serial")
        r["note"] = note
        runs.append(r)
        print(f"[A] {r['name']:<30} E2E={r['e2e_latency_s']:.6e}s  "
              f"coverage={r['coverage']:.1%}  programs={r['programs']}  "
              f"dma={r['dma_latency_s']:.6e}s")

    # ---- B axis: P_prog (B1) then pipeline modes (B2) at M3 ----
    # --- B 轴：先在 M3 全映射下扫编程并行度（B1），再对比流水线模式（B2）---
    if not args.no_b:
        m3 = scope_records(records, "m3")
        for p_prog in B1_P_PROG:
            r = run_config(f"b1_p_prog_{p_prog}", "B1", m3, mapper,
                           photonic_latency_fn, electronic_backend,
                           p_prog, "serial")
            runs.append(r)
            print(f"[B1] p_prog={p_prog:<3} E2E={r['e2e_latency_s']:.6e}s  "
                  f"prog_lat={r['programming_latency_s']:.6e}s  "
                  f"dma={r['dma_latency_s']:.6e}s")
        for pipe in B2_PIPELINES:
            r = run_config(f"b2_pipeline_{pipe}", "B2", m3, mapper,
                           photonic_latency_fn, electronic_backend,
                           p_prog=96, pipeline_mode=pipe)
            runs.append(r)
            print(f"[B2] {pipe:<24} E2E={r['e2e_latency_s']:.6e}s")

    # ---- Output ----
    # --- 写出 CSV：一行一个分组 ---
    csv_path = output / "ablation_summary.csv"
    cols = [
        "name", "axis", "note", "p_prog", "pipeline_mode",
        "e2e_latency_s", "hit_rate", "programs", "programming_latency_s",
        "programming_energy_j", "dma_latency_s",
        "photonic_calls", "electronic_calls",
        "coverage", "event_count", "conservation_passed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for r in runs:
            writer.writerow({c: r.get(c, "") for c in cols})

    # 按名字索引运行结果，方便构建结构化汇总
    by_name = {r["name"]: r for r in runs}

    def _build_summary():
        """构建 JSON 汇总：A 轴瀑布图 + B1 并行度扫描 + B2 流水线对比。"""
        # A 轴：每个分组相对全电子基准（m0）的加速比
        a_waterfall = {}
        for a_name, scope, _note in A_SCOPES:
            r = by_name.get(a_name)
            if r is None:
                continue
            m0 = by_name.get("m0_electronic")
            a_waterfall[a_name] = {
                "scope": scope,
                "e2e_latency_s": r["e2e_latency_s"],
                "coverage": r["coverage"],
                # 加速比 = 全电子时延 ÷ 本组时延（>1 说明光子更快）
                "speedup_vs_m0": m0["e2e_latency_s"] / r["e2e_latency_s"]
                if m0 and r["e2e_latency_s"] else None,
                "dma_latency_s": r.get("dma_latency_s"),
            }

        # B1：各编程并行度下的时延
        b_prog = []
        for p in B1_P_PROG:
            key = f"b1_p_prog_{p}"
            if key in by_name:
                r = by_name[key]
                b_prog.append({
                    "p_prog": p, "e2e_latency_s": r["e2e_latency_s"],
                    "programs": r["programs"],
                })
        # B2：各流水线模式下的时延
        b_pipe = []
        for p in B2_PIPELINES:
            key = f"b2_pipeline_{p}"
            if key in by_name:
                r = by_name[key]
                b_pipe.append({
                    "pipeline": p, "e2e_latency_s": r["e2e_latency_s"],
                })

        return {
            "a_waterfall": a_waterfall,
            "b_p_prog_sweep": b_prog,
            "b_pipeline_modes": b_pipe,
        }

    ablation_summary = _build_summary()

    # 组装完整的 JSON 汇总：附上轨迹哈希、tile 统计与注意事项（caveats）
    summary: dict = {
        "ablation": "experiment2_ab_dual_axis",
        "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path),
                   "manifest": manifest, "records": len(records)},
        "photonic_cost_source": cost_source,
        "tile_stats": stats["by_phase"],
        "sram_streaming": ablation_summary,
        "caveats": [
            # 记录实验假设与建模近似，提醒读者解读结果时的边界
            "Infinite-SRAM assumption: weights are resident, so every tile "
            "access programs only, with no HBM DMA on the critical path.",
            "Scheduler splits each photonic kernel into fixed DAC 10% / compute 60% / "
            "ADC 30% fractions.",
            f"Photonic kernel costs from: {cost_source}.",
        ],
    }
    with (output / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved: {csv_path}")
    print(f"       {output / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
