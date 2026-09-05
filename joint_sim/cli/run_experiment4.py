#!/usr/bin/env python
"""Experiment 4: SRAM-HBM scheduling strategies.

Tile-SRAM cache between HBM and MRR programming. Per-tile hit/miss with
pluggable eviction policies. A axis = capacity sweep, B axis = policy sweep.

A axis — SRAM capacity (fixed LRU, P_prog=96, M3):
  S48    48 MB — Dynamics (57.5 MB) does not fit, heavy eviction
  S64    64 MB — Dynamics barely fits, Context contention
  S80    80 MB — Dynamics + most of Context
  S96    96 MB — near 100% hit, upper bound

B axis — Policy (fixed 64 MB, P_prog=96, M3):
  lru            Least-recently-used (online baseline)
  phase_aware    Evict finished-phase tiles first
  rollout_aware  Score-based: retain high-reuse tiles
  belady_optimal Offline optimal lower bound (Belady's OPT)

Usage::

    python joint_sim/cli/run_experiment4.py
    python joint_sim/cli/run_experiment4.py --capacity 64 --policy rollout_aware
    python joint_sim/cli/run_experiment4.py --no-a  # B axis only

【中文说明】
这是"实验4"：研究 SRAM（片上静态内存）与 HBM（高带宽主存）之间怎么调度
权重 tile（把权重矩阵切成的分块）。

背景：光子阵列上要用的权重不能全塞进片上 SRAM（容量有限），大部分在 HBM 里。
仿真器在 HBM 与 MRR（微环，光计算器件）之间放了一个"tile-SRAM 缓存"：
一个权重 tile 首次使用时要花代价从 HBM 搬进 SRAM（DMA），之后的访问如果
tile 还留在 SRAM 里就命中（便宜），被挤出去就要再搬。本实验回答两个问题：
  1) SRAM 到底要多大？（A 轴：容量扫描）
  2) 容量有限时，用什么"逐出策略"（把哪些 tile 请出去）最划算？（B 轴：策略扫描）
另外 C 轴对比 MRR 的两种编程方式（每次用都重编程 vs 缺失时才编程），以及可选的
"定期刷新"模式（MRR 保持的权重会漂移，需要定期重新调谐）。

三条轴（都基于 M3 全映射、编程并行度 96）：
  A 轴：SRAM 容量扫描（固定 LRU 逐出策略）。因为 Dynamics 阶段的工作集约
       57.5 MB（一次 rollout 要用的全部权重总量），所以在 48~80 MB 附近密集采样。
  B 轴：逐出策略扫描（固定 64 MB）：LRU（最近最少用，在线基线）、
        phase_aware（先逐出已结束阶段的 tile）、rollout_aware（按复用得分保留）、
        belady_optimal（离线最优下限，需要"预知未来"的作弊参考）。
  C 轴：MRR 编程方式扫描。

运行方式（在仓库根目录下）：
    python joint_sim/cli/run_experiment4.py                  # 三条轴全跑
    python joint_sim/cli/run_experiment4.py --capacity 64 --policy rollout_aware
    python joint_sim/cli/run_experiment4.py --no-a           # 只跑 B/C 轴

产出文件（默认 results/experiment4/）：
- experiment4_summary.csv  每次运行一行（时延、命中率、SRAM/MRR 统计、能量分解）
- experiment4_summary.json 结构化汇总（A 容量扫描 / B 策略扫描 / C MRR 模式扫描）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

# 把项目根目录加进搜索路径，方便 import joint_sim 的模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.op_classifier import classify_operator as photonic_classifier
from joint_sim.scheduler import ResourcePool, ResourceScheduler
from joint_sim.sram_manager import (
    SramManager,  # SRAM 管理器：模拟 tile 缓存的存入/命中/逐出
    LruSramPolicy,  # LRU：逐出最久没用的 tile
    PhaseAwareSramPolicy,  # 阶段感知：优先逐出已完成阶段的 tile
    RolloutAwareSramPolicy,  # rollout 感知：按复用频率打分保留
    BeladyOptimalSramPolicy,  # Belady 最优：离线贪心下界（作弊参考）
)
from joint_sim.tile_mapper import TileMapper
from joint_sim.trace_io import load_manifest_jsonl


DATAFLOW = "weight_stationary"  # 光子矩阵乘的数据流方式：权重驻留

# ---- Axes ----
# Dense sampling around the measured Dynamics working-set knee (~57.5 MB).
# 在实测的 Dynamics 工作集拐点（约 57.5 MB）附近密集采样
A_CAPACITIES_MB = [48, 52, 56, 58, 60, 62, 64, 68, 72, 80]  # A 轴：SRAM 容量（MB）
B_POLICIES = ["lru", "phase_aware", "rollout_aware", "belady_optimal"]  # B 轴：逐出策略
MRR_PROGRAM_MODES = ["reprogram_each_use", "program_on_miss"]  # C 轴：MRR 编程方式

# 策略注册表：把策略名字映射到对应的策略类
POLICY_REGISTRY = {
    "lru": LruSramPolicy,
    "phase_aware": PhaseAwareSramPolicy,
    "rollout_aware": RolloutAwareSramPolicy,
    "belady_optimal": BeladyOptimalSramPolicy,
}


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
# 轨迹加载区

def sha256_file(path: Path) -> str:
    """计算文件 SHA-256（分块读取，避免大文件占满内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 每次读 1MB 分块
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(trace_path: Path):
    """读取"首行清单 + 每行一个算子"的 JSONL 轨迹，返回 (清单, 记录列表)。"""
    return load_manifest_jsonl(trace_path)


def scope_m3(records: List[dict]) -> List[dict]:
    """M3: all fixed-weight Linear are photonic.

    中文说明：实验4 固定使用 M3 全映射（所有固定权重 Linear 都上光子阵列），
    所以这里只是做一份浅拷贝，不筛掉任何算子（原样返回副本，避免污染原列表）。
    """
    return [dict(r) for r in records]


# ---------------------------------------------------------------------------
# Photonic cost cache
# ---------------------------------------------------------------------------
# 光子成本缓存区：按 (M,K,N,位宽) 签名缓存 SimPhony 的内核成本，避免重复计算

def build_photonic_latency_fn(records: List[dict]):
    """构建光子成本查询函数（带缓存），并附带暴露 cost_fn 与 backend 供能量核算。

    中文说明：
    预先把所有"可上光子的固定权重 Linear"的形状签名交给 SimPhony 后端算出
    kernel_cost 并缓存；返回的 latency_fn 对给定算子记录查缓存拿成本。
    和 run_ablation.py 里的版本不同：这里给返回的 latency_fn 额外挂了两个属性
    （.cost_fn 和 .backend），供 run_sram_config 做能量分解时直接使用。
    """
    from joint_sim.simphony_backend import SimPhonyBackend
    backend = SimPhonyBackend(use_subprocess=False)
    cache: Dict[tuple, dict] = {}
    for rec in records:
        if rec["op_type"] != "Linear" or not rec["weight_static"]:
            continue
        # 用 (M,K,N,输入位宽,权重位宽,输出位宽) 做签名
        sig = (int(rec.get("M") or 0), int(rec.get("K") or 0), int(rec.get("N") or 0),
               int(rec.get("input_bits", 8)), int(rec.get("weight_bits", 8)),
               int(rec.get("output_bits", 8)))
        if sig not in cache:  # 相同签名只算一次
            cache[sig] = backend.kernel_cost(
                M=sig[0], K=sig[1], N=sig[2],
                input_bits=sig[3], weight_bits=sig[4],
                output_bits=sig[5], dataflow=DATAFLOW,
            )

    def cost_fn(rec: dict) -> dict:
        # 按签名查缓存的成本字典；查不到直接报错
        sig = (int(rec.get("M") or 0), int(rec.get("K") or 0), int(rec.get("N") or 0),
               int(rec.get("input_bits", 8)), int(rec.get("weight_bits", 8)),
               int(rec.get("output_bits", 8)))
        if sig not in cache:
            raise KeyError(f"missing SimPhony kernel cost for signature {sig}")
        return cache[sig]

    def latency_fn(rec: dict) -> dict:
        return cost_fn(rec)

    # 把 cost_fn 和 backend 挂到 latency_fn 上，供能量分解逻辑调用
    latency_fn.cost_fn = cost_fn
    latency_fn.backend = backend
    return latency_fn, "simphony"


# ---------------------------------------------------------------------------
# Single-config run
# ---------------------------------------------------------------------------
# 单配置运行区：给定 SRAM 容量/逐出策略/MRR 编程方式等，完整仿真一次并统计。

def run_sram_config(
    name: str,
    axis: str,
    records: List[dict],
    tile_mapper: TileMapper,
    photonic_latency_fn,
    electronic_backend: ElectronicBackend,
    sram_capacity_mb: float,
    policy_name: str,
    p_prog: int = 96,
    mrr_program_mode: str = "reprogram_each_use",
    mrr_capacity_tiles: int = 96,
    mrr_refresh_interval_s: float = None,
) -> dict:
    """Run one SRAM config through the DAG scheduler.

    中文说明：
    与 run_ablation.run_config 类似，但额外带上 SRAM 管理器（带容量限制和
    逐出策略）和 MRR 驻留参数，仿真一次完整调度，返回该配置的详细统计。
    参数：
      name                运行名
      axis                所属轴（A/B/C）
      records             算子记录列表（M3 全映射）
      tile_mapper         tile 映射器
      photonic_latency_fn 光子成本查询函数（含 .cost_fn/.backend 附属属性）
      electronic_backend  电子后端
      sram_capacity_mb    SRAM 总容量（MB）
      policy_name         逐出策略名（查 POLICY_REGISTRY）
      p_prog              编程并行度
      mrr_program_mode    MRR 编程方式：每次用都重编程 / 缺失时才编程
      mrr_capacity_tiles  MRR 物理上能驻留多少个 64x64 tile 槽位
      mrr_refresh_interval_s  权重保持的刷新间隔（None 表示不刷新）
    返回值：结果字典（时延、SRAM/MRR 命中统计、能量分解等）。
    """
    # 统计可上光子的算子数
    photonic_calls = sum(
        1 for r in records
        if photonic_classifier(r.get("op_type", ""), r.get("op_role", "unknown"),
                               r.get("weight_static", False))["eligible"]
    )

    # 资源池：有光子算子就给 1 光子核 + 1 路 DAC/ADC；编程通道 = P_prog
    pool = ResourcePool(
        num_photonic_cores=1 if photonic_calls > 0 else 0,
        num_dac_channels=1 if photonic_calls > 0 else 0,
        num_adc_channels=1 if photonic_calls > 0 else 0,
        num_program_lanes=max(p_prog, 1),
    )

    # 从注册表取策略类（未知名字回退到 LRU），并创建 SRAM 管理器
    policy_cls = POLICY_REGISTRY.get(policy_name, LruSramPolicy)
    sram_mgr = SramManager(
        total_bytes=sram_capacity_mb * 1024 * 1024,  # MB → 字节
        policy=policy_cls(),
    )

    # 创建调度器：传入 SRAM 管理器与 MRR 驻留参数
    sched = ResourceScheduler(
        resource_pool=pool,
        electronic_latency_fn=electronic_backend.operator_latency_s,
        photonic_latency_fn=photonic_latency_fn,
        sram_manager=sram_mgr,
        mrr_program_mode=mrr_program_mode,
        mrr_capacity_tiles=mrr_capacity_tiles,
        mrr_refresh_interval_s=mrr_refresh_interval_s,
    )
    # 调度整个算子序列
    sched.schedule_trace(
        records,
        classifier_fn=photonic_classifier,
        tile_mapper=tile_mapper,
        tile_accesses=tile_mapper.tile_accesses,
    )

    electronic_calls = len(records) - photonic_calls
    sched_summary = sched.summary()
    ss = sched_summary["streaming_stats"]  # 流式（编程/DMA）统计
    sm = sched_summary["sram_manager_summary"] or {}  # SRAM 缓存统计
    mm = sched_summary["mrr_residency_summary"] or {}  # MRR 驻留统计

    # Invocation-weighted energy. Converter energy is event-local and can be
    # summed; laser/MRR hold are continuous and are integrated once over the
    # critical path below.
    # 能量核算（按"调用次数加权"）：DAC/ADC/MRR 调谐这类能量是"每次调用"的，
    # 可以直接累加；而激光器和 MRR 保持功率是"持续消耗"，只能在下面按
    # 端到端关键路径时长积分一次。
    cost_fn = getattr(photonic_latency_fn, "cost_fn", lambda _rec: {})
    photonic_energy = {
        "photonic_dynamic_energy_j": 0.0,  # 光子动态能耗（DAC/ADC/MRR 动态部分）
        "dac_energy_j": 0.0,
        "adc_energy_j": 0.0,
        "mrr_tuning_energy_j": 0.0,  # MRR 调谐（把微环调到目标波长）能耗
        "electronic_energy_j": 0.0,  # 电子后端能耗
    }
    # 逐算子累加：光子算子取 SimPhony 成本里的动态/转换/调谐能量，
    # 电子算子用电子后端的能量模型
    for rec in records:
        verdict = photonic_classifier(
            rec.get("op_type", ""), rec.get("op_role", "unknown"),
            rec.get("weight_static", False),
        )
        if verdict["eligible"]:
            cost = cost_fn(rec)
            for key in ("dynamic_energy_j", "dac_energy_j", "adc_energy_j",
                        "mrr_tuning_energy_j"):
                out_key = "photonic_dynamic_energy_j" if key == "dynamic_energy_j" else key
                photonic_energy[out_key] += float(cost.get(key, 0.0))
        else:
            lat = electronic_backend.operator_latency_s(rec)
            photonic_energy["electronic_energy_j"] += float(
                electronic_backend.operator_energy_j(rec, lat)
            )

    # 取激光器和 MRR 保持功率（持续功耗项，来自 SimPhony 后端）
    sim_backend = getattr(photonic_latency_fn, "backend", None)
    hold_power_w = 0.0  # MRR 保持功率（维持微环状态需持续供电）
    laser_power_w = 0.0  # 激光器功率
    if sim_backend is not None:
        hold_power_w = float(
            sim_backend.programming_cost().get("hold_power_w", 0.0)
        )
        laser_power_w = float(
            sim_backend.architecture_cost().get("laser_wall_plug_power_w", 0.0)
        )
    # MRR 调谐能量已包含在 SimPhony 的动态能量里，这里单独记录并从中扣除，
    # 以免重复计数（保持能量已单独作为 hold 项计算）
    embedded_tuning_j = photonic_energy["mrr_tuning_energy_j"]
    photonic_energy["raw_simphony_dynamic_energy_j"] = photonic_energy[
        "photonic_dynamic_energy_j"
    ]
    photonic_energy["embedded_mrr_tuning_excluded_j"] = embedded_tuning_j
    photonic_energy["photonic_dynamic_energy_j"] = max(
        0.0, photonic_energy["photonic_dynamic_energy_j"] - embedded_tuning_j
    )
    photonic_energy["mrr_tuning_energy_j"] = 0.0
    # 持续功耗项 = 功率 × 端到端总耗时（积分一次）
    e2e_s = float(sched.end_to_end_latency_s)
    photonic_energy["mrr_hold_energy_j"] = hold_power_w * e2e_s
    photonic_energy["laser_energy_j"] = laser_power_w * e2e_s
    photonic_energy["programming_energy_j"] = float(ss["programming_energy_j"])  # 调度编程事件能耗
    photonic_energy["hold_power_w"] = hold_power_w
    photonic_energy["laser_power_w"] = laser_power_w
    # ``photonic_dynamic_energy_j`` already contains the dynamic portions of
    # DAC/ADC/MRR devices. Do not add the component breakdown again here.
    # 总能量 = 光子动态 + 电子 + MRR 保持 + 激光 + 编程（转换器分量已含在动态里，
    # 不要再重复加）
    photonic_energy["total_energy_j"] = sum(
        photonic_energy[k] for k in (
            "photonic_dynamic_energy_j", "electronic_energy_j",
            "mrr_hold_energy_j", "laser_energy_j", "programming_energy_j",
        )
    )

    # 汇总所有统计字段返回
    return {
        "name": name,
        "axis": axis,
        "sram_capacity_mb": sram_capacity_mb,
        "policy": policy_name,
        "p_prog": p_prog,
        "mrr_program_mode": mrr_program_mode,
        "mrr_capacity_tiles": mm.get("capacity_tiles", mrr_capacity_tiles),
        "mrr_refresh_interval_s": mm.get("refresh_interval_s", mrr_refresh_interval_s),
        "e2e_latency_s": sched.end_to_end_latency_s,  # 端到端时延
        "hit_rate": sm.get("hit_rate", 0.0),  # SRAM 命中率
        "sram_hits": sm.get("hits", 0),
        "sram_misses": sm.get("misses", 0),
        "sram_cold_misses": sm.get("cold_misses", 0),  # 冷缺失（首次使用必缺）
        "sram_evictions": sm.get("evictions", 0),  # 逐出次数
        "sram_dma_latency_s": sm.get("dma_latency_s", 0.0),  # 从 HBM 搬 tile 的总时延
        "sram_dma_bytes": sm.get("dma_bytes", 0),
        "capacity_tiles": sm.get("capacity_tiles", 0),
        "mrr_hits": mm.get("hits", 0),
        "mrr_misses": mm.get("misses", 0),
        "mrr_evictions": mm.get("evictions", 0),
        "mrr_refreshes": mm.get("refreshes", 0),  # 刷新次数
        "mrr_hit_rate": mm.get("hit_rate", 0.0),  # MRR 命中率
        "programs": ss["programs"],  # 编程事件次数
        "programming_latency_s": ss["programming_latency_s"],
        "programming_critical_path_s": sched_summary.get(  # 编程在关键路径上的占比
            "programming_critical_path_s", ss["programming_latency_s"]
        ),
        "programming_energy_j": ss["programming_energy_j"],
        "dma_latency_s": ss["dma_latency_s"],
        "photonic_calls": photonic_calls,
        "electronic_calls": electronic_calls,
        "coverage": photonic_calls / len(records) if records else 0.0,
        **photonic_energy,  # 展开能量分解字段
        "event_count": sched_summary["total_events"],
        "conservation_passed": all(  # 守恒检查是否全部通过
            c["passed"] for c in sched_summary["conservation_checks"]
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """命令行入口：按 A/B/C 三条轴依次仿真并写出汇总结果。

    中文说明：
    A 轴（容量扫描）：固定 LRU 策略，扫 A_CAPACITIES_MB 里的每个容量。
    B 轴（策略扫描）：固定 --b-capacity 容量，对比 4 种逐出策略。
    C 轴（MRR 编程方式）：固定 --c-capacity 容量 + rollout_aware 策略，
       对比"每次重编程"和"缺失才编程"；若给 --mrr-refresh-us>0 还追加
       "缺失才编程+定期刷新"模式。
    命令行参数见各 parser.add_argument。产出 experiment4_summary.csv（每行
    一次运行）与 experiment4_summary.json（三条轴的结构化汇总）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", default="traces/lpwm_bair128_checkpoint_horizon30.jsonl",
    )
    parser.add_argument("--output", "-o", default="results/experiment4")
    parser.add_argument(
        "--capacity", type=float, default=0,
        help="Run a single SRAM capacity (MB) instead of the full A sweep.",
    )
    parser.add_argument(
        "--b-capacity", type=float, default=16.0,
        help="Fixed SRAM capacity (MB) for the B policy sweep.",
    )
    parser.add_argument(
        "--c-capacity", type=float, default=24.0,
        help="Fixed SRAM capacity (MB) for the C MRR-mode sweep.",
    )
    parser.add_argument(
        "--policy", choices=B_POLICIES, default="",
        help="Run a single policy instead of the full B sweep.",
    )
    parser.add_argument("--no-a", action="store_true",
                        help="Skip A axis (capacity sweep).")
    parser.add_argument("--no-b", action="store_true",
                        help="Skip B axis (policy sweep).")
    parser.add_argument("--no-c", action="store_true",
                        help="Skip C axis (MRR programming-mode sweep).")
    parser.add_argument(
        "--mrr-capacity-tiles", type=int, default=96,
        help="Physical MRR-resident 64x64 tile slots for program-on-miss.",
    )
    parser.add_argument(
        "--mrr-refresh-us", type=float, default=0.0,
        help="Optional refresh interval in microseconds; enables refresh mode in C.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--p-prog", type=int, default=96)
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    output = Path(args.output).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    # 防御：非空输出目录默认拒绝覆盖
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}. "
            "Use --overwrite for a deliberate replacement."
        )
    output.mkdir(parents=True, exist_ok=True)

    # 读轨迹、建电子后端与光子成本函数
    manifest, records = load_records(trace_path)
    electronic_backend = ElectronicBackend()
    photonic_latency_fn, cost_source = build_photonic_latency_fn(records)

    print(f"Trace: {trace_path}  ({len(records)} operators)")
    print(f"Photonic cost source: {cost_source}")

    # 建 tile 映射，按阶段统计 tile 数与所需 SRAM（每 tile 4KB）
    mapper = TileMapper()
    mapper.build_tile_trace(records)
    stats = mapper.stats()
    print(f"Tiles: {stats['total_unique_weights']} weights -> "
          f"{stats['total_unique_tiles']} tiles")
    for phase in ["encode", "context", "dynamics", "decode"]:
        ps = stats["by_phase"].get(phase, {})
        if ps:
            mb = ps["unique_tiles"] * 4096 / (1024 * 1024)
            print(f"  {phase}: {ps['unique_tiles']} tiles = {mb:.1f} MB")

    m3 = scope_m3(records)  # 实验4 固定 M3 全映射
    runs: List[dict] = []  # 收集所有运行结果

    # ---- A axis: capacity sweep ----
    # --- A 轴：SRAM 容量扫描（固定 LRU）---
    if not args.no_a:
        # --capacity 指定了就只跑单个容量，否则跑全列表
        capacities = [args.capacity] if args.capacity else A_CAPACITIES_MB
        print(f"\n{'='*60}")
        print(f"A axis: SRAM Capacity sweep (LRU, P_prog={args.p_prog})")
        print(f"{'='*60}")
        for cap_mb in capacities:
            r = run_sram_config(
                f"sram_{int(cap_mb)}mb_lru", "A", m3, mapper,
                photonic_latency_fn, electronic_backend,
                sram_capacity_mb=cap_mb, policy_name="lru",
                p_prog=args.p_prog, mrr_capacity_tiles=args.mrr_capacity_tiles,
            )
            r["note"] = f"{int(cap_mb)} MB SRAM, LRU"
            runs.append(r)
            print(f"[A] {r['name']:<22} E2E={r['e2e_latency_s']*1e3:6.2f}ms  "
                  f"hit={r['hit_rate']:.1%}  "
                  f"misses={r['sram_misses']:>6}  dma={r['sram_dma_latency_s']*1e3:.1f}ms")

    # ---- B axis: policy sweep ----
    # --- B 轴：逐出策略扫描（固定容量）---
    if not args.no_b:
        policies = [args.policy] if args.policy else B_POLICIES
        b_cap = args.b_capacity
        print(f"\n{'='*60}")
        print(f"B axis: Policy sweep (capacity={b_cap:.0f}MB, P_prog={args.p_prog})")
        print(f"{'='*60}")
        for pol in policies:
            r = run_sram_config(
                f"sram_{int(b_cap)}mb_{pol}", "B", m3, mapper,
                photonic_latency_fn, electronic_backend,
                sram_capacity_mb=b_cap, policy_name=pol,
                p_prog=args.p_prog, mrr_capacity_tiles=args.mrr_capacity_tiles,
            )
            r["note"] = f"{int(b_cap)} MB SRAM, {pol}"
            runs.append(r)
            print(f"[B] {r['name']:<30} E2E={r['e2e_latency_s']*1e3:6.2f}ms  "
                   f"hit={r['hit_rate']:.1%}  "
                   f"misses={r['sram_misses']:>6}  evict={r['sram_evictions']:>6}")

    # ---- C axis: MRR programming-mode sweep ----
    # --- C 轴：MRR 编程方式扫描（固定容量 + rollout_aware 策略）---
    if not args.no_c:
        c_modes = list(MRR_PROGRAM_MODES)
        if args.mrr_refresh_us > 0:
            c_modes.append("program_on_miss_with_refresh")  # 启用刷新时追加一个模式
        refresh_s = args.mrr_refresh_us * 1e-6 if args.mrr_refresh_us > 0 else None  # 微秒→秒
        print(f"\n{'='*60}")
        print(f"C axis: MRR mode sweep (capacity={args.c_capacity:.0f}MB, "
              f"MRR slots={args.mrr_capacity_tiles})")
        print(f"{'='*60}")
        for mode in c_modes:
            r = run_sram_config(
                f"sram_{int(args.c_capacity)}mb_rollout_aware_{mode}", "C", m3, mapper,
                photonic_latency_fn, electronic_backend,
                sram_capacity_mb=args.c_capacity, policy_name="rollout_aware",
                p_prog=args.p_prog, mrr_program_mode=mode,
                mrr_capacity_tiles=args.mrr_capacity_tiles,
                # 只有带 with_refresh 的模式才传刷新间隔
                mrr_refresh_interval_s=refresh_s if mode.endswith("with_refresh") else None,
            )
            r["note"] = f"{int(args.c_capacity)} MB SRAM, rollout_aware, {mode}"
            runs.append(r)
            print(f"[C] {mode:<30} E2E={r['e2e_latency_s']*1e3:6.2f}ms  "
                  f"programs={r['programs']:>6}  mrr_hit={r['mrr_hit_rate']:.1%}")

    # ---- Output ----
    # --- 写出 CSV：一行一次运行 ---
    csv_path = output / "experiment4_summary.csv"
    cols = [
        "name", "axis", "note", "sram_capacity_mb", "policy", "p_prog",
        "mrr_program_mode", "mrr_capacity_tiles", "mrr_refresh_interval_s",
        "e2e_latency_s", "hit_rate", "sram_hits", "sram_misses",
        "sram_cold_misses", "sram_evictions", "sram_dma_latency_s",
        "sram_dma_bytes", "capacity_tiles", "mrr_hits", "mrr_misses",
        "mrr_evictions", "mrr_refreshes", "mrr_hit_rate", "programs",
        "programming_latency_s", "programming_energy_j", "dma_latency_s",
        "programming_critical_path_s",
        "photonic_dynamic_energy_j", "raw_simphony_dynamic_energy_j",
        "embedded_mrr_tuning_excluded_j", "dac_energy_j", "adc_energy_j",
        "mrr_tuning_energy_j", "mrr_hold_energy_j", "laser_energy_j",
        "electronic_energy_j", "hold_power_w", "laser_power_w", "total_energy_j",
        "photonic_calls", "electronic_calls", "coverage",
        "event_count", "conservation_passed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for r in runs:
            writer.writerow({c: r.get(c, "") for c in cols})

    # 按名字索引，构建 JSON 汇总的三条轴数据
    by_name = {r["name"]: r for r in runs}

    # Summary
    # --- 汇总：A 轴容量扫描 ---
    a_capacity_sweep = []
    for cap_mb in (A_CAPACITIES_MB if not args.no_a else []):
        key = f"sram_{int(cap_mb)}mb_lru"
        if key in by_name:
            r = by_name[key]
            a_capacity_sweep.append({
                "capacity_mb": cap_mb,
                "capacity_tiles": r["capacity_tiles"],
                "e2e_latency_ms": round(r["e2e_latency_s"] * 1e3, 3),
                "hit_rate": r["hit_rate"],
                "misses": r["sram_misses"],
                "evictions": r["sram_evictions"],
                "total_energy_j": r["total_energy_j"],
            })

    # --- 汇总：B 轴策略扫描 ---
    b_policy_sweep = []
    b_cap_default = args.b_capacity
    for pol in (B_POLICIES if not args.no_b else []):
        key = f"sram_{int(b_cap_default)}mb_{pol}"
        if key in by_name:
            r = by_name[key]
            b_policy_sweep.append({
                "policy": pol,
                "e2e_latency_ms": round(r["e2e_latency_s"] * 1e3, 3),
                "hit_rate": r["hit_rate"],
                "misses": r["sram_misses"],
                "evictions": r["sram_evictions"],
                "total_energy_j": r["total_energy_j"],
            })

    # --- 汇总：C 轴 MRR 编程方式扫描 ---
    c_mrr_mode_sweep = []
    for mode in MRR_PROGRAM_MODES + (
        ["program_on_miss_with_refresh"] if args.mrr_refresh_us > 0 else []
    ):
        key = f"sram_{int(args.c_capacity)}mb_rollout_aware_{mode}"
        if key in by_name:
            r = by_name[key]
            c_mrr_mode_sweep.append({
                "mode": mode,
                "e2e_latency_ms": round(r["e2e_latency_s"] * 1e3, 3),
                "programs": r["programs"],
                "mrr_hit_rate": r["mrr_hit_rate"],
                "mrr_evictions": r["mrr_evictions"],
                "mrr_refreshes": r["mrr_refreshes"],
                "programming_energy_j": r["programming_energy_j"],
                "total_energy_j": r["total_energy_j"],
            })

    # 组装完整 JSON 汇总：附带轨迹信息、参数与注意事项（caveats）
    summary = {
        "experiment": "4_sram_hbm_scheduling",
        "architecture": "12x Tile-SRAM with managed residency between HBM and MRR",
        "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path),
                   "manifest": manifest, "records": len(records)},
        "photonic_cost_source": cost_source,
        "tile_stats": stats["by_phase"],
        "parameters": {
            "p_prog": args.p_prog,
            "mapping": "M3",
            "b_capacity_mb": args.b_capacity,
            "c_capacity_mb": args.c_capacity,
            "mrr_capacity_tiles": args.mrr_capacity_tiles,
            "mrr_refresh_us": args.mrr_refresh_us,
        },
        "a_capacity_sweep": a_capacity_sweep,
        "b_policy_sweep": b_policy_sweep,
        "c_mrr_mode_sweep": c_mrr_mode_sweep,
        "caveats": [
            # 记录建模假设，提醒解读结果时注意边界
            "SRAM capacity is the total across 12 tiles.",
            "Per-tile DMA on miss: 4KB / 512GB/s + 100ns = 108ns per tile.",
            "Programming: 1000ns per tile, 96 parallel lanes.",
            "MRR program-on-miss tracks a separate, LRU-managed physical MRR tile set; SRAM hit is not an MRR hit.",
            "Belady OPT requires complete future knowledge (offline lower bound).",
            f"Photonic kernel costs from: {cost_source}.",
            "Energy totals use invocation-weighted dynamic/electronic/programming energy; laser and MRR hold are integrated once over the E2E critical path.",
        ],
    }
    with (output / "experiment4_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    # Print comparison tables
    # --- 屏幕打印三张对比表 ---
    if a_capacity_sweep:
        print(f"\n{'─'*70}")
        print(f"A axis — Capacity sweep (LRU):")
        print(f"  {'Cap':>5}  {'Tiles':>6}  {'E2E(ms)':>8}  {'Hit%':>7}  "
              f"{'Misses':>7}  {'Evictions':>9}")
        for r in a_capacity_sweep:
            print(f"  {r['capacity_mb']:>4.0f}MB  {r['capacity_tiles']:>6}  "
                  f"{r['e2e_latency_ms']:>8.2f}  {r['hit_rate']:>6.1%}  "
                  f"{r['misses']:>7}  {r['evictions']:>9}")

    if b_policy_sweep:
        print(f"\n{'─'*70}")
        print(f"B axis — Policy sweep ({int(b_cap_default)} MB):")
        print(f"  {'Policy':<18}  {'E2E(ms)':>8}  {'Hit%':>7}  "
              f"{'Misses':>7}  {'Evictions':>9}")
        for r in b_policy_sweep:
            print(f"  {r['policy']:<18}  {r['e2e_latency_ms']:>8.2f}  "
                  f"{r['hit_rate']:>6.1%}  {r['misses']:>7}  {r['evictions']:>9}")

    if c_mrr_mode_sweep:
        print(f"\nMRR programming modes ({int(args.c_capacity)} MB):")
        print(f"  {'Mode':<32}  {'E2E(ms)':>8}  {'Programs':>9}  {'MRR hit%':>9}")
        for r in c_mrr_mode_sweep:
            print(f"  {r['mode']:<32}  {r['e2e_latency_ms']:>8.2f}  "
                  f"{r['programs']:>9}  {r['mrr_hit_rate']:>8.1%}")

    print(f"\nSaved: {csv_path}")
    print(f"       {output / 'experiment4_summary.json'}")


if __name__ == "__main__":
    main()
