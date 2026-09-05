"""Focused Experiment 4 comparison at a single capacity (total_slots).

Runs the 4 residency strategies on the real LPWM BAIR-128 trace at one
capacity point, P_prog=1, serial pipeline. Prints a comparison table and
saves the rows to results/experiment4_cap<cap>/.

Usage::
    python joint_sim/cli/run_experiment4_cap384.py [--cap N] [--output DIR]

【中文说明】
这是实验4（SRAM 容量 / 权重驻留策略）的一个"聚焦版"变体：只在【一个】容量
点下，对比不同权重驻留策略（lru / belady_optimal / rollout_aware）的表现。
与主脚本 run_experiment4.py 的区别：
- 主脚本扫多档容量和多条策略轴；本脚本固定一个容量，专注"策略间对比"。
- 使用固定的简化分类器（只有固定权重 Linear 上光子）。
- 固定 P_prog=1、串行流水线，以便单独观察驻留策略的影响。
运行方式：
    python joint_sim/cli/run_experiment4_cap384.py --cap 384
产出：results/experiment4_cap<cap>/ 下的对比报告（SweepReport 写出的文件）。
"""
import sys, os, json, time
# 把项目根目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from joint_sim.tile_mapper import TileMapper
from joint_sim.sram_manager import make_sram_manager  # 按策略名创建 SRAM 管理器
from joint_sim.scheduler import ResourcePool, ResourceScheduler
from joint_sim.report import SweepReport  # 报告生成器
from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.cli.run_experiment4 import build_photonic_latency_fn  # 复用主脚本的光子成本函数

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE = os.path.join(ROOT, "traces", "lpwm_bair128_checkpoint_horizon30.jsonl")
CAPACITY = 384  # 默认 SRAM 容量（MB），命令行 --cap 可覆盖


def load_records(trace_path):
    """从轨迹文件读入算子记录列表（跳过第一行的 manifest 清单行）。"""
    records = []
    with open(trace_path, "r") as f:
        f.readline()  # manifest  # 第一行是清单元信息，跳过
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # 每行一个算子记录（JSON）
            # 只抽取仿真需要的字段（其余字段丢弃，避免多余内存）
            records.append({
                "op_id": obj.get("op_id", ""),
                "order": obj.get("order", 0),
                "op_type": obj.get("op_type", ""),
                "op_role": obj.get("op_role", ""),
                "phase": obj.get("phase", ""),
                "block_kind": obj.get("block_kind", ""),
                "layer_index": obj.get("layer_index"),
                "rollout_step": obj.get("rollout_step"),
                "call_index": obj.get("call_index", 0),
             "M": obj.get("M"), "K": obj.get("K"), "N": obj.get("N"),
             "input_bits": obj.get("input_bits", 8),
             "weight_bits": obj.get("weight_bits", 8),
             "output_bits": obj.get("output_bits", 8),
                "batch_repetitions": obj.get("batch_repetitions", 1),
                "weight_id": obj.get("weight_id"),
                "weight_static": obj.get("weight_static", False),
                "dependencies": tuple(obj.get("dependencies", [])),  # 依赖存成元组便于哈希
            })
    return records


def classifier(op_type, op_role, ws):
    """简化分类器：只有"固定权重 Linear"才有资格上光子阵列。"""
    if op_type == "Linear" and ws:
        return {"eligible": True, "reason": "static_weight_linear"}
    return {"eligible": False, "reason": "not_eligible"}


def main():
    """命令行入口：在指定容量下依次仿真 4 种策略并输出对比表。

    流程：解析 --cap/--output → 读轨迹 → 建 tile 映射 → 对每种策略
    （lru/belady_optimal/rollout_aware）创建 SRAM 管理器并完整调度仿真 →
    把每次运行记入 SweepReport → 打印表格 → 按命中率排名并对比
    rollout_aware 相对 LRU 的改进 → 写出报告文件。
    """
    import argparse
    ap = argparse.ArgumentParser(description="Compare 4 residency strategies at one capacity")
    ap.add_argument("--cap", type=float, default=96.0, help="SRAM capacity in MB (default 96)")
    ap.add_argument("--output", default=None, help="output directory")
    args = ap.parse_args()
    capacity = args.cap
    output = args.output or f"results/experiment4_cap{capacity}"

    # 读轨迹、建电子后端、建光子成本函数
    records = load_records(TRACE)
    print(f"Loaded {len(records)} operators")
    electronic_backend = ElectronicBackend()
    photonic_latency_fn, photonic_cost_source = build_photonic_latency_fn(records)
    print(f"  Photonic cost source: {photonic_cost_source}")

    # 建立 tile 映射，得到权重分块的访问序列
    mapper = TileMapper()
    mapper.build_tile_trace(records)
    accesses = mapper.tile_accesses
    stats = mapper.stats()
    print(f"  {stats['total_unique_weights']} weights -> "
          f"{stats['total_unique_tiles']} tiles -> {len(accesses)} tile accesses")

    strategies = ["lru", "belady_optimal", "rollout_aware"]  # 待对比的驻留策略

    sweep = SweepReport(output)
    rows = []  # 收集每行结果用于排名
    header = (f"{'Strategy':<16} {'Cap':>5} {'HitRate':>8} {'Hits':>7} {'Misses':>7} "
              f"{'Evicts':>7} {'Programs':>8} {'E2E(s)':>12} {'ProgLat(s)':>12} "
              f"{'DMA(s)':>11} {'Time':>6}")
    print("\n" + header)
    print("-" * len(header))

    for name in strategies:
        t0 = time.perf_counter()
        # 按策略名字创建带容量上限的 SRAM 管理器
        sram_mgr = make_sram_manager(total_mb=capacity, policy=name)
        pool = ResourcePool(
            num_photonic_cores=1, num_dac_channels=1,
            num_adc_channels=1, num_program_lanes=1,  # 固定单核、编程并行度 1
        )
        sched = ResourceScheduler(
            resource_pool=pool,
            electronic_latency_fn=electronic_backend.operator_latency_s,
            photonic_latency_fn=photonic_latency_fn,
            sram_manager=sram_mgr,
            pipeline_mode="serial",  # 固定串行流水线，隔离策略本身的影响
        )
        # 完整调度一遍算子轨迹
        sched.schedule_trace(records, classifier_fn=classifier,
                             tile_mapper=mapper, tile_accesses=accesses)
        elapsed = time.perf_counter() - t0

        # 记录本次运行配置并写入报告
        config = {"strategy": name, "total_slots": capacity,
                  "program_parallelism": 1, "pipeline_mode": "serial"}
        sched_summary = sched.summary()
        sweep.add_run(config, sched_summary)
        sm = sched_summary["sram_manager_summary"]  # SRAM 统计（命中/缺失/逐出）
        ss = sched_summary["streaming_stats"]  # 流式统计（编程/DMA）
        rows.append({
            "strategy": name, "hit_rate": sm["hit_rate"],
            "hits": sm["hits"], "misses": sm["misses"],
            "evictions": sm["evictions"], "programs": ss["programs"],
            "e2e_s": sched.end_to_end_latency_s,
            "prog_lat_s": sched_summary.get("programming_critical_path_s", 0.0),
            "dma_lat_s": sm["dma_latency_s"],
        })
        # 打印该策略的一行结果
        print(f"{name:<16} {capacity:>5} {sm['hit_rate']:>7.1%} {sm['hits']:>7} "
              f"{sm['misses']:>7} {sm['evictions']:>7} {ss['programs']:>8} "
              f"{sched.end_to_end_latency_s:>12.4e} {sched_summary.get('programming_critical_path_s', 0.0):>12.4e} "
              f"{sm['dma_latency_s']:>11.4e} {elapsed:>5.0f}s")

    # 写出报告文件（CSV/JSON 等由 SweepReport 决定）
    sweep.write_all()

    # Rank by hit rate  # 按命中率排序打印排名
    print("\n=== Ranking by hit rate ===")
    for rank, r in enumerate(sorted(rows, key=lambda r: -r["hit_rate"]), 1):
        print(f"  {rank}. {r['strategy']:<16} {r['hit_rate']:>7.1%}")

    # Rollout-aware vs LRU improvement  # 量化 rollout_aware 相对 LRU 的命中率提升
    ra = next(r for r in rows if r["strategy"] == "rollout_aware")
    lru = next(r for r in rows if r["strategy"] == "lru")
    if lru["hit_rate"] > 0:
        imp = (ra["hit_rate"] - lru["hit_rate"]) / lru["hit_rate"]
        print(f"\nrollout_aware vs lru: {imp:+.1%} "
              f"({ra['hit_rate']:.3f} vs {lru['hit_rate']:.3f})")

    print(f"\nResults saved to {output}/")


if __name__ == "__main__":
    main()
