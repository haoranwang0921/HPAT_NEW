"""
Experiment 4 sweep on the real LPWM BAIR-128 trace (30-step rollout).

Runs 4 residency strategies x 5 capacity points = 20 configurations.

【中文说明】
这是实验4（SRAM 容量 / 驻留策略）在真实 LPWM BAIR-128 轨迹上的完整扫描版：
3 种驻留策略 × 5 档 SRAM 容量 = 15 个配置（模块 docstring 里写 4 种策略，
但实际列表只列了 3 种：lru / belady_optimal / rollout_aware）。每个配置都跑
一次完整调度仿真（30 步 rollout），结果写入 SweepReport 生成的报告目录
results/experiment4_real/。最后额外做两步分析：
- 按容量统计 LRU 策略下"每步时延"的增长速度（观察 rollout 越长时延怎么涨）；
- 在每档容量下对比三种策略的命中率。
运行方式：python joint_sim/cli/run_experiment4_real.py
产出：results/experiment4_real/ 下的报告文件 + 屏幕打印的对比表。
"""
import sys, json, time, os
# 把项目根目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from joint_sim.tile_mapper import TileMapper
from joint_sim.sram_manager import make_sram_manager
from joint_sim.scheduler import ResourcePool, ResourceScheduler
from joint_sim.report import SweepReport
from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.cli.run_experiment4 import build_photonic_latency_fn

# 轨迹文件路径（真实 LPWM BAIR-128、30 步 rollout 的导出轨迹）
TRACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "traces", "lpwm_bair128_checkpoint_horizon30.jsonl")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
# 加载轨迹：跳过第一行 manifest 清单，逐行解析算子记录
print("Loading trace...", flush=True)
records = []
with open(TRACE, "r") as f:
    f.readline()  # manifest  # 第一行是清单元信息，跳过
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)  # 每行一个算子记录（JSON）
        # 只抽取仿真需要的字段
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
            "dependencies": tuple(obj.get("dependencies", [])),
        })
print(f"  {len(records)} operators", flush=True)
# 建立电子后端与光子成本查询函数
electronic_backend = ElectronicBackend()
photonic_latency_fn, photonic_cost_source = build_photonic_latency_fn(records)
print(f"  photonic cost source: {photonic_cost_source}", flush=True)

# ---------------------------------------------------------------------------
# Tile map
# ---------------------------------------------------------------------------
# 建立 tile 映射：把权重切成分块，得到访问序列
print("Building tile map...", flush=True)
mapper = TileMapper()
mapper.build_tile_trace(records)
accesses = mapper.tile_accesses
stats = mapper.stats()
print(f"  {stats['total_unique_weights']} weights -> {stats['total_unique_tiles']} tiles -> {len(accesses)} tile accesses", flush=True)

def classifier(op_type, op_role, ws):
    """简化分类器：只有固定权重 Linear 才有资格上光子阵列。"""
    if op_type == "Linear" and ws:
        return {"eligible": True, "reason": "static_weight_linear"}
    return {"eligible": False, "reason": "not_eligible"}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
# 扫描定义：策略 × 容量
STRATEGIES = ["lru", "belady_optimal", "rollout_aware"]  # 3 种驻留策略
CAPACITIES = [24, 48, 96, 192, 384]  # SRAM capacity in MB  # 5 档容量（MB）

sweep = SweepReport("results/experiment4_real")
N = len(STRATEGIES) * len(CAPACITIES)  # 总配置数
print(f"\nSweep: {len(STRATEGIES)} strategies x {len(CAPACITIES)} capacities = {N} configs", flush=True)
print(f"{'Strategy':<16} {'Cap':>6} {'HitRate':>8} {'Hits':>8} {'Misses':>8} {'Evicts':>8} {'E2E(s)':>12} {'Prog(s)':>12} {'Time':>6}", flush=True)
print("-" * 106, flush=True)

total_start = time.perf_counter()
idx = 0
# 双重循环：每种策略 × 每档容量
for strategy_name in STRATEGIES:
    for cap in CAPACITIES:
        idx += 1
        t0 = time.perf_counter()
        # 按策略名创建带容量上限的 SRAM 管理器
        sram_mgr = make_sram_manager(total_mb=cap, policy=strategy_name)
        pool = ResourcePool(
            num_photonic_cores=1, num_dac_channels=1,
            num_adc_channels=1, num_program_lanes=1,  # 单核、编程并行度 1
        )
        sched = ResourceScheduler(
            resource_pool=pool,
            electronic_latency_fn=electronic_backend.operator_latency_s,
            photonic_latency_fn=photonic_latency_fn,
            sram_manager=sram_mgr,
            pipeline_mode="serial",  # 串行流水线
        )
        # 完整调度一遍算子轨迹
        sched.schedule_trace(records, classifier_fn=classifier,
                            tile_mapper=mapper, tile_accesses=accesses)
        elapsed = time.perf_counter() - t0

        # 记录本次运行配置并写入报告
        config = {"strategy": strategy_name, "total_mb": cap,
                  "program_parallelism": 1, "pipeline_mode": "serial"}
        sched_summary = sched.summary()
        sweep.add_run(config, sched_summary)

        sm = sched_summary["sram_manager_summary"]  # SRAM 命中/缺失/逐出统计
        hits, misses = sm["hits"], sm["misses"]
        # 打印一行进度结果
        print(f"  [{idx}/{N}] {strategy_name:<16} {cap:>6} {sm['hit_rate']:>7.1%} {hits:>8} {misses:>8} {sm['evictions']:>8} {sched.end_to_end_latency_s:>12.4e} {sched_summary.get('programming_critical_path_s', 0.0):>12.4e} {elapsed:>5.0f}s", flush=True)

total_elapsed = time.perf_counter() - total_start
print(f"\nTotal: {total_elapsed/60:.1f} min", flush=True)
sweep.write_all()  # 写出全部报告文件

# ---------------------------------------------------------------------------
# Per-step analysis
# ---------------------------------------------------------------------------
# 分析：LRU 策略下"每步时延"随 rollout 步数的增长斜率
print("\n=== Per-step latency growth: LRU at each capacity ===", flush=True)
for cap in CAPACITIES:
    runs = [r for r in sweep.to_dataframe()
            if r["strategy"] == "lru" and r["total_mb"] == cap]
    if runs:
        per_step = runs[0].get("per_step_latency_s", {})  # 每一步的累计时延字典
        steps = sorted(int(k) for k in per_step.keys())
        if len(steps) >= 2:
            first = per_step[str(steps[0])]
            last = per_step[str(steps[-1])]
            # 平均每步时延增量 =（最后步 - 第一步）÷ 步数差
            growth = (last - first) / max(1, steps[-1] - steps[0])
            print(f"  cap={cap:>5}: step_{steps[0]:02d}={first:.4e}s  step_{steps[-1]:02d}={last:.4e}s  growth/step={growth:.4e}s", flush=True)

# ---------------------------------------------------------------------------
# Policy comparison table
# ---------------------------------------------------------------------------
# 汇总表：每档容量下各策略的命中率，标出最优
print("\n=== Strategy comparison at each capacity ===", flush=True)
print(f"{'Cap':>6} {'LRU':>8} {'Belady':>8} {'Rollout':>8} {'Best':>8}", flush=True)
print("-" * 50, flush=True)
for cap in CAPACITIES:
    vals = {}
    for strat in STRATEGIES:
        runs = [r for r in sweep.to_dataframe()
                if r["strategy"] == strat and r["total_mb"] == cap]
        vals[strat] = runs[0]["hit_rate"] if runs else 0.0
    best = max(vals.values())  # 该容量下最高命中率
    print(f"  {cap:>6} {vals['lru']:>7.1%} {vals['belady_optimal']:>7.1%} {vals['rollout_aware']:>7.1%} {best:>7.1%}", flush=True)

print("\nDone.", flush=True)
