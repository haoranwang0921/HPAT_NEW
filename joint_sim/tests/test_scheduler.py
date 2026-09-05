# ============================================================================
# 模块说明：调度器（scheduler）单元测试
#   调度器是"26世界模型光子加速"仿真器的核心：它读入算子轨迹（trace，
#   即一份算子的执行清单），按照算子之间的 DAG（有向无环图）依赖关系，
#   把每个算子分派到光子后端或电子后端，并给每个子事件安排开始时间，
#   最终统计端到端时延。
#   本测试验证调度器的四大类行为：
#     1. 事件按依赖顺序调度（同一资源上事件不重叠）
#     2. 端到端的轨迹仿真（覆盖 DAC/光子计算/ADC/电子计算四类事件）
#     3. DAG 依赖约束（独立分支可并行，汇合点必须等两个分支都完成）
#     4. 资源守恒校验（check_conservation）
# ============================================================================
"""
Scheduler tests: DAG dependencies, event ordering, resource conservation.

Covers:
  1. Event scheduling with dependencies
  2. End-to-end trace simulation
  3. Conservation checks
"""

import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler import ResourceScheduler, ResourcePool, EventType
from op_classifier import classify_operator


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

def test_scheduler_orders_events():
    """验证调度器能把一条简单轨迹调度出事件，并产出合法时延。

    验证重点：轨迹含一个静态权重的 Linear（线性层，可上光子后端）和一个
    Softmax（可上电子后端）；调度后必须产生事件、首个事件起始时间非负、
    端到端时延为正。
    关键断言：events 非空、events[0].start_time_s >= 0、end_to_end_latency_s > 0。
    """
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
        electronic_latency_fn=lambda r: 0.5e-6,
    )
    records = [
        {"op_id": "op_1", "order": 1, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": "w1", "M": 1, "K": 16, "N": 16},
        {"op_id": "op_2", "order": 2, "op_type": "Softmax", "op_role": "softmax",
         "weight_static": False},
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    assert len(scheduler.events) > 0
    assert scheduler.events[0].start_time_s >= 0
    assert scheduler.end_to_end_latency_s > 0


def test_scheduler_event_types():
    """验证光子算子的完整事件链包含全部四类事件类型。

    验证重点：一个较大的 Linear（M=257）会依次产生
    DAC_ENCODE（数模转换编码）、PHOTONIC_COMPUTE（光子矩阵乘）、
    ADC_DECODE（模数转换解码）三类光子事件；Softmax 产生
    ELECTRONIC_COMPUTE（电子计算）事件。
    关键断言：四类事件类型都出现在调度结果的事件集合中。
    """
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
        electronic_latency_fn=lambda r: 0.5e-6,
    )
    records = [
        {"op_id": "ph1", "order": 1, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": "w1", "M": 257, "K": 512, "N": 512},
        {"op_id": "el1", "order": 2, "op_type": "Softmax", "op_role": "softmax",
         "weight_static": False},
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    types = {e.event_type for e in scheduler.events}
    assert EventType.DAC_ENCODE in types
    assert EventType.PHOTONIC_COMPUTE in types
    assert EventType.ADC_DECODE in types
    assert EventType.ELECTRONIC_COMPUTE in types


def test_scheduler_conservation():
    # 验证同一资源上的事件互不重叠（最早开始时间约束）。
    # 验证重点：5 个串行 Linear 算子可能共用光子资源；把事件按资源分组后，
    # 同一资源内按开始时间排序，相邻事件必须满足"后一个的开始 >= 前一个的结束"。
    # 关键断言：所有相邻事件都满足该守恒条件。
    """Events must not overlap on same resource (earliest-start constraint)."""
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
        electronic_latency_fn=lambda r: 0.5e-6,
    )
    records = [
        {"op_id": f"op_{i}", "order": i, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": f"w{i}", "M": 1, "K": 16, "N": 16}
        for i in range(5)
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    by_resource = {}
    for event in scheduler.events:
        by_resource.setdefault(event.resource, []).append(event)
    for events in by_resource.values():
        events.sort(key=lambda event: event.start_time_s)
        for previous, current in zip(events, events[1:]):
            assert current.start_time_s >= previous.end_time_s - 1e-12


def test_scheduler_consumes_dag_dependencies():
    # 验证 DAG（有向无环图）依赖约束：相互独立的两个分支可同时开始，
    # 但汇合点（依赖它们的算子）必须等两个分支都完成。
    # 关键断言：q 与 mask 都从 0 开始；merge 的开始时间不早于二者各自的完成时间。
    """Independent branches overlap; their merge waits for both."""
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
        electronic_latency_fn=lambda r: 2e-6,
    )
    records = [
        {"op_id": "q", "order": 1, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": "wq", "M": 1, "K": 16,
         "N": 16, "dependencies": []},
        {"op_id": "mask", "order": 2, "op_type": "Softmax",
         "op_role": "softmax", "weight_static": False, "dependencies": []},
        {"op_id": "merge", "order": 3, "op_type": "Softmax",
         "op_role": "softmax", "weight_static": False,
         "dependencies": ["q", "mask"]},
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    q_start = min(e.start_time_s for e in scheduler.events if e.op_id == "q")
    mask_start = min(e.start_time_s for e in scheduler.events if e.op_id == "mask")
    merge_start = min(e.start_time_s for e in scheduler.events if e.op_id == "merge")
    assert q_start == mask_start == 0.0
    assert merge_start >= scheduler._op_complete["q"]
    assert merge_start >= scheduler._op_complete["mask"]


def test_scheduler_rejects_forward_dependency():
    """验证调度器会拒绝"前向依赖"（依赖一个尚未定义的算子）。

    验证重点：op "first" 声明依赖 "later"，但 "later" 按顺序排在后面，
    这种依赖指向未来、无法满足。正常情况下 schedule_trace 应抛 ValueError；
    若没有抛错，则用 AssertionError 主动让测试失败。
    """
    scheduler = ResourceScheduler(electronic_latency_fn=lambda r: 1e-6)
    records = [{
        "op_id": "first", "order": 1, "op_type": "Softmax",
        "op_role": "softmax", "weight_static": False,
        "dependencies": ["later"],
    }, {
        "op_id": "later", "order": 2, "op_type": "Softmax",
        "op_role": "softmax", "weight_static": False,
        "dependencies": [],
    }]
    try:
        scheduler.schedule_trace(records, classifier_fn=classify_operator)
    except ValueError:
        return
    raise AssertionError("Forward dependency must be rejected")


def test_scheduler_timeline_csv():
    """验证调度结果能导出为时间轴 CSV 文本。

    验证重点：单个 Linear 算子调度后，timeline_csv() 应生成
    表头 + 至少一行事件记录，且表头以 event_type 开头。
    关键断言：lines 多于 1 行、lines[0] 以 "event_type" 开头。
    """
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
    )
    records = [
        {"op_id": "op_1", "order": 1, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": "w1", "M": 257, "K": 512, "N": 512},
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    csv = scheduler.timeline_csv()
    lines = csv.strip().split("\n")
    assert len(lines) > 1  # header + at least one event
    assert lines[0].startswith("event_type")


def test_scheduler_conservation_checks():
    # 验证 check_conservation() 在简单串行轨迹上应全部检查项通过。
    # 关键断言：所有守恒检查项的 "passed" 字段都为 True。
    """check_conservation() returns all-pass on a simple serial trace."""
    scheduler = ResourceScheduler(
        photonic_latency_fn=lambda r: 1e-6,
        electronic_latency_fn=lambda r: 0.5e-6,
    )
    records = [
        {"op_id": f"op_{i}", "order": i, "op_type": "Linear", "op_role": "Q",
         "weight_static": True, "weight_id": f"w{i}", "M": 1, "K": 16, "N": 16}
        for i in range(3)
    ]
    scheduler.schedule_trace(records, classifier_fn=classify_operator)
    checks = scheduler.check_conservation()
    assert all(c["passed"] for c in checks), checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 允许直接 `python test_scheduler.py` 运行：手写极简测试驱动，
    # 逐个执行测试函数并统计通过/失败数。
    import traceback

    tests = [
        ("scheduler_orders_events", test_scheduler_orders_events),
        ("scheduler_event_types", test_scheduler_event_types),
        ("scheduler_conservation", test_scheduler_conservation),
        ("scheduler_consumes_dag_dependencies", test_scheduler_consumes_dag_dependencies),
        ("scheduler_rejects_forward_dependency", test_scheduler_rejects_forward_dependency),
        ("scheduler_timeline_csv", test_scheduler_timeline_csv),
        ("scheduler_conservation_checks", test_scheduler_conservation_checks),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        # 逐个执行：成功记 PASS，异常记 FAIL 并打印堆栈
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n  {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
