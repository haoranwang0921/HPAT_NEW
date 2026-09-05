# ============================================================================
# 模块说明：SRAM 管理器的单元测试
#   SRAM（Static Random-Access Memory，静态随机存取存储器）在光子加速器中
#   充当"片上缓存"：算子需要的权重 tile 从内存搬进 SRAM 后才能被光子核
#   使用。SramManager 负责管理"哪些 tile 当前在 SRAM 里、满了淘汰谁"。
#   本测试验证：
#     1. 构造参数校验（容量、tile 大小、带宽、DMA 延迟的合法性）
#     2. lookup_tile 的命中 / 未命中 / LRU 淘汰
#     3. 按阶段释放（phase release）与 mark_tile_ready（标记就绪）
#     4. 冷未命中统计、命中率、reset
#     5. 工厂函数 make_sram_manager 与多种置换策略（policy）的语义
#   "冒烟测试"（smoke tests）指只做最基本的功能性验证、确认主流程不崩。
# ============================================================================
"""
Unit tests for SramManager and its residency policies.

Covers:
  1. Constructor validation
  2. lookup_tile hit/miss and eviction
  3. Phase release and mark_tile_ready
  4. cold_misses / hit_rate / reset
  5. Factory and policy semantics
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sram_manager import (
    SramManager,
    make_sram_manager,
    LruSramPolicy,
    PhaseAwareSramPolicy,
    RolloutAwareSramPolicy,
    BeladyOptimalSramPolicy,
)
from schema import TileAccess


def _raises(exc_type, fn):
    """断言 fn() 会抛出指定类型的异常。

    作用：把"期望某操作失败"的测试意图写成简洁布尔表达式。
    参数：
        exc_type: 期望抛出的异常类型。
        fn: 零参数可调用对象，通常用 lambda 包装。
    返回值：
        bool：若 fn() 抛出了 exc_type 异常返回 True，否则返回 False。
    """
    try:
        fn()
        return False
    except exc_type:
        return True


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_validation():
    """验证构造参数的合法性校验。

    验证重点：总容量、tile 大小必须为正；tile 不能比总容量还大；
    HBM（高带宽内存）带宽必须为正；DMA（直接内存访问）固定延迟不能为负。
    关键断言：这些非法输入都应抛出 ValueError。
    """
    assert _raises(ValueError, lambda: SramManager(total_bytes=0))
    assert _raises(ValueError, lambda: SramManager(total_bytes=-1))
    assert _raises(ValueError, lambda: SramManager(tile_bytes=0))
    assert _raises(ValueError, lambda: SramManager(total_bytes=100, tile_bytes=200))
    assert _raises(ValueError, lambda: SramManager(hbm_bandwidth_bytes_per_s=0))
    assert _raises(ValueError, lambda: SramManager(dma_fixed_latency_s=-1.0))


def test_capacity_tiles_floor_division():
    """验证 capacity_tiles 采用向下取整（floor）计算。

    验证重点：capacity_tiles = total_bytes // tile_bytes；
    多出不足以放下一个 tile 的字节会被丢弃（向下取整）。
    关键断言：8192 字节容量 → 2 个 tile；8192+100 字节仍 → 2 个 tile。
    """
    mgr = SramManager(total_bytes=2 * 4096)
    assert mgr.capacity_tiles == 2
    mgr2 = SramManager(total_bytes=2 * 4096 + 100)  # 100 bytes extra → floor
    assert mgr2.capacity_tiles == 2


# ---------------------------------------------------------------------------
# lookup_tile hit / miss
# ---------------------------------------------------------------------------

def test_lookup_first_miss():
    """验证首次访问必然未命中（miss），且触发 DMA 搬运。

    验证重点：SRAM 里没有的 tile 需要从内存搬运（DMA），因此
    dma_latency_s > 0；ready_time_s 为 0.0 表示搬运可立即开始。
    关键断言：hit 为 False、dma_latency_s > 0、misses 统计为 1。
    """
    mgr = SramManager(total_bytes=4096)
    r = mgr.lookup_tile("t0", 0.0)
    assert r["hit"] is False
    assert r["dma_latency_s"] > 0
    assert r["ready_time_s"] == 0.0
    assert r["evicted_tile_ids"] == []
    assert mgr.stats["misses"] == 1
    assert mgr.stats["hits"] == 0


def test_lookup_second_hit():
    """验证同一 tile 第二次访问命中且无需 DMA。

    验证重点：tile 已在 SRAM 中，第二次访问直接命中，DMA 延迟为 0。
    关键断言：hit 为 True、dma_latency_s == 0.0、hits 统计为 1。
    """
    mgr = SramManager(total_bytes=4096)
    mgr.lookup_tile("t0", 0.0)
    r = mgr.lookup_tile("t0", 1.0)
    assert r["hit"] is True
    assert r["dma_latency_s"] == 0.0
    assert mgr.stats["hits"] == 1
    assert mgr.stats["misses"] == 1


def test_lru_eviction():
    """验证 LRU（最近最少使用）淘汰策略。

    验证重点：容量为 2 个 tile，依次放入 t0、t1，再"触碰"t0 使其成为
    最新使用，于是 t1 变成最久未用；新来 t2 触发未命中且容量已满，
    被淘汰的应是 t1。
    关键断言：evicted_tile_ids == ["t1"]、evictions 统计为 1、
    驻留数 occupied 为 2（t0 + t2）。
    """
    mgr = SramManager(total_bytes=2 * 4096, policy=LruSramPolicy())
    mgr.lookup_tile("t0", 0.0)
    mgr.lookup_tile("t1", 1.0)
    # Touch t0 so t1 becomes LRU
    # 再访问一次 t0，使它成为"最近使用"，t1 变成最久未用
    mgr.lookup_tile("t0", 2.0)
    r = mgr.lookup_tile("t2", 3.0)  # miss → evict t1
    assert r["hit"] is False
    assert r["evicted_tile_ids"] == ["t1"]
    assert mgr.stats["evictions"] == 1
    assert mgr.occupied == 2  # t0 + t2


# ---------------------------------------------------------------------------
# Phase release and mark_tile_ready
# ---------------------------------------------------------------------------

def test_evict_phase_tiles():
    """验证按阶段释放（phase release）tile。

    验证重点：推理过程分 encode / dynamics 等阶段，一个阶段结束后可以
    主动释放该阶段的 tile。metadata 里记录 tile 所属阶段，evict_phase_tiles
    只清除指定阶段的 tile。
    关键断言：只释放了 "encode" 阶段的 t0、t1 保留；
    且阶段释放不计入 evictions（不是因容量不足而淘汰）。
    """
    mgr = SramManager(total_bytes=4 * 4096)
    mgr.lookup_tile("t0", 0.0, metadata={"phase": "encode"})
    mgr.lookup_tile("t1", 0.0, metadata={"phase": "dynamics"})
    evicted = mgr.evict_phase_tiles("encode")
    assert evicted == ["t0"]
    assert not mgr.contains("t0")
    assert mgr.contains("t1")
    # Phase release must NOT increment the eviction stat
    # 阶段释放属于主动清理，不应计入"因容量不足而被淘汰"的统计
    assert mgr.stats["evictions"] == 0


def test_mark_tile_ready():
    """验证 mark_tile_ready（标记 tile 已就绪）与前置条件校验。

    验证重点：把 t0 标记为在 5.0 秒就绪后，t=6.0 秒访问应命中且
    ready_time_s 更新为 5.0；对不存在的 tile 标记应抛 KeyError，
    就绪时间为负应抛 ValueError。
    关键断言：命中且 ready_time_s==5.0；两个非法调用都抛错。
    """
    mgr = SramManager(total_bytes=4096)
    mgr.lookup_tile("t0", 0.0)
    mgr.mark_tile_ready("t0", 5.0)
    r = mgr.lookup_tile("t0", 6.0)
    assert r["hit"] is True
    assert r["ready_time_s"] == 5.0
    assert _raises(KeyError, lambda: mgr.mark_tile_ready("missing", 1.0))
    assert _raises(ValueError, lambda: mgr.mark_tile_ready("t0", -1.0))


# ---------------------------------------------------------------------------
# Stats and lifecycle
# ---------------------------------------------------------------------------

def test_cold_misses_without_future_info():
    """验证没有"未来访问信息"时，所有未命中都被算作冷未命中。

    验证重点：register_future_info 提供未来访问计划，用于区分
    "冷未命中"（tile 从没进过 SRAM）与"容量冲突未命中"。
    若没有注册任何未来信息，则每次未命中都归为冷未命中。
    关键断言：stats["cold_misses"] == 1。
    """
    mgr = SramManager(total_bytes=4096)
    mgr.lookup_tile("t0", 0.0)
    # No register_future_info → every miss is a cold miss
    assert mgr.stats["cold_misses"] == 1


def test_hit_rate():
    """验证命中率（hit_rate）统计。

    验证重点：一未命中一命中共两次访问，命中率应为 0.5。
    关键断言：mgr.hit_rate == 0.5。
    """
    mgr = SramManager(total_bytes=2 * 4096)
    mgr.lookup_tile("t0", 0.0)  # miss
    mgr.lookup_tile("t0", 1.0)  # hit
    assert mgr.hit_rate == 0.5


def test_reset_clears_state():
    """验证 reset 清空全部状态。

    验证重点：访问一个 tile 后 reset，驻留数、未命中统计、命中率都应归零。
    关键断言：occupied==0、misses==0、hit_rate==0.0。
    """
    mgr = SramManager(total_bytes=4096)
    mgr.lookup_tile("t0", 0.0)
    mgr.reset()
    assert mgr.occupied == 0
    assert mgr.stats["misses"] == 0
    assert mgr.hit_rate == 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_make_sram_manager_factory():
    """验证工厂函数 make_sram_manager 能按名字创建正确的策略。

    验证重点：工厂根据 policy 字符串（如 "lru"、"rollout_aware"）创建
    对应的置换策略对象；total_mb 以兆字节为单位换算成 tile 容量，
    策略参数（如 rollout 的 eta）也会透传给策略对象。
    关键断言：策略 name() 正确、容量换算正确、eta 参数被保留。
    """
    m = make_sram_manager(total_mb=1.0, policy="lru")
    assert m.policy.name() == "lru"
    assert m.capacity_tiles == (1024 * 1024) // 4096
    m2 = make_sram_manager(total_mb=1.0, policy="rollout_aware", eta=3.0)
    assert m2.policy.name() == "rollout_aware"
    assert m2.policy.eta == 3.0


# ---------------------------------------------------------------------------
# Policy semantics (smoke tests)
# ---------------------------------------------------------------------------

def _accesses(*specs):
    """把简写参数构造成 TileAccess 对象列表。

    每个 spec 依次是 (tile_id, phase, order, next_use, remaining)，
    即 tile 名、所属阶段、访问顺序、下次使用距离、剩余使用次数。
    作用：让后续测试注册"未来访问信息"时写起来更简洁。
    参数：
        specs: 零个或多个五元组。
    返回值：
        list[TileAccess]：转换后的访问记录列表。
    """
    return [
        TileAccess(tile_id, "op", phase, 0, order, next_use, remaining)
        for tile_id, phase, order, next_use, remaining in specs
    ]


def test_belady_evicts_furthest_next_use():
    """验证 Belady 最优算法会淘汰"下次使用最远"的 tile（冒烟测试）。

    验证重点：Belady 利用"未来访问计划"，容量为 2 时先放 t0、t1；
    新来 t2 需要腾位，按"下次使用距离最大者先淘汰"原则，
    t0 的 next_use=100 > t1 的 50，因此淘汰 t0。
    关键断言：evicted_tile_ids == ["t0"]。
    """
    mgr = SramManager(total_bytes=2 * 4096, policy=BeladyOptimalSramPolicy())
    mgr.register_future_info(_accesses(
        ("t0", "dynamics", 0, 100, 1),
        ("t1", "dynamics", 1, 50, 1),
        ("t2", "dynamics", 2, 999, 1),
    ))
    mgr.lookup_tile("t0", 0.0)
    mgr.lookup_tile("t1", 0.0)
    r = mgr.lookup_tile("t2", 0.0)  # miss → evict furthest next_use (t0: 100 > 50)
    assert r["evicted_tile_ids"] == ["t0"]


def test_policy_names():
    """验证每种置换策略的 name() 返回值（冒烟测试）。

    验证重点：四种策略应分别返回约定的策略名，
    供工厂函数按名字匹配。
    关键断言：四个 name() 依次等于预期字符串。
    """
    assert LruSramPolicy().name() == "lru"
    assert PhaseAwareSramPolicy().name() == "phase_aware"
    assert RolloutAwareSramPolicy().name() == "rollout_aware"
    assert BeladyOptimalSramPolicy().name() == "belady_optimal"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 允许直接 `python test_sram_manager.py` 运行：手写极简测试驱动，
    # 逐个执行测试函数并统计通过/失败数。
    import traceback

    tests = [
        ("constructor_validation", test_constructor_validation),
        ("capacity_tiles_floor_division", test_capacity_tiles_floor_division),
        ("lookup_first_miss", test_lookup_first_miss),
        ("lookup_second_hit", test_lookup_second_hit),
        ("lru_eviction", test_lru_eviction),
        ("evict_phase_tiles", test_evict_phase_tiles),
        ("mark_tile_ready", test_mark_tile_ready),
        ("cold_misses_without_future_info", test_cold_misses_without_future_info),
        ("hit_rate", test_hit_rate),
        ("reset_clears_state", test_reset_clears_state),
        ("make_sram_manager_factory", test_make_sram_manager_factory),
        ("belady_evicts_furthest_next_use", test_belady_evicts_furthest_next_use),
        ("policy_names", test_policy_names),
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
