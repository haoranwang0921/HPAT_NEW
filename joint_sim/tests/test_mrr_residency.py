# ============================================================================
# 模块说明：MRR 驻留管理器的单元测试
#   MRR（Micro-Ring Resonator，微环谐振器）是光子计算芯片的核心器件：
#   把权重"烧录"（program）到微环上之后，该权重可以长期"驻留"
#   （residency）在芯片里，之后同一权重反复使用时就不用重新烧录，从而省时省电。
#   MrrResidencyManager 就是负责管理"哪些权重 tile 当前驻留在芯片上"的模块。
#   本测试逐一验证它的全部行为：
#     1. 构造参数校验（非法参数应报错）
#     2. 冷未命中（cold miss）/ 命中（hit）/ 刷新过期（refresh expiry）
#     3. record_program（记录烧录动作）
#     4. LRU（最近最少使用）淘汰策略
#     5. invalidate / invalidate_many（显式失效）
#     6. hit_rate（命中率）与 reset（重置）
# ============================================================================
"""
Unit tests for MrrResidencyManager.

Covers:
  1. Constructor validation
  2. Cold miss / hit / refresh expiry
  3. record_program
  4. LRU eviction
  5. invalidate / invalidate_many
  6. hit_rate / reset
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mrr_residency import MrrResidencyManager


def _raises(exc_type, fn):
    """断言 fn() 会抛出指定类型的异常。

    作用：把"期望某操作失败"的测试意图写成简洁的布尔表达式，
    避免每个用例都写一大段 try/except。
    参数：
        exc_type: 期望抛出的异常类型（如 ValueError、KeyError）。
        fn: 零参数可调用对象，通常用 lambda 包装真实的调用。
    返回值：
        bool：若 fn() 确实抛出了 exc_type 异常，返回 True；否则返回 False。
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

    验证重点：capacity_tiles（可容纳的 tile 数）和 refresh_interval_s
    （刷新间隔，秒）都必须是正数；传 0 或负数应抛 ValueError。
    refresh_interval_s=None 表示关闭自动刷新，属于合法输入。
    关键断言：全部依赖 _raises 辅助函数，只要抛错即通过。
    """
    assert _raises(ValueError, lambda: MrrResidencyManager(capacity_tiles=0))
    assert _raises(ValueError, lambda: MrrResidencyManager(capacity_tiles=-1))
    assert _raises(ValueError, lambda: MrrResidencyManager(refresh_interval_s=0.0))
    assert _raises(ValueError, lambda: MrrResidencyManager(refresh_interval_s=-1.0))
    # None disables refresh (valid)
    # None 表示关闭自动刷新，是合法取值，不应报错
    MrrResidencyManager(refresh_interval_s=None)


# ---------------------------------------------------------------------------
# Cold miss / hit
# ---------------------------------------------------------------------------

def test_cold_miss():
    """验证"冷未命中"（cold miss）行为。

    验证重点：一个从未访问过的 tile，第一次查找必然未命中；
    统计中 cold_misses（冷未命中数）与 misses（总未命中数）都应 +1，
    且此时没有发生刷新、没有淘汰其他 tile。
    关键断言：r["hit"] 为 False、r["refresh"] 为 False、evicted_tile_id 为 None。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    r = mgr.lookup_tile("t0", 0.0)
    assert r["hit"] is False
    assert r["refresh"] is False
    assert r["evicted_tile_id"] is None
    assert mgr.stats["cold_misses"] == 1
    assert mgr.stats["misses"] == 1


def test_second_hit():
    """验证同一 tile 第二次查找能命中（hit）。

    验证重点：t0 首次访问后驻留在芯片上，第二次访问（时间 1.0 秒）
    应命中且无需刷新。
    关键断言：r["hit"] 为 True、r["refresh"] 为 False、hits 统计为 1。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    mgr.lookup_tile("t0", 0.0)
    r = mgr.lookup_tile("t0", 1.0)
    assert r["hit"] is True
    assert r["refresh"] is False
    assert mgr.stats["hits"] == 1


# ---------------------------------------------------------------------------
# Refresh expiry
# ---------------------------------------------------------------------------

def test_refresh_expiry_boundary():
    """验证刷新过期（refresh expiry）的临界时间点。

    验证重点：refresh_interval_s=10.0 表示驻留超过 10 秒就需要重新刷新。
    "刚刚低于边界"（9.999 秒）仍是命中；"正好等于边界"（10.0 秒）
    按大于等于规则判定为刷新未命中（refresh miss）。
    关键断言：边界前 hit=True、refresh=False；边界上 hit=False、refresh=True，
    且 refreshes 统计为 1。
    """
    mgr = MrrResidencyManager(capacity_tiles=96, refresh_interval_s=10.0)
    mgr.lookup_tile("t0", 0.0)          # cold miss, last_program_s = 0.0
    mgr.record_program("t0", 0.0)       # program at t=0
    # Just below the boundary → hit
    # 时间还差一点点到 10 秒，未过期，应命中
    r = mgr.lookup_tile("t0", 9.999)
    assert r["hit"] is True
    assert r["refresh"] is False
    # Exactly at the boundary (>=) → refresh miss
    # 刚好达到 10 秒边界（判断条件是 >=），应判定需要刷新
    r = mgr.lookup_tile("t0", 10.0)
    assert r["hit"] is False
    assert r["refresh"] is True
    assert mgr.stats["refreshes"] == 1


def test_refresh_miss_keeps_resident():
    """验证刷新未命中之后，tile 仍保持在芯片上。

    验证重点：刷新未命中只是"需要重新烧录"，并不会把 tile 逐出驻留区。
    因此在未烧录新权重之前再次访问，仍会被标记为需要刷新，
    且 ready_time_s 仍停留在上一次烧录的时间点。
    关键断言：再次查找 refresh 仍为 True，ready_time_s 仍为 0.0。
    """
    mgr = MrrResidencyManager(capacity_tiles=96, refresh_interval_s=10.0)
    mgr.lookup_tile("t0", 0.0)
    mgr.record_program("t0", 0.0)
    mgr.lookup_tile("t0", 10.0)  # refresh miss
    # Still resident; ready_time_s is the stale last_program_s
    # tile 仍驻留；ready_time_s 仍是旧的烧录时间（尚未重新烧录）
    r = mgr.lookup_tile("t0", 10.0)
    assert r["refresh"] is True
    assert r["ready_time_s"] == 0.0


# ---------------------------------------------------------------------------
# record_program
# ---------------------------------------------------------------------------

def test_record_program_updates_last_program():
    """验证 record_program 会更新烧录时间戳。

    验证重点：在 t=5.0 秒重新烧录 t0 后，t=6.0 秒访问应立即命中，
    且 ready_time_s（可用时间）更新为 5.0。
    关键断言：r["hit"] 为 True、r["ready_time_s"] == 5.0。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    mgr.lookup_tile("t0", 0.0)
    mgr.record_program("t0", 5.0)
    r = mgr.lookup_tile("t0", 6.0)
    assert r["hit"] is True
    assert r["ready_time_s"] == 5.0


def test_record_program_requires_resident():
    """验证 record_program 的前置条件校验。

    验证重点：对"从未驻留的 tile"烧录应抛 KeyError（根本不在驻留表里）；
    烧录时间传负数应抛 ValueError。
    关键断言：两个非法调用都通过 _raises 判定抛错。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    assert _raises(KeyError, lambda: mgr.record_program("missing", 1.0))
    mgr.lookup_tile("t0", 0.0)
    assert _raises(ValueError, lambda: mgr.record_program("t0", -1.0))


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction():
    """验证 LRU（最近最少使用）淘汰策略。

    验证重点：驻留区容量为 2 个 tile，先放 t0、t1，再访问 t0 使其成为
    最新使用，此时 t1 变成最久未用；新来 t2 触发冷未命中且容量已满，
    于是被淘汰的应是 t1。
    关键断言：r["evicted_tile_id"] == "t1"，evictions 统计为 1。
    """
    mgr = MrrResidencyManager(capacity_tiles=2)
    mgr.lookup_tile("t0", 0.0)   # last_used_order = 1
    mgr.lookup_tile("t1", 1.0)   # last_used_order = 2
    mgr.lookup_tile("t0", 2.0)   # hit → last_used_order = 3 (t1 is now LRU)
    r = mgr.lookup_tile("t2", 3.0)  # cold miss, capacity full → evict t1
    assert r["hit"] is False
    assert r["evicted_tile_id"] == "t1"
    assert mgr.stats["evictions"] == 1


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------

def test_invalidate():
    """验证显式失效（invalidate）功能。

    验证重点：invalidate("t0") 可主动把一个 tile 从驻留区移除，返回 True；
    对已不存在的 tile 再失效返回 False。invalidate_many 返回成功移除的数量。
    关键断言：失效统计 invalidations 累计为 2。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    mgr.lookup_tile("t0", 0.0)
    mgr.lookup_tile("t1", 0.0)
    assert mgr.invalidate("t0") is True
    assert mgr.invalidate("t0") is False  # already gone
    assert mgr.invalidate_many(["t1", "missing"]) == 1
    assert mgr.stats["invalidations"] == 2


# ---------------------------------------------------------------------------
# hit_rate / reset
# ---------------------------------------------------------------------------

def test_hit_rate():
    """验证命中率（hit_rate）统计。

    验证重点：两次访问中一次未命中、一次命中，命中率应为 0.5。
    关键断言：summary()["hit_rate"] == 0.5。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    mgr.lookup_tile("t0", 0.0)  # miss
    mgr.lookup_tile("t0", 1.0)  # hit
    assert mgr.summary()["hit_rate"] == 0.5


def test_reset_clears_state():
    """验证 reset 会清空全部状态。

    验证重点：访问一个 tile 后调用 reset，驻留表应清空（resident_tiles=0），
    访问统计也应归零（accesses=0）。
    关键断言：summary()["resident_tiles"] == 0 且 stats["accesses"] == 0。
    """
    mgr = MrrResidencyManager(capacity_tiles=96)
    mgr.lookup_tile("t0", 0.0)
    mgr.reset()
    assert mgr.summary()["resident_tiles"] == 0
    assert mgr.stats["accesses"] == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 允许直接 `python test_mrr_residency.py` 运行：不使用 pytest，
    # 而是手写一个极简测试驱动：逐个执行测试函数并统计通过/失败数。
    import traceback

    tests = [
        ("constructor_validation", test_constructor_validation),
        ("cold_miss", test_cold_miss),
        ("second_hit", test_second_hit),
        ("refresh_expiry_boundary", test_refresh_expiry_boundary),
        ("refresh_miss_keeps_resident", test_refresh_miss_keeps_resident),
        ("record_program_updates_last_program", test_record_program_updates_last_program),
        ("record_program_requires_resident", test_record_program_requires_resident),
        ("lru_eviction", test_lru_eviction),
        ("invalidate", test_invalidate),
        ("hit_rate", test_hit_rate),
        ("reset_clears_state", test_reset_clears_state),
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
