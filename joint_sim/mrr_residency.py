"""MRR tile-residency state for Experiment 4.

Tile-SRAM answers whether a weight tile must be fetched from HBM.  This
manager separately answers whether that SRAM-resident tile is already tuned
on an MRR slot.  It is intentionally small and deterministic: the physical
MRR array is represented as a fully-associative set of tile slots with LRU
replacement.

中文阅读提示：该类不做 HBM/SRAM 搬运；它只回答“这个权重是否还留在某个
光子 core 的微环中”，从而决定是否必须重新编程。
"""
# =============================================================================
# 本文件角色一句话：管理"权重是否已调谐在 MRR（微环谐振器）上"。
# 注意与 sram_manager.py 的分工：
#   - sram_manager  回答"这块权重在不在片上 SRAM 里"（数据搬运层）；
#   - 本文件         回答"这块权重是否还在某个光子 core 的微环中、能否
#                    直接复用调谐状态"（器件驻留层）。
# 每个槽位对应一个能放 64x64 权重 tile 的物理光子 core，共 96 个。
# 若权重被 SRAM 踢掉（backing 没了），本文件必须同步失效对应的调谐，
# 否则会出现"光路上以为有、其实没有"的错误。
# =============================================================================

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class MrrSlot:
    # 中文说明：MRR 上一个槽位的状态。
    #   last_used_order 上次被访问的全局序号（LRU 替换用，越旧越先被踢）
    #   last_program_s   上次编程（调谐）完成的时刻（刷新判据用）
    tile_id: str
    last_used_order: int
    last_program_s: float


class MrrResidencyManager:
    # 一个 slot 对应一个可装载 64×64 权重 tile 的物理光子 core。
    """Track programmed MRR tiles and optional periodic refreshes.

    Parameters
    ----------
    capacity_tiles:
        Number of independently retained 64x64 weight tiles.  The default
        physical configuration uses 96 slots (one per photonic core).
    refresh_interval_s:
        If positive, a resident tile whose last program is older than this
        interval is a refresh miss.  ``None`` disables refreshes.
    """
    # 中文说明：MRR 驻留管理器。它的存在意义：光路的调谐（把权重烧进微环）
    # 很慢且耗能，所以权重尽量"留在光路上复用"；本类用一张全相联小表 +
    # LRU 替换来模拟这个行为，并支持可选的周期刷新（防漂移）。
    # 类比：MRR 驻留相当于把常用数据留在"寄存器"里，换走重装代价大。

    def __init__(
        self,
        capacity_tiles: int = 96,
        refresh_interval_s: Optional[float] = None,
    ):
        if capacity_tiles <= 0:
            raise ValueError("capacity_tiles must be positive")
        if refresh_interval_s is not None and refresh_interval_s <= 0:
            raise ValueError("refresh_interval_s must be positive or None")
        self.capacity_tiles = int(capacity_tiles)
        self.refresh_interval_s = refresh_interval_s
        self._resident: Dict[str, MrrSlot] = {}   # tile_id -> 槽位
        self._access_order = 0                     # 全局访问计数器（LRU 用）
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict:
        # 统计：访问/命中/未命中/冷启动未命中/踢出/刷新/失效次数
        return {
            "accesses": 0,
            "hits": 0,
            "misses": 0,
            "cold_misses": 0,
            "evictions": 0,
            "refreshes": 0,
            "invalidations": 0,
        }

    def lookup_tile(self, tile_id: str, now_s: float) -> dict:
        """Return whether ``tile_id`` can reuse its current MRR tuning.

        A miss reserves an MRR slot immediately; callers must subsequently
        call :meth:`record_program` with the scheduled programming end time.
        The scheduler's main Experiment-4 path is serial, so the reservation
        is deterministic and cannot be observed before its program event.
        """
        # 中文说明：查询某 tile 是否还能复用当前调谐。
        #   - 命中（resident 且未过期）：直接复用，返回上次编程完成时刻；
        #   - 刷新缺失（resident 但超过刷新周期）：需要重新编程，标记 refresh；
        #   - 完全未命中（冷启动）：预留一个槽位（容量满则先 LRU 踢一块），
        #     返回后调用方必须安排编程事件，并在结束时调 record_program。
        # 类比：问"这个微环的调谐还能用吗"，能用就省一次昂贵的重新调谐。
        self._access_order += 1
        self.stats["accesses"] += 1
        slot = self._resident.get(tile_id)
        if slot is not None:
            # 判断是否超过刷新周期：距离上次编程 >= 间隔 → 需要刷新
            expired = (
                self.refresh_interval_s is not None
                and now_s - slot.last_program_s >= self.refresh_interval_s
            )
            slot.last_used_order = self._access_order
            if not expired:
                self.stats["hits"] += 1
                return {
                    "hit": True,
                    "refresh": False,
                    "evicted_tile_id": None,
                    "ready_time_s": slot.last_program_s,
                }
            self.stats["misses"] += 1
            self.stats["refreshes"] += 1
            return {
                "hit": False,
                "refresh": True,
                "evicted_tile_id": None,
                "ready_time_s": slot.last_program_s,
            }

        self.stats["misses"] += 1
        self.stats["cold_misses"] += 1
        evicted = None
        # 容量满时按 LRU 踢掉"最久没用"的槽位
        if len(self._resident) >= self.capacity_tiles:
            evicted = min(
                self._resident,
                key=lambda tid: self._resident[tid].last_used_order,
            )
            del self._resident[evicted]
            self.stats["evictions"] += 1

        # 预留槽位（调谐尚未真正完成，完成时刻由 record_program 回填）
        self._resident[tile_id] = MrrSlot(
            tile_id=tile_id,
            last_used_order=self._access_order,
            last_program_s=now_s,
        )
        return {
            "hit": False,
            "refresh": False,
            "evicted_tile_id": evicted,
            "ready_time_s": now_s,
        }

    def record_program(self, tile_id: str, program_end_s: float) -> None:
        """Record the completion time of a program or refresh operation."""
        # 中文说明：调度器把编程/刷新事件排完后，回填其完成时刻。
        # 之后对这块 tile 的查询就以这个时刻为基准判断"能否复用/是否该刷新"。
        if program_end_s < 0:
            raise ValueError("program_end_s must be non-negative")
        slot = self._resident.get(tile_id)
        if slot is None:
            raise KeyError(f"cannot record program for non-resident MRR tile: {tile_id}")
        slot.last_program_s = program_end_s

    def invalidate(self, tile_id: str) -> bool:
        """Invalidate a tuned tile when its backing SRAM copy is evicted."""
        # 中文说明：当该 tile 在 SRAM 里被踢掉（backing 数据没了）时，
        # 必须同时把 MRR 上的调谐作废，否则光路上还"以为"有这份权重。
        if tile_id not in self._resident:
            return False
        del self._resident[tile_id]
        self.stats["invalidations"] += 1
        return True

    def invalidate_many(self, tile_ids: Iterable[str]) -> int:
        """批量失效多个 tile，返回实际失效的数量。"""
        return sum(1 for tile_id in tile_ids if self.invalidate(tile_id))

    def reset(self) -> None:
        """清空状态，准备新一轮仿真。"""
        self._resident.clear()
        self._access_order = 0
        self.stats = self._empty_stats()

    def summary(self) -> dict:
        """返回一行统计摘要（命中率、驻留数、刷新间隔等）。"""
        accesses = self.stats["accesses"]
        return {
            **self.stats,
            "capacity_tiles": self.capacity_tiles,
            "resident_tiles": len(self._resident),
            "hit_rate": self.stats["hits"] / accesses if accesses else 0.0,
            "refresh_interval_s": self.refresh_interval_s,
        }
