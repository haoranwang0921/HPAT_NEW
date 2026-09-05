"""
SRAM tile residency manager for Tile-SRAM architecture (Experiment 4).

Models 12×8MB SRAM (96MB total) as a managed cache between HBM and MRR
programming. Tiles are loaded from HBM on first access and may be evicted
when capacity is exceeded. Pluggable policies determine eviction order.

Models the Tile-SRAM layer between HBM and MRR programming.  MRR capacity
(96 slots) is too small for meaningful residency; SRAM (96 MB, ~24K tiles) is
the right granularity for cache-line replacement decisions.

Policies:
  LRU           Least-recently-used (online baseline)
  PhaseAware    Evict finished-phase tiles first, prefer Dynamics during rollout
  RolloutAware  Score-based: retain tiles with high remaining_uses / distance
  BeladyOptimal Evict tile with furthest next use (offline optimal lower bound)

中文阅读提示：SRAM 命中只表示权重已在片上；它不等于 MRR 已完成调谐。
MRR 这一层的状态由 mrr_residency.py 单独管理。
"""
# =============================================================================
# 本文件角色一句话：管理"HBM <-> MRR"之间的片上 SRAM 权重缓存。
# 类比：SRAM 相当于 CPU 的高速缓存（L2），MRR 驻留相当于把常用数据留在
# 寄存器里。本文件只关心"这块权重要不要从 HBM 搬上来、满了踢哪块走"，
# 不关心 MRR 是否已调谐（那是 mrr_residency.py 的事）。
#
# 容量数字：默认 12 x 8 MB = 96 MB SRAM，一个 64x64 tile 占 4 KB，
# 所以约能容纳 96*1024*1024/4096 = 24576 个 tile。
# 换页（eviction）策略可插拔：LRU / PhaseAware / RolloutAware / BeladyOptimal。
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import heapq


TILE_BYTES = 4096  # 64×64 × 8-bit
# 一个 tile 占的字节数：64x64 个 8 位权重 = 4096 字节 = 4 KB。


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class SramPolicy(ABC):
    """Abstract eviction policy for SRAM tile cache."""
    # 中文说明：所有换页（eviction）策略的抽象基类。子类只需回答一个问题：
    # "缓存满了，该踢哪块 tile 走？" 不同策略给出不同答案：
    #   LRU           踢最久没用的（在线基线，不看未来）
    #   PhaseAware    先踢已结束阶段的，rollout 阶段优先保 dynamics 的
    #   RolloutAware  利用未来使用次数/距离打分，分低的先踢
    #   BeladyOptimal 踢"下一次使用最晚"的（离线最优下界）

    @abstractmethod
    def name(self) -> str:
        # 策略名（字符串），用于日志与报表
        ...

    @abstractmethod
    def select_victim(
        self,
        resident: Dict[str, dict],
        capacity_tiles: int,
        current_time_s: float,
        future_info: Optional[Dict[str, dict]] = None,
    ) -> Optional[str]:
        """Choose one tile_id to evict, or None if eviction impossible."""
        # resident：当前 SRAM 里驻留的 tile_id -> 槽位信息 dict
        # future_info：未来使用信息（tile_id -> 元数据），供先知型策略使用
        # 返回要踢掉的 tile_id；没有可踢的返回 None
        ...

    def on_phase_transition(self, new_phase: str):
        """Optional hook called when the trace phase changes."""
        # 阶段切换时的回调（比如 encode -> context），策略可借此重置内部状态
        pass

    def on_access(self, tile_id: str, future_info: Optional[dict]):
        """Optional hook after a tile's future-use metadata advances."""
        # 每次访问 tile 后的回调，用于更新 LRU 堆等状态
        pass

    def reset(self):
        """Optional lifecycle hook for stateful policies."""
        # 新一轮仿真开始时清空内部状态
        pass


class LruSramPolicy(SramPolicy):
    """Evict the least-recently-used tile."""
    # 中文说明：最经典的"最近最少使用"策略——踢掉最久没被访问的 tile。
    # 用最小堆实现：堆顶永远是"访问顺序最小"（=最久没被用）的 tile。
    # 惰性失效：若堆顶的 tile 已不在缓存里或版本过期，就把它弹出再找下一个，
    # 避免每次访问都扫描全缓存。

    def __init__(self):
        self._heap = []                    # 堆元素：(最后访问order, 版本号, tile_id)
        self._versions: Dict[str, int] = {}  # tile_id -> 当前版本号

    def name(self) -> str:
        return "lru"

    def reset(self):
        self._heap = []
        self._versions = {}

    def on_access(self, tile_id: str, future_info: Optional[dict]):
        # 每次访问把该 tile 的最新"最后访问顺序"推进堆里（旧条目留作失效标记）
        version = self._versions.get(tile_id, 0) + 1
        self._versions[tile_id] = version
        order = (future_info or {}).get("last_order", 0)
        heapq.heappush(self._heap, (order, version, tile_id))

    def select_victim(self, resident, capacity_tiles, current_time_s, future_info=None):
        if not resident:
            return None
        # 跳过过期条目（已被访问更新过版本，或已被踢走）
        while self._heap:
            _order, version, tile_id = self._heap[0]
            if tile_id in resident and self._versions.get(tile_id) == version:
                return tile_id
            heapq.heappop(self._heap)
        # 防御性重建：堆里没有有效条目时，把当前驻留的 tile 全部重新入堆再选
        for tile_id in resident:
            self.on_access(tile_id, (future_info or {}).get(tile_id))
        return self.select_victim(resident, capacity_tiles, current_time_s, future_info)


class PhaseAwareSramPolicy(SramPolicy):
    """Phase-aware eviction with priority for Dynamics tiles.

    Rules:
      1. Finished-phase tiles (encode, context after their phase ends) evicted first.
      2. Within active phase: LRU.
      3. Dynamics tiles given highest retention during dynamics phase.
    """
    # 中文说明：利用"阶段"信息做替换。模型推理分 encode(编码)/context(上下文)/
    # dynamics(动态演化)/decode(解码) 四个阶段，每个阶段只用一部分权重。
    # 规则：先踢"已经结束的阶段"的 tile；当前阶段内按 LRU 踢；
    # dynamics 阶段最需要反复复用权重，给最高保留优先级。
    # 排序键：(phase_prio, 最后访问order, 版本号, tile_id)，堆顶即要踢的。

    _FINISHED_ORDER = {"encode": 0, "context": 1, "decode": 2, "dynamics": 99}

    def __init__(self):
        self._current_phase = ""
        self._heap = []
        self._versions: Dict[str, int] = {}

    def name(self) -> str:
        return "phase_aware"

    def on_phase_transition(self, new_phase: str):
        # 阶段切换：清空堆（各 tile 的优先级随之重算）
        self._current_phase = new_phase
        self._heap = []

    def reset(self):
        self._current_phase = ""
        self._heap = []
        self._versions = {}

    def on_access(self, tile_id: str, future_info: Optional[dict]):
        info = future_info or {}
        tile_phase = info.get("phase", "")
        # 已结束阶段的 tile 用 _FINISHED_ORDER 里的低优先级（先踢）；
        # 当前阶段内的 tile 优先级为 99（尽量保留）
        if tile_phase != self._current_phase and tile_phase in self._FINISHED_ORDER:
            phase_prio = self._FINISHED_ORDER.get(tile_phase, 50)
        else:
            phase_prio = 99
        version = self._versions.get(tile_id, 0) + 1
        self._versions[tile_id] = version
        heapq.heappush(
            self._heap,
            (phase_prio, info.get("last_order", 0), version, tile_id),
        )

    def select_victim(self, resident, capacity_tiles, current_time_s, future_info=None):
        if not resident:
            return None

        # 惰性失效：跳过版本过期的堆条目
        while self._heap:
            _priority, _order, version, tile_id = self._heap[0]
            if tile_id in resident and self._versions.get(tile_id) == version:
                return tile_id
            heapq.heappop(self._heap)
        # 防御性重建：重新为当前驻留的 tile 建立堆
        for tile_id in resident:
            self.on_access(tile_id, (future_info or {}).get(tile_id))
        victim = self.select_victim(
            resident, capacity_tiles, current_time_s, future_info
        )
        if victim is not None:
            return victim

        # 兜底：堆完全不可用时，直接在 resident 里按排序键取最小值
        def _key(tid: str) -> tuple:
            info = resident[tid]
            tile_phase = info.get("phase", "")
            if tile_phase != self._current_phase and tile_phase in self._FINISHED_ORDER:
                phase_prio = self._FINISHED_ORDER.get(tile_phase, 50)
            else:

            # Finished phases → low priority (evict first = low sort key)
                phase_prio = 99  # active phase — keep
            # Secondary sort: older = evict first


            age = info.get("last_used_s", 0.0)
            return (phase_prio, age)

        return min(resident, key=_key)


class RolloutAwareSramPolicy(SramPolicy):
    """Score-based eviction using future-use metadata.

    Score_i = remaining_uses * T_dma / (next_use_distance + 1)
              - drift_risk * eta

    High score → keep resident. Evict lowest score.

    Requires register_future_info() before simulation.
    """
    # 中文说明：利用"预知的未来"给每个 tile 打分，分高的留下、分低的踢走。
    # 打分直觉（得分越高越值得留）：
    #   - 未来还要用很多次（remaining_uses 大）→ 值得留；
    #   - 很快就要再用（next_use_distance 小，即下一次用得越近）→ 值得留；
    #   - 漂移风险高（drift_risk 大，光器件老化/温漂）→ 扣分；
    #   - 已经结束阶段的 tile → 大幅扣分（phase_penalty=1000），优先踢。
    # 公式：score = remaining_uses / distance - eta*drift - phase_penalty
    # 需要先通过 register_future_info() 把未来信息喂进来才能打分。

    def __init__(self, eta: float = 2.0):
        self.eta = eta              # 漂移风险的惩罚权重
        self._current_phase = ""
        self._heap = []
        self._versions: Dict[str, int] = {}

    def name(self) -> str:
        return "rollout_aware"

    def on_phase_transition(self, new_phase: str):
        self._current_phase = new_phase
        # Phase penalties affect every resident entry. Rebuild lazily at the
        # next eviction rather than rescoring the full cache at every access.
        self._heap = []

    def reset(self):
        self._current_phase = ""
        self._heap = []
        self._versions = {}

    def _score_from_future(self, future_info: Optional[dict]) -> float:
        # 根据未来信息算一个 tile 的得分（越小越优先被踢）
        fi = future_info or {}
        remaining = fi.get("remaining_uses", 0)
        next_use = fi.get("next_use_order")
        if next_use is None:
            return float("-inf")    # 以后再也不用了 → 最理想的受害者
        order = fi.get("last_order", 0)
        distance = max(next_use - order, 1)   # 距下次使用的"步数"（至少为1）
        phase_penalty = 0.0
        tile_phase = fi.get("phase", "")
        # 已结束阶段的 tile 扣 1000 分，保证它们最先被淘汰
        if tile_phase != self._current_phase and tile_phase in ("encode", "context", "decode"):
            phase_penalty = 1000.0
        return remaining / distance - self.eta * fi.get("drift_risk", 0.0) - phase_penalty

    def on_access(self, tile_id: str, future_info: Optional[dict]):
        # 每次访问把最新得分压入堆（旧条目作为失效标记留在堆里）
        version = self._versions.get(tile_id, 0) + 1
        self._versions[tile_id] = version
        heapq.heappush(
            self._heap,
            (self._score_from_future(future_info), version, tile_id),
        )

    def select_victim(self, resident, capacity_tiles, current_time_s, future_info=None):
        if not resident:
            return None

        # Lazy invalidation makes the policy O(log capacity) per access rather
        # than O(capacity) per eviction on long rollout traces.
        while self._heap:
            _score, version, tile_id = self._heap[0]
            if tile_id in resident and self._versions.get(tile_id) == version:
                return tile_id
            heapq.heappop(self._heap)

        # 防御性重建：堆为空时把当前驻留 tile 全部重新入堆再选
        for tile_id in resident:
            self.on_access(tile_id, (future_info or {}).get(tile_id))
        victim = self.select_victim(
            resident, capacity_tiles, current_time_s, future_info
        )
        if victim is not None:
            return victim

        # 兜底：直接在 resident 里按得分取最小值（得分最低的先踢）
        def _score(tid: str) -> float:
            info = resident[tid]
            fi = (future_info or {}).get(tid, {})

            remaining = fi.get("remaining_uses", 0)
            next_use = fi.get("next_use_order")
            if next_use is None:
                return float("-inf")  # never used again → ideal victim

            order = fi.get("last_order", 0)
            distance = max(next_use - order, 1)
            drift = info.get("drift_risk", 0.0)

            # Finished-phase tiles: heavily penalize
            tile_phase = info.get("phase", "")
            phase_penalty = 0.0
            if tile_phase != self._current_phase and tile_phase in ("encode", "context", "decode"):
                phase_penalty = 1000.0

            return remaining / distance - self.eta * drift - phase_penalty

        return min(resident, key=_score)


class BeladyOptimalSramPolicy(SramPolicy):
    """Belady's OPT: evict tile with furthest next use.

    Requires complete future knowledge. Theoretical lower bound on misses.
    """
    # 中文说明：Belady 最优算法（离线）——踢"下一次使用得最晚"的 tile。
    # 它是缓存替换的理论最优解（缺失次数下界），现实中无法实现（需要完整
    # 预知未来），这里用于给其他在线策略（LRU 等）提供一个"性能上界"参照。
    # 注意实现细节：堆是最小堆，而我们要取"next_use 最大"的，所以
    # 键取负值（-next_use）入堆，这样堆顶恰好是"下一次使用最晚"的。

    def __init__(self):
        self._heap = []
        self._versions: Dict[str, int] = {}

    @staticmethod
    def _key(future_info: Optional[dict]) -> float:
        next_use = (future_info or {}).get("next_use_order")
        # heapq is a min-heap; never-used-again tiles must be selected first.
        return float("-inf") if next_use is None else -float(next_use)

    def on_access(self, tile_id: str, future_info: Optional[dict]):
        version = self._versions.get(tile_id, 0) + 1
        self._versions[tile_id] = version
        heapq.heappush(self._heap, (self._key(future_info), version, tile_id))

    def reset(self):
        self._heap = []
        self._versions = {}

    def name(self) -> str:
        return "belady_optimal"

    def select_victim(self, resident, capacity_tiles, current_time_s, future_info=None):
        if not resident:
            return None

        # Entries become stale when a resident tile is accessed again or when
        # a phase-release path removes it. Lazy invalidation avoids scanning
        # every cached tile for each miss on a long rollout trace.
        while self._heap:
            _key, version, tile_id = self._heap[0]
            if tile_id in resident and self._versions.get(tile_id) == version:
                return tile_id
            heapq.heappop(self._heap)

        # Defensive rebuild for managers restored without prior callbacks.
        for tile_id in resident:
            self.on_access(tile_id, (future_info or {}).get(tile_id))
        victim = self.select_victim(
            resident, capacity_tiles, current_time_s, future_info
        )
        if victim is not None:
            return victim

        # 兜底：直接在 resident 里取"下次使用最晚"（即 max）的那个
        def _next_use(tid: str) -> float:
            fi = (future_info or {}).get(tid, {})
            nu = fi.get("next_use_order")
            if nu is None:
                return float("inf")  # never used again → ideal victim
            return nu

        return max(resident, key=_next_use)


# ---------------------------------------------------------------------------
# SRAM Manager
# ---------------------------------------------------------------------------

@dataclass
class SramSlot:
    """One tile slot in SRAM."""
    # 中文说明：SRAM 里一个 tile 槽位的运行时状态。
    #   phase        该 tile 属于哪个阶段（encode/context/dynamics/decode）
    #   last_used_s  上次被访问的时刻（LRU 用）
    #   ready_time_s 数据真正就绪（DMA 搬完）的时刻
    #   drift_risk   累积漂移风险（器件老化/温漂，0=全新）
    tile_id: str
    phase: str = ""
    last_used_s: float = 0.0
    ready_time_s: float = 0.0
    drift_risk: float = 0.0


class SramManager:
    # 这是 12×8 MB 片上权重缓存的抽象；slot 以 4 KB 的 64×64 权重块为单位。
    """Manages tile residency in the Tile-SRAM cache.

    Parameters
    ----------
    total_bytes : float
        Total SRAM capacity in bytes (e.g. 96 * 1024 * 1024 for 96 MB).
    policy : SramPolicy
        Eviction policy.
    hbm_bandwidth_bytes_per_s : float
        HBM read bandwidth for DMA latency calculation.
    dma_fixed_latency_s : float
        Fixed per-transfer DMA overhead.
    tile_bytes : int
        Bytes per tile (default 4096 = 64×64 × 8-bit).
    """
    # 中文说明：SRAM 缓存管理器。它回答"某 tile 在不在片上；不在的话搬一次
    # 要多久、要不要先踢一块走"。lookup_tile 未命中时只"预留位置"并返回
    # DMA 时延，真正的搬运由调用方（scheduler）安排后再 mark_tile_ready 回填。

    def __init__(
        self,
        total_bytes: float = 96 * 1024 * 1024,
        policy: SramPolicy = None,
        hbm_bandwidth_bytes_per_s: float = 512e9,
        dma_fixed_latency_s: float = 100e-9,
        tile_bytes: int = TILE_BYTES,
    ):
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if tile_bytes <= 0:
            raise ValueError("tile_bytes must be positive")
        if total_bytes < tile_bytes:
            raise ValueError("total_bytes must hold at least one complete tile")
        if hbm_bandwidth_bytes_per_s <= 0:
            raise ValueError("hbm_bandwidth_bytes_per_s must be positive")
        if dma_fixed_latency_s < 0:
            raise ValueError("dma_fixed_latency_s must be non-negative")
        self.total_bytes = total_bytes
        self.policy = policy or LruSramPolicy()
        self.hbm_bandwidth = hbm_bandwidth_bytes_per_s
        self.dma_fixed_latency = dma_fixed_latency_s
        self.tile_bytes = tile_bytes

        # 容量换算：总字节 / 每 tile 字节 = 最多放多少块（向下取整）
        self.capacity_tiles = int(total_bytes // tile_bytes)
        self._resident: Dict[str, SramSlot] = {}
        # Keep the complete per-tile access queue. A single metadata entry
        # would let the last occurrence overwrite all earlier next-use data.
        # 未来信息分开存放：_future_info 是"当前这条访问"的元数据；
        # _future_sequences 保留每个 tile 的完整访问队列（逐条消费），
        # 避免"最后一次访问"覆盖掉前面所有访问的未来信息。
        self._future_info: Dict[str, dict] = {}
        self._future_sequences: Dict[str, List[dict]] = {}
        self._future_cursor: Dict[str, int] = {}
        self.current_time_s: float = 0.0
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict:
        # 统计字段：命中/未命中/踢出次数、累计 DMA 时延与字节、冷启动未命中数
        return {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "dma_latency_s": 0.0,
            "dma_bytes": 0,
            "cold_misses": 0,
        }

    # ------------------------------------------------------------------
    # Future-info registration
    # ------------------------------------------------------------------

    def register_future_info(self, tile_accesses):
        """Register per-tile future-use metadata for oracle policies."""
        # 把 tile 访问序列注册进来，供先知型策略（rollout_aware/belady）使用。
        # 每个 tile 的访问按 order 排序后存入 _future_sequences，
        # 并把第一条访问的元数据设为当前指针所指的信息。
        self.policy.reset()
        self._future_info = {}
        self._future_sequences = {}
        self._future_cursor = {}
        for ta in tile_accesses:
            self._future_sequences.setdefault(ta.tile_id, []).append({
                "next_use_order": ta.next_use_order,
                "remaining_uses": ta.remaining_uses,
                "phase": ta.phase,
                "rollout_step": ta.rollout_step,
                "last_order": ta.order,
            })
        for tile_id, sequence in self._future_sequences.items():
            sequence.sort(key=lambda info: info.get("last_order", 0))
            self._future_cursor[tile_id] = 0
            self._future_info[tile_id] = dict(sequence[0])

    def _advance_future_info(self, tile_id: str):
        """Consume one access and expose its next-use metadata."""
        # 每访问一次就把该 tile 的未来信息指针往后挪一位，
        # 让"下次该用什么未来信息"始终与当前访问一一对应。
        sequence = self._future_sequences.get(tile_id)
        if not sequence:
            return
        cursor = self._future_cursor.get(tile_id, 0)
        if cursor < len(sequence):
            self._future_info[tile_id] = dict(sequence[cursor])
            self._future_cursor[tile_id] = cursor + 1

    def _notify_access(self, tile_id: str) -> None:
        """Update a stateful policy after the resident slot is current."""
        # 把"本次访问"通知给策略（更新 LRU 堆/打分），让替换决策跟上节奏
        slot = self._resident.get(tile_id)
        if slot is None:
            return
        info = {
            **self._future_info.get(tile_id, {}),
            "phase": slot.phase,
            "drift_risk": slot.drift_risk,
        }
        self.policy.on_access(tile_id, info)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _compute_dma_latency(self) -> float:
        """DMA latency for one tile from HBM to SRAM."""
        # 搬运时延 = 传输时间（字节数/带宽）+ 固定开销
        return self.tile_bytes / max(self.hbm_bandwidth, 1.0) + self.dma_fixed_latency

    @property
    def occupied(self) -> int:
        """当前 SRAM 里已驻留的 tile 数。"""
        return len(self._resident)

    @property
    def hit_rate(self) -> float:
        """命中率 = 命中次数 / (命中+未命中)。"""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0

    def resident_tile_ids(self) -> Set[str]:
        """返回当前驻留的全部 tile_id。"""
        return set(self._resident.keys())

    def contains(self, tile_id: str) -> bool:
        """判断某 tile 是否已在 SRAM 中。"""
        return tile_id in self._resident

    def lookup_tile(
        self,
        tile_id: str,
        current_time_s: float,
        metadata: Optional[dict] = None,
    ) -> dict:
        """查询 tile 是否已在 SRAM；未命中时预留位置，调用方随后安排 HBM DMA。

        Returns
        -------
        dict with:
            hit : bool
            dma_latency_s : float (0 on hit)
            evicted_tile_ids : list of str
        """
        # 中文说明：SRAM 缓存查询的核心方法。调用流程分两支：
        #   命中：tile 已在片上，更新访问时间/漂移风险，返回 ready_time_s；
        #   未命中：累计缺失统计，若已满则先按策略踢走若干 tile（记录到
        #           evicted_tile_ids，调用方还要同步失效 MRR 上对应调谐），
        #           然后"预留"槽位、返回 DMA 时延——真正的搬运事件由
        #           调度器随后发出，搬完再调 mark_tile_ready 回填就绪时间。
        self.current_time_s = current_time_s

        # Hit
        if tile_id in self._resident:
            slot = self._resident[tile_id]
            slot.last_used_s = current_time_s
            if metadata:
                slot.drift_risk = metadata.get("drift_risk", slot.drift_risk)
            self.stats["hits"] += 1
            self._advance_future_info(tile_id)
            self._notify_access(tile_id)
            return {
                "hit": True,
                "dma_latency_s": 0.0,
                "evicted_tile_ids": [],
                "ready_time_s": slot.ready_time_s,
            }

        # Miss — may need eviction
        self.stats["misses"] += 1
        # 冷启动未命中：这块 tile 从没出现在未来信息里（首访）
        if tile_id not in self._future_info:
            self.stats["cold_misses"] += 1

        evicted = []
        # 若已满，循环踢人直到腾出至少一个槽位（每次踢一块）
        while self.occupied >= self.capacity_tiles:
            victim = self.policy.select_victim(
                self._resident, self.capacity_tiles, current_time_s,
                self._future_info,
            )
            if victim is None or victim not in self._resident:
                raise RuntimeError(
                    f"SRAM policy {self.policy.name()!r} failed to select a valid victim"
                )
            del self._resident[victim]
            evicted.append(victim)
            self.stats["evictions"] += 1

        # Load tile into SRAM
        dma_lat = self._compute_dma_latency()
        phase = ""
        if metadata:
            phase = metadata.get("phase", "")
        elif tile_id in self._future_info:
            phase = self._future_info[tile_id].get("phase", "")

        # 先以当前时刻建槽位，DMA 完成时间由调度器事后回填
        self._resident[tile_id] = SramSlot(
            tile_id=tile_id,
            phase=phase,
            last_used_s=current_time_s + dma_lat,
            ready_time_s=current_time_s,
            drift_risk=metadata.get("drift_risk", 0.0) if metadata else 0.0,
        )
        self.stats["dma_latency_s"] += dma_lat
        self.stats["dma_bytes"] += self.tile_bytes
        # The miss is also the current access. Keep the cursor after this
        # access so the next eviction sees the correct future distance.
        self._advance_future_info(tile_id)
        self._notify_access(tile_id)

        return {
            "hit": False,
            "dma_latency_s": dma_lat,
            "evicted_tile_ids": evicted,
            "ready_time_s": current_time_s,
        }

    def mark_tile_ready(self, tile_id: str, ready_time_s: float) -> None:
        """Record the real completion time of the scheduled HBM-to-SRAM DMA."""
        # 中文说明：调度器把 HBM->SRAM 的 DMA 事件真正排进时间线并计算好
        # 完成时刻后，调用本方法回填"数据就绪时间"。这是 lookup_tile 与
        # 事件调度之间唯一的握手点。
        if ready_time_s < 0:
            raise ValueError("ready_time_s must be non-negative")
        slot = self._resident.get(tile_id)
        if slot is None:
            raise KeyError(f"cannot mark non-resident SRAM tile ready: {tile_id}")
        slot.ready_time_s = ready_time_s
        slot.last_used_s = max(slot.last_used_s, ready_time_s)

    def notify_phase(self, phase: str):
        """Notify policy of a phase transition."""
        # 阶段切换时通知策略（如 PhaseAware 会清堆重算优先级）
        self.policy.on_phase_transition(phase)

    def evict_phase_tiles(self, phase: str) -> List[str]:
        """Evict all tiles belonging to a given phase."""
        # 阶段结束后批量踢掉该阶段的所有 tile（encode/context 阶段数据一次性
        # 用完即弃，避免占着 SRAM 直到仿真结束）。返回被踢的 tile_id 列表，
        # 调用方需同步失效 MRR 上对应的调谐状态。
        evicted = []
        for tid in list(self._resident.keys()):
            if self._resident[tid].phase == phase:
                del self._resident[tid]
                evicted.append(tid)
        return evicted

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def sync_time(self, absolute_time_s: float):
        """把当前仿真时钟同步到 absolute_time_s（秒）。"""
        self.current_time_s = absolute_time_s

    def reset(self):
        """清空全部状态，准备新一轮仿真。"""
        self._resident.clear()
        self.policy.reset()
        self._future_info.clear()
        self._future_sequences.clear()
        self._future_cursor.clear()
        self.current_time_s = 0.0
        self.stats = self._empty_stats()

    def summary(self) -> dict:
        """返回一行统计摘要（命中率、容量、占用、策略名等）。"""
        return {
            **self.stats,
            "hit_rate": self.hit_rate,
            "capacity_tiles": self.capacity_tiles,
            "capacity_bytes": self.total_bytes,
            "occupied": self.occupied,
            "policy": self.policy.name(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# 策略注册表：按名字找到对应的策略类，供 make_sram_manager 使用
POLICY_REGISTRY = {
    "lru": LruSramPolicy,
    "phase_aware": PhaseAwareSramPolicy,
    "rollout_aware": RolloutAwareSramPolicy,
    "belady_optimal": BeladyOptimalSramPolicy,
}


def make_sram_manager(
    total_mb: float = 96.0,
    policy: str = "lru",
    **kwargs,
) -> SramManager:
    """Create an SramManager with a given capacity and policy.

    Parameters
    ----------
    total_mb : float
        SRAM capacity in megabytes.
    policy : str
        One of: lru, phase_aware, rollout_aware, belady_optimal.
    """
    # 工厂函数：total_mb 换成字节、按名字选策略类，一步创建 SramManager。
    # kwargs 会透传给策略构造函数（如 rollout_aware 的 eta 参数）。
    policy_cls = POLICY_REGISTRY.get(policy, LruSramPolicy)
    return SramManager(
        total_bytes=total_mb * 1024 * 1024,
        policy=policy_cls(**kwargs),
    )
