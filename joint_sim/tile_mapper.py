"""
64x64 physical weight tile mapper (Experiment 4, Section 7.2).

Supports the legacy unsigned 64x64 mapping and a fixed-area signed row-pair
mapping whose 64 physical output rows represent 32 logical signed outputs.
Splits every KxN Linear weight into physical tiles,
annotates tile access traces with future-use metadata, and validates
that tile counts match trace statistics.

中文阅读提示：这里把“大权重矩阵”翻译成硬件能放下的 64×64 小块；它不安排
执行时间，时间和资源争用由 scheduler.py 负责。
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from schema import TileRecord, TileAccess, CORE_SIZE
except ImportError:
    from .schema import TileRecord, TileAccess, CORE_SIZE


# 支持的映射模式：无符号 64x64、固定面积有符号行配对；
# 存储模式：紧凑(每个逻辑权重存 1 个值)、扩展(正负权重各存 1 份)。
MAPPING_MODES = {"unsigned_64x64", "signed_row_pair_fixed_area"}
STORAGE_MODES = {"compact", "expanded"}


def _validate_modes(mapping_mode: str, storage_mode: str) -> None:
    """校验映射模式与存储模式参数是否合法，不合法直接抛异常。"""
    if mapping_mode not in MAPPING_MODES:
        raise ValueError(
            f"mapping_mode must be one of {sorted(MAPPING_MODES)}, got {mapping_mode!r}"
        )
    if storage_mode not in STORAGE_MODES:
        raise ValueError(
            f"storage_mode must be one of {sorted(STORAGE_MODES)}, got {storage_mode!r}"
        )


def split_weight_to_tiles(
    weight_id: str,
    K: int,
    N: int,
    core_size: int = CORE_SIZE,
    mapping_mode: str = "unsigned_64x64",
    storage_mode: str = "compact",
) -> List[TileRecord]:
    """Split a KxN weight into physical MRR tiles.

    ``unsigned_64x64`` uses 64 logical N outputs per physical core.
    ``signed_row_pair_fixed_area`` uses adjacent positive/negative physical
    rows, so a 64-row core exposes 32 logical N outputs.  Compact storage keeps
    one signed int8 value per logical weight; expanded storage keeps W+ and W-.
    """
    # 中文说明：把一个 KxN 的大权重矩阵拆成若干 64x64 的 tile（小块）。
    # 每个 tile 之后会映射到光子阵列的一个 64x64 核心上。
    #   - 有符号行配对模式下，64 个物理行只能表达 32 个逻辑输出，
    #     所以每个核心的"逻辑容量 N"只有 32（logical_capacity_n）。
    #   - k_tiles / n_tiles 用向上取整除法得到行列方向的 tile 数量；
    #     边缘不够 64 的"尾巴 tile"的有效行列数会更小（is_tail=True）。
    if not weight_id:
        raise ValueError("weight_id must be non-empty")
    if core_size <= 0:
        raise ValueError("core_size must be positive")
    if core_size % 2:
        raise ValueError("core_size must be even for signed row-pair mapping")
    _validate_modes(mapping_mode, storage_mode)
    if K <= 0 or N <= 0:
        return []

    # 每个核心能容纳的逻辑输出列数：无符号=64；有符号行配对=32
    logical_capacity_n = (
        core_size // 2
        if mapping_mode == "signed_row_pair_fixed_area"
        else core_size
    )
    # 行列两个方向各需要多少个 tile（向上取整：除不尽就多一块）
    k_tiles = (K + core_size - 1) // core_size
    n_tiles = (N + logical_capacity_n - 1) // logical_capacity_n
    tiles = []

    for k_idx in range(k_tiles):
        # 本块实际用到的行数：不足 core_size 时取剩余行数
        eff_k = min(core_size, K - k_idx * core_size)
        for n_idx in range(n_tiles):
            eff_n = min(
                logical_capacity_n,
                N - n_idx * logical_capacity_n,
            )
            # tile 命名规则：{weight_id}_k{k_idx}_n{n_idx}，全局唯一
            tile_id = f"{weight_id}_k{k_idx}_n{n_idx}"
            # 有符号行配对：每个权重占 2 个物理行 → 有效 MRR 数翻倍
            sign_factor = 2 if mapping_mode == "signed_row_pair_fixed_area" else 1
            # 扩展存储：正负权重各存一份 → 存储字节数再翻倍
            storage_factor = (
                2
                if mapping_mode == "signed_row_pair_fixed_area"
                and storage_mode == "expanded"
                else 1
            )
            tiles.append(TileRecord(
                tile_id=tile_id,
                weight_id=weight_id,
                k_idx=k_idx,
                n_idx=n_idx,
                effective_K=eff_k,
                effective_N=eff_n,
                valid_mrrs=sign_factor * eff_k * eff_n,
                physical_mrrs=core_size * core_size,
                is_tail=(eff_k < core_size or eff_n < logical_capacity_n),
                mapping_mode=mapping_mode,
                storage_mode=storage_mode,
                storage_bytes=storage_factor * eff_k * eff_n,
                logical_capacity_n=logical_capacity_n,
            ))
    return tiles


class TileMapper:
    # 同时保存静态切分结果与动态访问顺序，供后面的缓存/驻留策略复用。
    """Maps operator trace records to tile-level access sequences.

    Parameters
    ----------
    core_size : int
        Core dimension (default 64).
    """
    # 中文说明：本类负责两件事——
    #   (1) 把所有唯一权重按 (K, N) 切成 tile 并缓存（build_tile_map）；
    #   (2) 把 trace 里对权重的访问"展开"成对每个 tile 的访问序列，并
    #       预计算未来使用信息（build_tile_trace），供 SRAM/MRR 驻留策略使用。
    # 注意：它只做"切块"和"标注"，不安排执行时间（那是 scheduler 的事）。

    def __init__(
        self,
        core_size: int = CORE_SIZE,
        mapping_mode: str = "unsigned_64x64",
        storage_mode: str = "compact",
    ):
        # 内部数据结构：
        #   _tile_map      weight_id -> 该权重切出的所有 tile 记录列表
        #   _tile_accesses 全部 tile 访问的序列（按 trace 顺序）
        #   _tile_index    tile_id -> TileRecord 的快速索引（字典查找用）
        if core_size <= 0:
            raise ValueError("core_size must be positive")
        if core_size % 2:
            raise ValueError("core_size must be even for signed row-pair mapping")
        _validate_modes(mapping_mode, storage_mode)
        self.core_size = core_size
        self.mapping_mode = mapping_mode
        self.storage_mode = storage_mode
        self.logical_capacity_n = (
            core_size // 2
            if mapping_mode == "signed_row_pair_fixed_area"
            else core_size
        )
        self._tile_map: Dict[str, List[TileRecord]] = {}   # weight_id → tiles
        self._tile_accesses: List[TileAccess] = []
        self._tile_index: Dict[str, TileRecord] = {}        # tile_id → TileRecord

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_tile_map(
        self,
        records: List[dict],
    ) -> Dict[str, List[TileRecord]]:
        """Extract unique (weight_id, K, N) from trace records and split into tiles.

        Returns {weight_id: [TileRecord, ...]} and caches internally.
        """
        # 中文说明：从 trace 里收集所有出现过的权重（按 weight_id 去重），
        # 校验每个权重形状一致后，逐个切块，并把结果缓存到 self._tile_map。
        seen: Dict[str, Tuple[int, int]] = {}
        for rec in records:
            wid = rec.get("weight_id")
            K = rec.get("K") or 0
            N = rec.get("N") or 0
            if not wid or K <= 0 or N <= 0:
                continue
            shape = (K, N)
            # 同一个权重在不同算子记录里必须形状一致，否则切块会矛盾
            if wid in seen and seen[wid] != shape:
                raise ValueError(
                    f"Weight {wid!r} has inconsistent shapes: {seen[wid]} and {shape}"
                )
            seen[wid] = shape

        self._tile_map = {}
        self._tile_index = {}
        for wid, (K, N) in seen.items():
            tiles = split_weight_to_tiles(
                wid,
                K,
                N,
                self.core_size,
                mapping_mode=self.mapping_mode,
                storage_mode=self.storage_mode,
            )
            self._tile_map[wid] = tiles
            for t in tiles:
                self._tile_index[t.tile_id] = t

        return self._tile_map

    def build_tile_trace(
        self,
        records: List[dict],
    ) -> List[TileAccess]:
        """Annotate each trace record with tile-level access metadata.

        For each photonic-eligible operator, emits one TileAccess per
        constituent tile. Populates next_use_order and remaining_uses
        by scanning forward through the trace.
        """
        # 中文说明：把"对权重的访问"细化成"对 tile 的访问"。
        # 对每个访问，预计算两个未来信息：
        #   next_use_order = 该 tile 下一次被访问的全局 order（无则 None）；
        #   remaining_uses = 之后还剩多少次访问。
        # 有了"未来信息"，调度器的驻留策略才能决定"该不该把这块权重留下"。
        # Rebuild so reusing a mapper for a different trace cannot retain a
        # stale weight map from the previous run.
        self.build_tile_map(records)

        # Group records by weight_id to compute future-use metadata
        wid_orders: Dict[str, List[int]] = defaultdict(list)
        opid_by_order: Dict[int, Tuple[str, str, int]] = {}
        wid_by_order: Dict[int, str] = {}
        seen_orders = set()

        for rec in records:
            wid = rec.get("weight_id")
            order = rec.get("order", 0)
            if not wid or wid not in self._tile_map:
                continue
            if order in seen_orders:
                raise ValueError(f"Duplicate operator order in tile trace: {order}")
            seen_orders.add(order)
            wid_orders[wid].append(order)
            opid_by_order[order] = (
                rec.get("op_id", "unknown"),
                rec.get("phase", "unknown"),
                rec.get("rollout_step", 0) or 0,
            )
            wid_by_order[order] = wid

        # Build next_use_order map for each (weight_id, order)
        sorted_orders: Dict[str, List[int]] = {}
        for wid, orders in wid_orders.items():
            sorted_orders[wid] = sorted(orders)

        # 对每个 (权重, order)：在排序后的访问序列里，取紧邻的下一个
        # order 作为 next_use，用"剩余长度"作为 remaining_uses
        next_use: Dict[Tuple[str, int], Optional[int]] = {}
        remaining: Dict[Tuple[str, int], int] = {}
        for wid, orders in sorted_orders.items():
            for i, order in enumerate(orders):
                nxt = orders[i + 1] if i + 1 < len(orders) else None
                next_use[(wid, order)] = nxt
                remaining[(wid, order)] = len(orders) - i - 1

        # Emit tile accesses
        self._tile_accesses = []
        for rec in records:
            wid = rec.get("weight_id")
            order = rec.get("order", 0)
            if not wid or wid not in self._tile_map:
                continue

            tiles = self._tile_map[wid]
            op_id, phase, step = opid_by_order.get(order, ("unknown", "unknown", 0))
            nu = next_use.get((wid, order))
            rem = remaining.get((wid, order), 0)

            # 一次对权重的访问会展开成对该权重所有 tile 的访问
            for t in tiles:
                self._tile_accesses.append(TileAccess(
                    tile_id=t.tile_id,
                    op_id=op_id,
                    phase=phase,
                    rollout_step=step,
                    order=order,
                    next_use_order=nu,
                    remaining_uses=rem,
                ))

        return self._tile_accesses

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def tile_map(self) -> Dict[str, List[TileRecord]]:
        """返回权重切块结果：{weight_id: [TileRecord, ...]}。"""
        return self._tile_map

    @property
    def tile_accesses(self) -> List[TileAccess]:
        """返回 tile 级访问序列（build_tile_trace 的产物）。"""
        return self._tile_accesses

    def get_tile(self, tile_id: str) -> Optional[TileRecord]:
        """按 tile_id 查一个 TileRecord，查不到返回 None。"""
        return self._tile_index.get(tile_id)

    def tile_count(self) -> int:
        """全部 tile 的总数。"""
        return len(self._tile_index)

    def total_valid_mrrs(self) -> int:
        """所有 tile 实际点亮（参与计算）的 MRR 微环总数。"""
        return sum(t.valid_mrrs for t in self._tile_index.values())

    def total_physical_mrrs(self) -> int:
        """所有 tile 物理上占用的 MRR 微环总数（含空闲）。"""
        return sum(t.physical_mrrs for t in self._tile_index.values())

    def stats(self) -> dict:
        """Return tile statistics matching the document's Table 2.2."""
        if not self._tile_accesses:
            return {"error": "build_tile_trace() must be called first"}

        # 按阶段（encode/context/dynamics/decode）统计去重后的 tile 与算子
        by_phase: Dict[str, set] = defaultdict(set)
        by_phase_ops: Dict[str, set] = defaultdict(set)
        for ta in self._tile_accesses:
            by_phase[ta.phase].add(ta.tile_id)
            by_phase_ops[ta.phase].add(ta.op_id)

        # Unique weight count per phase
        wid_phase: Dict[str, set] = defaultdict(set)
        for ta in self._tile_accesses:
            t = self._tile_index.get(ta.tile_id)
            if t:
                wid_phase[ta.phase].add(t.weight_id)

        return {
            "total_unique_weights": len(self._tile_map),
            "total_unique_tiles": self.tile_count(),
            "total_valid_mrrs": self.total_valid_mrrs(),
            "total_physical_mrrs": self.total_physical_mrrs(),
            "mapping_mode": self.mapping_mode,
            "storage_mode": self.storage_mode,
            "logical_capacity_n": self.logical_capacity_n,
            "total_storage_bytes": sum(
                t.storage_bytes for t in self._tile_index.values()
            ),
            "by_phase": {
                phase: {
                    "linear_calls": len(by_phase_ops.get(phase, set())),
                    "unique_weights": len(wid_phase.get(phase, set())),
                    "unique_tiles": len(by_phase.get(phase, set())),
                }
                for phase in ["encode", "context", "dynamics", "decode"]
            },
        }

    def dump(self, filepath: str):
        """Save tile map and accesses to JSONL for audit."""
        # 第一行是 tile_manifest（切块元信息），之后每行一条 tile 访问，
        # 供审阅/复现时核对切块结果与访问序列。
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "tile_manifest",
                "core_size": self.core_size,
                "mapping_mode": self.mapping_mode,
                "storage_mode": self.storage_mode,
                "logical_capacity_n": self.logical_capacity_n,
                "total_weights": len(self._tile_map),
                "total_tiles": self.tile_count(),
                "total_storage_bytes": sum(
                    t.storage_bytes for t in self._tile_index.values()
                ),
            }) + "\n")
            for ta in self._tile_accesses:
                t = self._tile_index.get(ta.tile_id)
                f.write(json.dumps({
                    "tile_id": ta.tile_id,
                    "weight_id": t.weight_id if t else None,
                    "k_idx": t.k_idx if t else None,
                    "n_idx": t.n_idx if t else None,
                    "valid_mrrs": t.valid_mrrs if t else None,
                    "is_tail": t.is_tail if t else None,
                    "op_id": ta.op_id,
                    "phase": ta.phase,
                    "rollout_step": ta.rollout_step,
                    "order": ta.order,
                    "remaining_uses": ta.remaining_uses,
                }, default=str) + "\n")
