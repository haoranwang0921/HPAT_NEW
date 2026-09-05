# ============================================================================
# 模块说明：tile_mapper（权重分块映射器）的测试
#   光子 MRR 阵列的物理尺寸有限，一个大的权重矩阵无法整体塞进芯片，
#   必须切成若干小"tile"（瓦片/分块），逐块映射到光子阵列上计算。
#   TileMapper 负责：把权重切成 tile、给每个 tile 生成稳定 id、
#   标注 tile 访问轨迹（含"未来使用"元信息，供缓存策略使用）、
#   统计 tile 数量等工作。
#   本测试验证（对应论文第 4 实验 / 7.2 节 1-3 项）：
#     1. 对齐尺寸与边界尺寸（tail，余数）的切分是否正确
#     2. tile_id 生成的稳定性
#     3. tile 轨迹的"未来使用"标注
#     4. tile 统计与预期工作集大小一致
# ============================================================================
"""
Tests for tile_mapper.py (Experiment 4, Section 7.2 items 1-3).

Verifies:
  - Correct tile splitting for aligned and tail dimensions
  - Stable tile_id generation
  - Tile trace annotation with future-use metadata
  - Tile statistics match expected work set sizes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from joint_sim.tile_mapper import split_weight_to_tiles, TileMapper, CORE_SIZE
from joint_sim.schema import TileRecord


# PASS / FAIL：手工测试框架的全局计数器，记录通过/失败的检查项数量
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    """手工断言：记录并打印一个检查项的结果。

    作用：本文件未使用 pytest，而是用 check() 自己收集通过/失败数量，
    便于直接 `python test_tile_mapper.py` 运行。
    参数：
        name: 检查项名称（打印时展示）。
        condition: 布尔值，为 True 表示通过。
        detail: 失败时的补充说明。
    返回值：无；通过/失败数累加到全局变量 PASS/FAIL。
    """
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# -------------------------------------------------------------------
# Split weight to tiles
# -------------------------------------------------------------------

def test_aligned_weight():
    # 验证"对齐尺寸"权重的切分：64×128 权重 → 1 个 k-tile × 2 个 n-tile
    # 共 2 个 tile，且每个 tile 都不是尾部（尺寸恰好整除）。
    # 关键断言：tile 数量、k/n 索引、valid/physical_mrrs 都符合 64×64 分块。
    """64x128 weight → 1 k-tile × 2 n-tiles = 2 tiles."""
    tiles = split_weight_to_tiles("w1", K=64, N=128)
    check("aligned: 2 tiles", len(tiles) == 2)
    check("aligned: tile 0 not tail", not tiles[0].is_tail)
    check("aligned: tile 1 not tail", not tiles[1].is_tail)
    check("aligned: valid_mrrs = 4096", all(t.valid_mrrs == 4096 for t in tiles))
    check("aligned: physical_mrrs = 4096", all(t.physical_mrrs == 4096 for t in tiles))
    check("aligned: k_idx range", tiles[0].k_idx == 0 and tiles[1].k_idx == 0)
    check("aligned: n_idx range", tiles[0].n_idx == 0 and tiles[1].n_idx == 1)


def test_tail_weight():
    # 验证"边界尺寸"权重的切分：100×100 → 2 个 k-tile × 2 个 n-tile 共 4 个，
    # 其中右下角（36×36）两个维度都是余数，其余 tile 有一个维度是余数。
    # 关键断言：非尾部 tile 用满 4096 个 MRR，尾部 tile 的有效 MRR 数按余数算。
    """100×100 weight → 2 k-tiles × 2 n-tiles, tail on both dims."""
    tiles = split_weight_to_tiles("w2", K=100, N=100)
    check("tail: 4 tiles", len(tiles) == 4)
    # Tile (0,0): 64×64, no tail
    # 左上 tile：(0,0) 是 64×64，尺寸整除，不是尾部
    t00 = [t for t in tiles if t.k_idx == 0 and t.n_idx == 0][0]
    check("tail: (0,0) not tail", not t00.is_tail)
    check("tail: (0,0) 4096 MRRs", t00.valid_mrrs == 4096)
    # Tile (0,1): 64×36, tail on N
    # 右上 tile：(0,1) 是 64×36，N 维有余数（36），属于尾部
    t01 = [t for t in tiles if t.k_idx == 0 and t.n_idx == 1][0]
    check("tail: (0,1) is tail", t01.is_tail)
    check("tail: (0,1) 2304 MRRs", t01.valid_mrrs == 64 * 36)
    # Tile (1,0): 36×64, tail on K
    # 左下 tile：(1,0) 是 36×64，K 维有余数（36），属于尾部
    t10 = [t for t in tiles if t.k_idx == 1 and t.n_idx == 0][0]
    check("tail: (1,0) is tail", t10.is_tail)
    check("tail: (1,0) 2304 MRRs", t10.valid_mrrs == 36 * 64)
    # Tile (1,1): 36×36, tail on both
    # 右下 tile：(1,1) 是 36×36，两个维度都有余数
    t11 = [t for t in tiles if t.k_idx == 1 and t.n_idx == 1][0]
    check("tail: (1,1) is tail", t11.is_tail)
    check("tail: (1,1) 1296 MRRs", t11.valid_mrrs == 36 * 36)


def test_tile_id_stability():
    # 验证 tile_id 的稳定性：同一权重名、同一尺寸切分两次，
    # 生成的 tile_id 序列必须完全一致（用于跨多次实验对得上缓存命中等）。
    # 关键断言：两次切分的 id 列表相等。
    """Same weight split twice → same tile_ids."""
    t1 = split_weight_to_tiles("abc123", K=128, N=256)
    t2 = split_weight_to_tiles("abc123", K=128, N=256)
    ids1 = [t.tile_id for t in t1]
    ids2 = [t.tile_id for t in t2]
    check("tile_id stable", ids1 == ids2)


def test_empty_weight():
    # 验证空权重（K=0 或 N=0）应返回空列表，而不是报错或产生非法 tile。
    # 关键断言：两个方向为 0 时都返回 []。
    """K=0 or N=0 → empty list."""
    check("K=0 → empty", split_weight_to_tiles("w", K=0, N=64) == [])
    check("N=0 → empty", split_weight_to_tiles("w", K=64, N=0) == [])


def test_large_weight_tile_count():
    # 验证大权重的 tile 数量：512×512 → 每维 512/64=8 段，
    # 共 8×8=64 个 tile。
    # 关键断言：tile 数量为 64。
    """512×512 weight → 8×8 = 64 tiles."""
    tiles = split_weight_to_tiles("big", K=512, N=512)
    check("512×512 → 64 tiles", len(tiles) == 64)


# -------------------------------------------------------------------
# TileMapper with synthetic trace
# -------------------------------------------------------------------

def make_synthetic_trace():
    """Create a minimal trace with 2 weights, 3 accesses each."""
    records = []
    # weight_1: 128×256 → 2×4 = 8 tiles
    # weight_2: 64×128 → 1×2 = 2 tiles
    for step in range(3):
        records.append({
            "trace_version": "test",
            "op_id": f"op_w1_s{step}",
            "order": step * 2,
            "module_path": "dynamics.0",
            "op_type": "Linear",
            "op_role": "FFN_up",
            "phase": "dynamics",
            "block_kind": "temporal",
            "layer_index": 0,
            "rollout_step": step + 1,
            "call_index": 0,
            "input_shapes": (1, 128),
            "output_shape": (1, 256),
            "M": 1, "K": 128, "N": 256,
            "batch_repetitions": 1,
            "dtype": "float32",
            "input_bits": 32,
            "weight_bits": 32,
            "output_bits": 32,
            "weight_id": "weight_1",
            "weight_static": True,
            "input_bytes": 512,
            "weight_bytes": 128 * 256 * 4,
            "output_bytes": 1024,
            "dependencies": (),
        })
        records.append({
            "trace_version": "test",
            "op_id": f"op_w2_s{step}",
            "order": step * 2 + 1,
            "module_path": "dynamics.1",
            "op_type": "Linear",
            "op_role": "FFN_down",
            "phase": "dynamics",
            "block_kind": "temporal",
            "layer_index": 1,
            "rollout_step": step + 1,
            "call_index": 0,
            "input_shapes": (1, 64),
            "output_shape": (1, 128),
            "M": 1, "K": 64, "N": 128,
            "batch_repetitions": 1,
            "dtype": "float32",
            "input_bits": 32,
            "weight_bits": 32,
            "output_bits": 32,
            "weight_id": "weight_2",
            "weight_static": True,
            "input_bytes": 256,
            "weight_bytes": 64 * 128 * 4,
            "output_bytes": 512,
            "dependencies": (f"op_w1_s{step}",),
        })
    return records


def test_tile_mapper_counts():
    mapper = TileMapper()
    records = make_synthetic_trace()
    mapper.build_tile_map(records)

    check("tile_map: 2 weights", len(mapper.tile_map) == 2)
    check("tile_map: weight_1 = 8 tiles", len(mapper.tile_map["weight_1"]) == 8)
    check("tile_map: weight_2 = 2 tiles", len(mapper.tile_map["weight_2"]) == 2)
    check("total tiles = 10", mapper.tile_count() == 10)


def test_tile_trace_annotation():
    mapper = TileMapper()
    records = make_synthetic_trace()
    accesses = mapper.build_tile_trace(records)

    # 3 steps × (8 + 2) tiles = 30 accesses
    check("30 tile accesses", len(accesses) == 30)
    # Last access for each tile should have remaining_uses=0
    last_accesses = [a for a in accesses if a.remaining_uses == 0]
    check("10 last-accesses (1 per tile)", len(last_accesses) == 10)
    # First access should have remaining_uses=2
    first_w1 = [a for a in accesses if a.tile_id.startswith("weight_1_k0_n0")]
    check("weight_1_k0_n0: 3 accesses", len(first_w1) == 3)
    check("first remaining_uses=2", first_w1[0].remaining_uses == 2)
    check("last remaining_uses=0", first_w1[-1].remaining_uses == 0)


def test_tile_trace_next_use():
    mapper = TileMapper()
    records = make_synthetic_trace()
    accesses = mapper.build_tile_trace(records)

    first_w1 = [a for a in accesses if a.tile_id == "weight_1_k0_n0"]
    # next_use_order should point to the next access
    check("first has next_use", first_w1[0].next_use_order is not None)
    check("last has no next_use", first_w1[-1].next_use_order is None)


def test_tile_stats():
    mapper = TileMapper()
    records = make_synthetic_trace()
    mapper.build_tile_trace(records)
    stats = mapper.stats()

    check("stats has by_phase", "by_phase" in stats)
    check("total_unique_weights=2", stats["total_unique_weights"] == 2)
    check("total_unique_tiles=10", stats["total_unique_tiles"] == 10)


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_tile_mapper ===")
    test_aligned_weight()
    test_tail_weight()
    test_tile_id_stability()
    test_empty_weight()
    test_large_weight_tile_count()
    test_tile_mapper_counts()
    test_tile_trace_annotation()
    test_tile_trace_next_use()
    test_tile_stats()
    print(f"\n  {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
