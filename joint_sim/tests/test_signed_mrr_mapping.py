# ============================================================================
# 模块说明：有符号"行对权重映射"（signed row-pair weight mapping）的契约测试
#   MRR 芯片上一块固定的物理器件面积能容纳的权重是有限的，超大权重矩阵
#   必须被切分成若干小"tile"（瓦片/分块）逐个计算。有符号映射把一行权重
#   拆成正/负两个物理槽位，从而在不增加面积的前提下表示正负权重。
#   本测试验证：
#     1. 512×512 权重在固定面积模式下的切分结果（128 个物理 tile）
#     2. 边界尺寸（tail，尾巴/余数）的切分是否正确
#     3. compact / expanded（紧凑 / 展开）两种存储模式的字节数差异
#     4. 无符号旧模式的向后兼容性
#     5. TileMapper 对外暴露的映射语义（logical_capacity_n 等）
#     6. 非法映射配置必须被拒绝
# ============================================================================
"""Contract tests for signed row-pair weight mapping."""

import pytest

from joint_sim.tile_mapper import TileMapper, split_weight_to_tiles


def test_signed_fixed_area_512_square_uses_128_physical_tiles():
    """验证 512×512 权重在固定面积模式下的切分数量与容量。

    验证重点：signed_row_pair_fixed_area 模式下每个 tile 的逻辑容量上限是
    32 列（logical_capacity_n=32），因此 512 列需要 512/32=16 段；
    每个 tile 再带上正负两行的物理槽位（valid_mrrs=4096 等价于逻辑 64×64）。
    关键断言：共 128 个 tile；每个 tile physical_mrrs/valid_mrrs 均为 4096；
    effective_N<=32；总存储字节恰等于 512*512 个权重元素。
    """
    tiles = split_weight_to_tiles(
        "w",
        K=512,
        N=512,
        mapping_mode="signed_row_pair_fixed_area",
        storage_mode="compact",
    )

    assert len(tiles) == 128
    assert all(tile.physical_mrrs == 4096 for tile in tiles)
    assert all(tile.effective_N <= 32 for tile in tiles)
    assert all(tile.valid_mrrs == 4096 for tile in tiles)
    assert sum(tile.storage_bytes for tile in tiles) == 512 * 512


def test_signed_fixed_area_tail_counts_two_physical_slots_per_weight():
    """验证尾部（tail，余数）tile 的正负双槽位统计。

    验证重点：65×33 的权重除了完整 tile 外会留下 1 行×1 列的余数 tile；
    有符号行对映射下，该余数 tile 需占用正、负两个物理槽位，
    因此 valid_mrrs==2（两个槽位都算有效），存储只需 1 字节。
    关键断言：共 4 个 tile；余数 tile 的 effective_K/effective_N 均为 1、
    valid_mrrs==2、storage_bytes==1、is_tail 为 True。
    """
    tiles = split_weight_to_tiles(
        "tail",
        K=65,
        N=33,
        mapping_mode="signed_row_pair_fixed_area",
        storage_mode="compact",
    )

    assert len(tiles) == 4
    tail = next(tile for tile in tiles if tile.k_idx == 1 and tile.n_idx == 1)
    assert tail.effective_K == 1
    assert tail.effective_N == 1
    assert tail.valid_mrrs == 2
    assert tail.storage_bytes == 1
    assert tail.is_tail


def test_expanded_storage_doubles_compact_storage():
    """验证 expanded（展开）存储的字节数是 compact（紧凑）的两倍。

    验证重点：紧凑存储只保存每个 tile 真正用到的权重；展开存储则按完整
    物理槽位（64×64=4096 个权重）对齐，因此空间翻倍。
    关键断言：同一权重下 compact 首 tile 为 2048 字节、expanded 为 4096 字节。
    """
    compact = split_weight_to_tiles(
        "w",
        K=64,
        N=32,
        mapping_mode="signed_row_pair_fixed_area",
        storage_mode="compact",
    )
    expanded = split_weight_to_tiles(
        "w",
        K=64,
        N=32,
        mapping_mode="signed_row_pair_fixed_area",
        storage_mode="expanded",
    )

    assert compact[0].storage_bytes == 2048
    assert expanded[0].storage_bytes == 4096


def test_unsigned_mode_remains_backward_compatible():
    """验证无符号旧模式保持向后兼容。

    验证重点：不传 mapping_mode 的旧调用（默认 unsigned_64x64、compact）
    与显式传入这两个参数的调用结果必须完全一致。
    关键断言：legacy == explicit；64×128 权重切成 2 个 tile；
    总存储字节等于 64*128。
    """
    legacy = split_weight_to_tiles("w", K=64, N=128)
    explicit = split_weight_to_tiles(
        "w",
        K=64,
        N=128,
        mapping_mode="unsigned_64x64",
        storage_mode="compact",
    )

    assert legacy == explicit
    assert len(explicit) == 2
    assert sum(tile.storage_bytes for tile in explicit) == 64 * 128


def test_mapper_manifest_exposes_mapping_semantics():
    """验证 TileMapper 对外暴露的映射元信息。

    验证重点：signed_row_pair_fixed_area 模式把逻辑容量上限暴露为 32 列，
    并记录当前使用的映射与存储模式。
    关键断言：logical_capacity_n==32、mapping_mode/storage_mode 与构造一致。
    """
    mapper = TileMapper(
        mapping_mode="signed_row_pair_fixed_area",
        storage_mode="expanded",
    )

    assert mapper.logical_capacity_n == 32
    assert mapper.mapping_mode == "signed_row_pair_fixed_area"
    assert mapper.storage_mode == "expanded"


@pytest.mark.parametrize("field,value", [
    ("mapping_mode", "unknown"),
    ("storage_mode", "unknown"),
])
def test_invalid_mapping_configuration_is_rejected(field, value):
    """验证非法映射/存储配置会被拒绝。

    验证重点：mapping_mode 或 storage_mode 传未知值（如 "unknown"）
    时，TileMapper 构造必须抛 ValueError。
    关键断言：pytest.raises(ValueError) 捕获构造异常。
    """
    kwargs = {field: value}
    with pytest.raises(ValueError):
        TileMapper(**kwargs)
