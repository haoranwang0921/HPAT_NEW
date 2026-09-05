# ============================================================================
# 模块说明：有符号权重消融实验 CLI 的配置测试
#   "消融"（ablation）指在实验中人为去掉/替换某个因素，观察它对结果的影响。
#   本项目的 signed-weight ablation 实验比较四种映射模式（无符号 / balanced /
#   digital / dual bank）在仿真器里的表现。CLI 入口根据所选模式生成一份
#   "模式配置"（build_mode_config），本测试验证这份配置是否正确：
#     1. 无符号模式沿用旧的 64×64 映射、4KB tile，能耗系数为 1
#     2. 固定面积 compact 模式用 2KB 物理 tile
#     3. 固定面积 expanded 模式用 4KB 物理 tile
#     4. dual bank 模式模拟并行烧录，能耗系数翻倍
# ============================================================================
"""Configuration tests for the signed-weight ablation CLI."""

from joint_sim.cli.run_signed_weight_ablation import build_mode_config
from joint_sim.signed_mrr_adapter import (
    SIGNED_DUAL_BANK,
    SIGNED_FIXED_BALANCED,
    SIGNED_FIXED_DIGITAL,
    UNSIGNED,
)


def test_unsigned_mode_uses_legacy_mapper_and_4kb_tiles():
    """验证无符号模式复用旧映射器与 4KB tile，且能耗不加倍。

    验证重点：UNSIGNED 模式应映射到 unsigned_64x64、每个 tile 4KB，
    烧录能耗与保持功耗的倍率都是 1.0（没有任何额外开销）。
    关键断言：mapping_mode/weight_tile_bytes 及两个倍率字段都等于预期值。
    """
    cfg = build_mode_config(UNSIGNED, "compact")
    assert cfg.mapping_mode == "unsigned_64x64"
    assert cfg.weight_tile_bytes == 4096
    assert cfg.program_energy_multiplier == 1.0
    assert cfg.hold_power_multiplier == 1.0


def test_fixed_area_compact_uses_2kb_physical_tile_payload():
    """验证固定面积 + 紧凑存储使用 2KB 物理 tile。

    验证重点：固定面积模式下逻辑容量只有 32 列，compact 存储只放有效权重，
    因此每个物理 tile 是 2KB；能耗倍率保持 1.0。
    关键断言：mapping_mode 为 signed_row_pair_fixed_area、tile 为 2048 字节。
    """
    cfg = build_mode_config(SIGNED_FIXED_BALANCED, "compact")
    assert cfg.mapping_mode == "signed_row_pair_fixed_area"
    assert cfg.weight_tile_bytes == 2048
    assert cfg.program_energy_multiplier == 1.0
    assert cfg.hold_power_multiplier == 1.0


def test_fixed_area_expanded_uses_4kb_payload():
    """验证固定面积 + 展开存储使用 4KB 物理 tile。

    验证重点：expanded 存储按完整物理槽位对齐，每个 tile 占满 4KB。
    关键断言：mapping_mode 仍为 signed_row_pair_fixed_area、tile 为 4096 字节。
    """
    cfg = build_mode_config(SIGNED_FIXED_DIGITAL, "expanded")
    assert cfg.mapping_mode == "signed_row_pair_fixed_area"
    assert cfg.weight_tile_bytes == 4096


def test_dual_bank_models_parallel_programming_with_double_energy():
    """验证 dual bank 模式把烧录能耗与保持功耗都翻倍。

    验证重点：双份器件同时烧录正/负两套权重，因此烧录能耗倍率为 2.0；
    两块权重要一直保持，保持功耗倍率也是 2.0；映射仍沿用旧的 64×64。
    关键断言：program_energy_multiplier == 2.0、hold_power_multiplier == 2.0。
    """
    cfg = build_mode_config(SIGNED_DUAL_BANK, "compact")
    assert cfg.mapping_mode == "unsigned_64x64"
    assert cfg.weight_tile_bytes == 4096
    assert cfg.program_energy_multiplier == 2.0
    assert cfg.hold_power_multiplier == 2.0
