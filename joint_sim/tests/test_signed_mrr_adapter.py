# ============================================================================
# 模块说明：有符号 MRR 适配器（signed-MRR cost adapter）的契约测试
#   光子 MRR 阵列默认只能表示"无符号（非负）"权重，而真实神经网络的权重
#   可正可负。为支持有符号权重，本项目实现了多种映射方案：
#     UNSIGNED              ：保持原有无符号映射（最简单，但只能表示非负权重）
#     SIGNED_FIXED_BALANCED ：每个输出分成正/负两条分支（balanced，平衡式），
#                             面积不变但 ADC（模数转换）能耗减半
#     SIGNED_FIXED_DIGITAL  ：正负分支 + 电子减法，把减法成本计进数字域
#     SIGNED_DUAL_BANK      ：双份器件（面积翻倍）的直接上界方案
#   本测试验证 adapt_kernel_cost / adapt_architecture_cost 这两个"成本适配"
#   函数是否如实反映上述每种方案在时延、能耗、面积上的变化。
# ============================================================================
"""Contract tests for signed-MRR SimPhony cost adaptation."""

import math

from joint_sim.signed_mrr_adapter import (
    SIGNED_DUAL_BANK,
    SIGNED_FIXED_BALANCED,
    SIGNED_FIXED_DIGITAL,
    UNSIGNED,
    adapt_architecture_cost,
    adapt_kernel_cost,
)


class FakeKernelBackend:
    """一个假的算子后端（kernel backend），用来模拟光子核的实际行为。

    它接收 N（输出宽度）等关键字参数，然后按 N 的倍数返回一组仿真的
    时延与能耗数值。测试中用它能精确算出"适配函数应该得到什么结果"，
    从而对适配逻辑做逐项数值校验。
    """

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        # 记录每次调用参数，方便测试断言后端被请求的实际 N
        self.calls.append(kwargs)
        n = kwargs["N"]
        return {
            "compute_latency_s": n * 1e-9,
            "operand_encoding_latency_s": n * 2e-9,
            "conversion_latency_s": n * 3e-9,
            "programming_latency_s": n * 4e-9,
            "dynamic_energy_j": n * 10e-12,
            "dac_energy_j": n * 2e-12,
            "adc_energy_j": n * 4e-12,
            "laser_energy_j": n * 1e-12,
            "mrr_tuning_energy_j": n * 3e-12,
            "mrr_hold_energy_j": n * 5e-12,
            "input_bytes": kwargs["M"] * kwargs["K"],
            "output_bytes": kwargs["M"] * n,
            "iter_M": 1,
            "iter_K": 1,
            "iter_N": n // 64,
            "switching_cycles": n // 64,
            "max_cycles": n // 64,
            "utilization": 1.0,
        }


# REC：测试用的"算子记录"（record），模拟一个 K=64、N=64、每元素 8 bit 的
# 矩阵乘（M=2 个 batch），作为适配函数的统一输入。
REC = {
    "M": 2,
    "K": 64,
    "N": 64,
    "input_bits": 8,
    "weight_bits": 8,
    "output_bits": 8,
}


def test_unsigned_mode_is_a_passthrough_with_logical_metadata():
    """验证 UNSIGNED（无符号）模式只是"直通"，并附加逻辑元信息。

    验证重点：无符号模式不改变后端请求的 N（仍为 64），输出字节按
    M×N=128 计算，且映射模式被标记为 UNSIGNED。
    关键断言：后端实际请求 N=64；output_bytes==128；
    mapping_mode==UNSIGNED；physical_branch_N==64（物理分支宽度不变）。
    """
    backend = FakeKernelBackend()
    result = adapt_kernel_cost(backend, REC, UNSIGNED)

    assert backend.calls[-1]["N"] == 64
    assert result["output_bytes"] == 128
    assert result["mapping_mode"] == UNSIGNED
    assert result["physical_branch_N"] == 64


def test_balanced_fixed_area_doubles_branch_work_and_halves_raw_adc_energy():
    """验证 balanced（平衡式）固定面积方案的代价变化。

    验证重点：正负两条分支各占半个面积，等价于把后端请求的 N 翻倍为 128；
    两条分支的 ADC 加总后除以 2 即每条分支的能耗，因此 raw adc_energy 减半；
    dynamic_energy 相应扣除被减掉的 ADC 部分。
    关键断言：后端请求 N=128、physical_branch_N==128、
    adc_energy_j==128*4e-12*0.5、iter_N==2（N 维度分两次迭代）。
    """
    backend = FakeKernelBackend()
    result = adapt_kernel_cost(backend, REC, SIGNED_FIXED_BALANCED)

    assert backend.calls[-1]["N"] == 128
    assert result["output_bytes"] == 128
    assert result["physical_branch_N"] == 128
    assert result["adc_energy_j"] == 128 * 4e-12 * 0.5
    assert result["dynamic_energy_j"] == 128 * 10e-12 - 128 * 4e-12 * 0.5
    assert result["iter_N"] == 2


def test_digital_sub_keeps_two_branch_adc_energy_and_reports_subtraction_cost():
    """验证数字减法（digital subtract）方案把减法成本计入数字域。

    验证重点：与 balanced 方案不同，这里保留两条分支完整的 ADC 能耗，
    另把"两分支相减"这一步的电子减法能耗与延迟单独记账。
    关键断言：adc_energy_j 保持 128*4e-12；digital_sub_energy_j==2*64*0.25e-12；
    dynamic_energy 在其基础上加回减法能耗；digital_sub_latency_s==2*0.5e-9。
    """
    backend = FakeKernelBackend()
    result = adapt_kernel_cost(
        backend,
        REC,
        SIGNED_FIXED_DIGITAL,
        digital_sub_energy_per_output_j=0.25e-12,
        digital_sub_latency_per_vector_s=0.5e-9,
    )

    assert backend.calls[-1]["N"] == 128
    assert result["adc_energy_j"] == 128 * 4e-12
    assert result["digital_sub_energy_j"] == 2 * 64 * 0.25e-12
    assert result["dynamic_energy_j"] == 128 * 10e-12 + result["digital_sub_energy_j"]
    assert result["digital_sub_latency_s"] == 2 * 0.5e-9


def test_dual_bank_preserves_latency_and_doubles_dynamic_physical_energy():
    """验证双份器件（dual bank）方案的时延与能耗特征。

    验证重点：双份器件可以让正/负分支并行算，因此计算时延不增加（仍为
    64e-9），但物理上的动态能耗翻倍（因为用了两套器件）。
    关键断言：后端请求 N=64、compute_latency_s==64e-9、
    dynamic_energy_j==2*64*10e-12、output_bytes==128。
    """
    backend = FakeKernelBackend()
    result = adapt_kernel_cost(backend, REC, SIGNED_DUAL_BANK)

    assert backend.calls[-1]["N"] == 64
    assert result["compute_latency_s"] == 64e-9
    assert result["dynamic_energy_j"] == 2 * 64 * 10e-12
    assert result["output_bytes"] == 128


def test_fixed_area_architecture_conserves_mrr_and_pd_counts():
    """验证固定面积方案在架构层面"器件数量守恒"。

    验证重点：balanced/digital 两种固定面积方案不增加 MRR 与光电探测器
    （PD，photodetector）数量；balanced 的 ADC 数量减半（两分支共用一半
    ADC），且不引入额外分光损耗；digital 保留全部 ADC。
    关键断言：mrr_count/pd_count 与基准一致；balanced.adc_count 减半；
    balanced.extra_split_loss_db==0.0；digital.adc_count 保持不变。
    """
    base = {
        "mrr_count": 393216,
        "pd_count": 6144,
        "dac_count": 12288,
        "adc_count": 6144,
        "pic_area_um2": 10.0,
        "rf_eic_area_um2": 5.0,
        "total_area_um2": 15.0,
        "core_insertion_loss_db": 9.7,
        "laser_wall_plug_power_w": 2.69,
    }

    balanced = adapt_architecture_cost(base, SIGNED_FIXED_BALANCED)
    digital = adapt_architecture_cost(base, SIGNED_FIXED_DIGITAL)

    assert balanced["mrr_count"] == base["mrr_count"]
    assert balanced["pd_count"] == base["pd_count"]
    assert balanced["adc_count"] == base["adc_count"] // 2
    assert balanced["extra_split_loss_db"] == 0.0
    assert digital["adc_count"] == base["adc_count"]


def test_dual_bank_architecture_is_explicit_upper_bound():
    """验证双份器件（dual bank）方案是一个明确的成本上界。

    验证重点：面积、器件数全部翻倍，且额外引入 3dB 分光损耗
    （10*log10(2)≈3.01），并显式标记 is_upper_bound=True，
    表示这是一个"最坏情况"的估算上界。
    关键断言：mrr/pd/面积均翻倍；extra_split_loss_db 接近 10*log10(2)；
    is_upper_bound 为 True。
    """
    base = {
        "mrr_count": 393216,
        "pd_count": 6144,
        "dac_count": 12288,
        "adc_count": 6144,
        "pic_area_um2": 10.0,
        "rf_eic_area_um2": 5.0,
        "total_area_um2": 15.0,
        "core_insertion_loss_db": 9.7,
        "laser_wall_plug_power_w": 2.69,
    }

    dual = adapt_architecture_cost(base, SIGNED_DUAL_BANK)

    assert dual["mrr_count"] == 2 * base["mrr_count"]
    assert dual["pd_count"] == 2 * base["pd_count"]
    assert dual["total_area_um2"] == 2 * base["total_area_um2"]
    assert math.isclose(dual["extra_split_loss_db"], 10 * math.log10(2))
    assert dual["is_upper_bound"] is True
