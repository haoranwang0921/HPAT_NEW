"""Cost adaptation for signed weights on non-coherent MRR intensity arrays.

The existing SimPhony HPAT model exposes 64 physical output rows per core.
Fixed-area row pairing maps two physical rows to one signed logical output;
therefore a logical KxN GEMM is costed as Kx(2N) physical branch work.  The
adapter keeps logical tensor traffic separate from physical branch activity.
"""
# =============================================================================
# 中文阅读说明
# =============================================================================
# 背景：光强调制阵列只能表示非负的"光强度"，无法直接表示负数权重。
# 本文件负责把带符号（可正可负）的权重 GEMM 映射到物理光阵列上，并给出
# 对应三种映射模式的成本换算：
#   - unsigned_64x64                 （不做有符号处理，只作基准/对比）
#   - signed_row_pair_fixed_area_balanced  两个物理行表示一个有符号输出
#                                          （固定面积行配对），模拟"光学减"
#   - signed_row_pair_fixed_area_digital_sub 同上，但在电子侧做数字相减
#   - signed_dual_bank_fixed_throughput     用两套物理阵列分别算正、负权重
# 关键思想：逻辑上仍是 [M,K]x[K,N]，但物理阵列实际计算的分支数是
# K x (2N)（负权重拆成正负两路）；本适配器把"逻辑张量流量"与"物理分支
# 活动量"分开记账，避免重复计能耗。
# =============================================================================

from __future__ import annotations

import math
from typing import Callable, Dict


# 四种映射模式的名字。SIGNED_MODES 是它们的集合，供参数校验用。
UNSIGNED = "unsigned_64x64"
SIGNED_FIXED_BALANCED = "signed_row_pair_fixed_area_balanced"
SIGNED_FIXED_DIGITAL = "signed_row_pair_fixed_area_digital_sub"
SIGNED_DUAL_BANK = "signed_dual_bank_fixed_throughput"

SIGNED_MODES = {
    UNSIGNED,
    SIGNED_FIXED_BALANCED,
    SIGNED_FIXED_DIGITAL,
    SIGNED_DUAL_BANK,
}


def _logical_output_bytes(rec: dict) -> int:
    """计算"逻辑"输出张量占的字节数（按 M x N x 位宽，再除以 8 得字节）。"""
    return (
        int(rec["M"])
        * int(rec["N"])
        * int(rec.get("output_bits", 8))
        // 8
    )


def adapt_kernel_cost(
    kernel_cost_fn: Callable[..., Dict[str, float]],
    rec: dict,
    mode: str,
    *,
    digital_sub_energy_per_output_j: float = 0.0,
    digital_sub_latency_per_vector_s: float = 0.0,
) -> dict:
    # 中文说明：这是"逻辑 GEMM -> 物理 GEMM"的成本换算入口。
    # 输入 rec 描述一个逻辑矩阵乘 [M,K]x[K,N]，mode 指定负权重的编码方式；
    # kernel_cost_fn 是 SimPhony 的接口（接受 M/K/N/位宽/dataflow 关键字）。
    # 对于固定面积行配对（balanced/digital），把 N 翻倍成 2N 再请求物理成本；
    # 最后按模式微调各项能耗/时延，并把映射信息附到结果里。
    # digital_sub_* 两个参数默认全 0，表示"数字相减"模式在真正标定数字电路
    # 之前，先作为显式标注的下界（lower bound）使用。
    """Return a SimPhony-compatible cost dictionary for one logical GEMM.

    ``kernel_cost_fn`` must accept SimPhony's keyword API.  Digital subtraction
    defaults to zero additional cost so the corresponding mode is an explicitly
    labelled lower bound until a target digital implementation is calibrated.
    """
    if mode not in SIGNED_MODES:
        raise ValueError(f"unsupported signed MRR mode: {mode!r}")
    if digital_sub_energy_per_output_j < 0:
        raise ValueError("digital_sub_energy_per_output_j must be non-negative")
    if digital_sub_latency_per_vector_s < 0:
        raise ValueError("digital_sub_latency_per_vector_s must be non-negative")

    # 逻辑矩阵的三个维度
    logical_m = int(rec["M"])
    logical_k = int(rec["K"])
    logical_n = int(rec["N"])
    # 物理 N：行配对模式下每 2 个物理行才表达 1 个逻辑输出，所以物理 N=2N；
    # 双库/无符号模式下物理 N 就等于逻辑 N。
    physical_n = (
        logical_n * 2
        if mode in {SIGNED_FIXED_BALANCED, SIGNED_FIXED_DIGITAL}
        else logical_n
    )
    request = {
        "M": logical_m,
        "K": logical_k,
        "N": physical_n,
        "input_bits": int(rec.get("input_bits", 8)),
        "weight_bits": int(rec.get("weight_bits", 8)),
        "output_bits": int(rec.get("output_bits", 8)),
        "dataflow": "weight_stationary",
    }
    # 调用 SimPhony 拿到物理成本，再补充映射元信息
    result = dict(kernel_cost_fn(**request))
    result["mapping_mode"] = mode
    result["logical_N"] = logical_n
    result["physical_branch_N"] = physical_n
    result["output_bytes"] = _logical_output_bytes(rec)
    result["digital_sub_energy_j"] = 0.0
    result["digital_sub_latency_s"] = 0.0
    result["is_lower_bound"] = False
    result["is_upper_bound"] = False

    if mode == SIGNED_FIXED_BALANCED:
        # "光学减"模式：假设 ADC 转换有一半是冗余（两条物理行合成一路输出），
        # 因此把 ADC 能耗减半，并从总动态能耗里扣掉相应部分。
        raw_adc = float(result.get("adc_energy_j", 0.0))
        adc_correction = raw_adc * 0.5
        result["adc_energy_j"] = adc_correction
        result["dynamic_energy_j"] = max(
            0.0,
            float(result.get("dynamic_energy_j", 0.0))
            - (raw_adc - adc_correction),
        )
    elif mode == SIGNED_FIXED_DIGITAL:
        # "数字相减"模式：光阵列照常算，另在电子侧做一次数字减法。
        # 额外能耗 = 每个输出元素一个减法成本 x 输出元素总数；
        # 额外时延 = 每行一个减法时延 x 行数。
        sub_energy = (
            logical_m * logical_n * digital_sub_energy_per_output_j
        )
        sub_latency = logical_m * digital_sub_latency_per_vector_s
        result["digital_sub_energy_j"] = sub_energy
        result["digital_sub_latency_s"] = sub_latency
        result["dynamic_energy_j"] = (
            float(result.get("dynamic_energy_j", 0.0)) + sub_energy
        )
        result["conversion_latency_s"] = (
            float(result.get("conversion_latency_s", 0.0)) + sub_latency
        )
        # 数字减法参数全为 0 时，本模式只是一个"标称下界"，如实标注出来
        result["is_lower_bound"] = (
            digital_sub_energy_per_output_j == 0.0
            and digital_sub_latency_per_vector_s == 0.0
        )
    elif mode == SIGNED_DUAL_BANK:
        # "双库"模式：正、负权重各用一套物理阵列，器件数量翻倍，
        # 所有与器件规模成正比的能耗项也翻倍（因此是成本上界）。
        for key in (
            "dynamic_energy_j",
            "dac_energy_j",
            "adc_energy_j",
            "laser_energy_j",
            "mrr_tuning_energy_j",
            "mrr_hold_energy_j",
        ):
            result[key] = 2.0 * float(result.get(key, 0.0))
        result["is_upper_bound"] = True

    return result


def adapt_architecture_cost(base: dict, mode: str) -> dict:
    # 中文说明：根据映射模式，把"基准架构"的器件数量、面积、损耗等系统级
    # 参数换算成对应模式的架构成本。双库模式器件/面积/激光功率翻倍并附加
    # 3dB 分光损耗；行配对模式把每个核心的逻辑输出数从 64 降到 32，ADC 减半。
    """Apply device-count and fanout semantics to a SimPhony architecture."""
    if mode not in SIGNED_MODES:
        raise ValueError(f"unsupported signed MRR mode: {mode!r}")

    result = dict(base)
    result.update({
        "mapping_mode": mode,
        "logical_outputs_per_core": 64,
        "extra_split_loss_db": 0.0,
        "is_upper_bound": False,
    })

    if mode in {SIGNED_FIXED_BALANCED, SIGNED_FIXED_DIGITAL}:
        # 行配对：64 个物理输出行 = 32 个逻辑输出；balanced 模式下 ADC 减半
        result["logical_outputs_per_core"] = 32
        if mode == SIGNED_FIXED_BALANCED:
            result["adc_count"] = int(base.get("adc_count", 0)) // 2
    elif mode == SIGNED_DUAL_BANK:
        # 双库：器件数量与面积全部翻倍，激光壁插功率翻倍，
        # 分光带来 10*log10(2)≈3.01 dB 的额外插入损耗
        for key in ("mrr_count", "pd_count", "dac_count", "adc_count"):
            result[key] = 2 * int(base.get(key, 0))
        for key in ("pic_area_um2", "rf_eic_area_um2", "total_area_um2"):
            result[key] = 2.0 * float(base.get(key, 0.0))
        result["extra_split_loss_db"] = 10.0 * math.log10(2.0)
        result["core_insertion_loss_db"] = (
            float(base.get("core_insertion_loss_db", 0.0))
            + result["extra_split_loss_db"]
        )
        result["laser_wall_plug_power_w"] = (
            2.0 * float(base.get("laser_wall_plug_power_w", 0.0))
        )
        result["is_upper_bound"] = True

    return result
