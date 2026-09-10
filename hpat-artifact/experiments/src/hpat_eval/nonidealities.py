"""非理想性（non-idealities）误差注入：模拟光子芯片"不完美"造成的计算误差。

背景：真实光子芯片并不理想——PD/TIA 有电噪声、相邻波长的光会串扰
（crosstalk）、ADC/DAC 量化有精度损失、激光经过波导会有插入损耗、
波长会随温度漂移、微环（MRR）制造尺寸有偏差、运行中会热漂移。
这些非理想性都会让"光子算出来的结果"偏离"数学上的精确结果"。

本文件用"代理仿真"（proxy simulation）：每次试验随机生成一组输入 x
和权重 w，算出精确点积 y = x·w，再施加某种非理想性误差得到 y_hat，
统计相对误差 |y_hat - y| / |y|，最后输出"误差敏感度分析表"。
注意这里的精度指标只是"点积相对误差"的代理分数，不是 ImageNet 精度。

易混淆点：本文件是非理想性的"通用建模"，injection_boundary.py 是
"误差注入边界"（哪些算子允许注入到光子端），两者是不同层面。
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from .mobilevit_loader import variants_from_config


def dot(a: list[float], b: list[float]) -> float:
    """计算两个等长列表的点积（向量内积）。

    用于在随机试验里模拟"一次光子矩阵乘法中的一行"。

    :param a: 向量 a。
    :param b: 向量 b。
    :return: 点积标量。
    """
    return sum(x * y for x, y in zip(a, b))


def roll_one(x: list[float]) -> list[float]:
    """把列表整体向右滚动一位（最后一个元素移到开头）。

    用于模拟 WDM 波分复用中"相邻波长串扰"：某个波长通道的部分
    光信号漏进相邻通道，相当于用相邻位置的数据做混合。

    例：roll_one([a, b, c]) == [c, a, b]。

    :param x: 输入列表。
    :return: 滚动一位后的列表。
    """
    return [x[-1]] + x[:-1] if x else x


def quantize_unit(x: float, bits: int) -> float:
    """把 [-1, 1] 区间的数量化（quantize）到 bits 位精度。

    原理：把实数 x 映射到 2^bits - 1 个量化阶梯上的最近阶梯，
    再还原回 [-1, 1] 区间。例如 bits=4 时只有 15 个可表示的值，
    中间值会被就近取整，这就是 ADC/DAC 有限精度的本质。

    :param x: 待量化的数。
    :param bits: 量化位数。
    :return: 量化后的值（仍在 [-1, 1] 内）。
    """
    levels = (1 << bits) - 1
    clipped = min(1.0, max(-1.0, x))  # 先截断到 [-1,1]，防止越界
    q = round((clipped + 1.0) * levels / 2.0)  # 映射到阶梯序号并就近取整
    return (2.0 * q / levels) - 1.0  # 还原回对称区间


def summarize(values: list[float]) -> dict[str, float]:
    """汇总一批相对误差样本的统计量（均值/中位数/95 分位/最大值）。

    :param values: 误差样本列表。
    :return: {"mean","median","p95","max"} 统计字典（空样本全为 0.0）。
    """
    ordered = sorted(values)
    if not ordered:
        return {key: 0.0 for key in ["mean", "median", "p95", "max"]}
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": max(ordered),
    }


def _proxy_score(mean_relative_error: float, severity: float = 1.0) -> float:
    """把平均相对误差换算成一个 0~100 的"代理分数"（100 最好）。

    公式：100 - 100 × severity × min(误差, 1)。severity 用于调节
    不同非理想性的"惩罚力度"——误差超过 100% 时按 100% 封顶。

    :param mean_relative_error: 平均相对误差（0~1 及以上）。
    :param severity: 惩罚系数（默认 1.0）。
    :return: 代理分数（0~100）。
    """
    return max(0.0, 100.0 - 100.0 * severity * min(mean_relative_error, 1.0))


def precision_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """ADC/DAC 位宽（精度）敏感度分析：位宽越低误差越大。

    每档位宽做 trials 次随机点积试验，统计相对误差；同时按经验公式
    (位宽/8)^1.35 估算"转换器能耗倍率"（相对 8 位），用于说明：
    降低位宽省能耗，但会损失精度——论文需要在这个权衡上给出数据。

    :param config: 实验配置（读 seed / precision_bit_width_sweep 等）。
    :return: 精度敏感度分析表（每行一档位宽 × 一个模型变体）。
    """
    rng = random.Random(int(config.get("seed", 1)))  # 固定种子保证可复现
    trials = int(config.get("nonideality", {}).get("trials", 200))
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        dim = int(variant["embedding_dim"])  # 点积维数（取嵌入维度）
        baseline_energy = float(config.get("energy_envelope_mj", {}).get(variant["variant"], 1.0))
        for bits in config.get("precision_bit_width_sweep", [4, 6, 8, 10, 12]):
            errors: list[float] = []
            for _ in range(trials):
                # 每次试验随机生成 x、w，先算精确点积 y
                x = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                w = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                y = dot(x, w)
                # 对 x、w 分别量化到 bits 位，再算"带量化误差"的点积 y_hat
                xq = [quantize_unit(v, int(bits)) for v in x]
                wq = [quantize_unit(v, int(bits)) for v in w]
                y_hat = dot(xq, wq)
                # 相对误差：绝对偏差 / |精确值|（精确值极小则退避 1e-9 防除零）
                errors.append(abs(y_hat - y) / max(abs(y), 1e-9))
            stats = summarize(errors)
            # 转换器能耗随位宽的经验缩放（相对 8 位，指数 1.35）
            converter_energy_multiplier = (float(bits) / 8.0) ** 1.35
            rows.append(
                {
                    "variant": variant["variant"],
                    "embedding_dim": dim,
                    "adc_dac_bits": int(bits),
                    "trials": trials,
                    "mean_relative_error": f"{stats['mean']:.8f}",
                    "p95_relative_error": f"{stats['p95']:.8f}",
                    "proxy_score_percent": f"{_proxy_score(stats['mean'], severity=0.25):.4f}",
                    "converter_energy_multiplier_vs_8bit": f"{converter_energy_multiplier:.6f}",
                    "estimated_energy_mj": f"{baseline_energy * converter_energy_multiplier:.6f}",
                    "metric_kind": "synthetic transfer proxy, not ImageNet top-1",  # 明确：不是 ImageNet 精度
                    "evidence_label": "analytical/modelled precision sensitivity",
                }
            )
    return rows


def nonideality_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """七类非理想性的误差注入敏感度分析（本文件核心函数）。

    每类非理想性有一个"扫描变量"和一组待扫取值；对每个取值做 trials
    次随机点积试验。七类及其注入方式：
    - gaussian_pd_tia_noise：给读出值加高斯噪声（幅度按 signal/255 缩放）；
    - wdm_adjacent_crosstalk：把相邻波长通道的光混进来（roll_one 实现）；
    - uniform_converter_quantization：ADC/DAC 量化；
    - insertion_loss：激光插损，输出等比衰减，激光能耗按 10^(dB/10) 放大；
    - wavelength_detuning：波长失谐，传输系数下降（平方律近似）；
    - mrr_variation：MRR 制造误差 → 权重乘性高斯扰动；
    - thermal_drift：热漂移 → 结果衰减 + 重调谐开销。

    :param config: 实验配置（读 nonideality 段的参数）。
    :return: 非理想性敏感度分析表。
    """
    rng = random.Random(int(config.get("seed", 1)) + 99)  # 偏移种子，与 precision_rows 结果互不重叠
    cfg = config.get("nonideality", {})
    trials = int(cfg.get("trials", 200))
    rows: list[dict[str, Any]] = []
    # 扫描计划：(效果名, 扫描变量名, 待扫取值列表)
    sweep_plan = [
        ("gaussian_pd_tia_noise", "noise_lsb", cfg.get("noise_lsb", [0.0, 0.25, 0.5, 1.0])),
        ("wdm_adjacent_crosstalk", "crosstalk_alpha", cfg.get("crosstalk_alpha", [0.0, 0.01, 0.03, 0.05])),
        ("uniform_converter_quantization", "quant_bits", cfg.get("quant_bits", [4, 6, 8, 10])),
        ("insertion_loss", "path_db", cfg.get("insertion_loss_db", [0.0, 1.0, 2.0, 3.0])),
        ("wavelength_detuning", "delta_lambda_pm", cfg.get("detuning_pm", [0.0, 5.0, 10.0, 20.0])),
        ("mrr_variation", "sigma_percent", cfg.get("mrr_variation_percent", [0.0, 1.0, 3.0, 5.0])),
        ("thermal_drift", "delta_c", cfg.get("thermal_drift_c", [0.0, 2.0, 5.0, 10.0])),
    ]
    for variant in variants_from_config(config):
        dim = int(variant["embedding_dim"])
        for effect, variable, values in sweep_plan:
            for value in values:
                errors: list[float] = []
                energy_multiplier = 1.0                 # 激光能耗倍率（默认不放大）
                retuning_overhead_percent = 0.0         # 重调谐开销占比（默认 0）
                for _ in range(trials):
                    x = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                    w = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                    y = dot(x, w)
                    # 按非理想性类别分别注入误差，得到带误差的结果 y_hat
                    if effect == "gaussian_pd_tia_noise":
                        # 噪声幅度按 |y|（与信号同量级）缩放，模拟读出信噪比下降
                        scale = max(abs(y), 1.0) / 255.0
                        y_hat = y + rng.gauss(0.0, float(value) * scale)
                    elif effect == "wdm_adjacent_crosstalk":
                        # 每个输入都混入"相邻通道"的同位置值：α×相邻 + (1-α)×自身
                        mixed = [(1.0 - float(value)) * a + float(value) * b for a, b in zip(x, roll_one(x))]
                        y_hat = dot(mixed, w)
                    elif effect == "uniform_converter_quantization":
                        # 量化误差：x、w 都量化到 value 位
                        xq = [quantize_unit(v, int(value)) for v in x]
                        wq = [quantize_unit(v, int(value)) for v in w]
                        y_hat = dot(xq, wq)
                    elif effect == "insertion_loss":
                        # dB 转线性衰减：10^(-dB/20)；激光能耗按 10^(dB/10) 放大
                        loss_linear = 10.0 ** (-float(value) / 20.0)
                        y_hat = y * loss_linear
                        energy_multiplier = 10.0 ** (float(value) / 10.0)
                    elif effect == "wavelength_detuning":
                        # 失谐误差按 (Δλ/40pm)² 近似上升，截断到最多 50% 衰减
                        transfer_error = (float(value) / 40.0) ** 2
                        y_hat = y * (1.0 - min(0.5, transfer_error))
                    elif effect == "mrr_variation":
                        # 制造偏差 → 权重增益随机扰动（标准差为 sigma%）
                        y_hat = y * (1.0 + rng.gauss(0.0, float(value) / 100.0))
                    else:  # thermal_drift
                        # 热漂移 → 结果衰减（最多 40%）+ 每度 0.8% 的重调谐开销
                        drift = float(value)
                        y_hat = y * (1.0 - min(0.4, drift / 100.0))
                        retuning_overhead_percent = drift * 0.8
                    errors.append(abs(y_hat - y) / max(abs(y), 1e-9))
                stats = summarize(errors)
                # 证据标签：前三类有解析模型支撑，其余标为"解析灵敏度/待未来验证"
                label = "modelled selected non-ideality" if effect in {
                    "gaussian_pd_tia_noise",
                    "wdm_adjacent_crosstalk",
                    "uniform_converter_quantization",
                } else "analytical sensitivity / future validation"
                rows.append(
                    {
                        "variant": variant["variant"],
                        "effect": effect,
                        "sweep_variable": variable,
                        "sweep_value": value,
                        "trials": trials,
                        "mean_relative_error": f"{stats['mean']:.8f}",
                        "p95_relative_error": f"{stats['p95']:.8f}",
                        "proxy_score_percent": f"{_proxy_score(stats['mean'], severity=0.30):.4f}",
                        "laser_energy_multiplier": f"{energy_multiplier:.6f}",
                        "retuning_overhead_percent": f"{retuning_overhead_percent:.4f}",
                        "metric_kind": "synthetic/proxy transfer sensitivity, not silicon robustness",
                        "evidence_label": label,
                    }
                )
    return rows


# 精度分析表的列名清单
PRECISION_FIELDS = [
    "variant",
    "embedding_dim",
    "adc_dac_bits",
    "trials",
    "mean_relative_error",
    "p95_relative_error",
    "proxy_score_percent",
    "converter_energy_multiplier_vs_8bit",
    "estimated_energy_mj",
    "metric_kind",
    "evidence_label",
]


# 非理想性分析表的列名清单
NONIDEALITY_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "trials",
    "mean_relative_error",
    "p95_relative_error",
    "proxy_score_percent",
    "laser_energy_multiplier",
    "retuning_overhead_percent",
    "metric_kind",
    "evidence_label",
]
