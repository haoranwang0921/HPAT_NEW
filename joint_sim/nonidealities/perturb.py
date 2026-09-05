"""
HPAT 非理想性扰动函数，移植自
hpat-artifact/experiments/scripts/run_nonideality_accuracy_sweep.py:_make_perturb_fn()

只保留 LPWM 实验 3 选择的五类误差。

中文阅读提示：这些函数不改变硬件调度；它们在模型输出上注入器件误差，
用于观察精度损失。每种 effect 都可独立扫描。
"""
from __future__ import annotations

import torch


# ── Effect 1: PD/TIA Gaussian noise ──────────────────────────────────────────
def _perturb_pd_tia(output: torch.Tensor, value: float) -> torch.Tensor:
    # 中文说明：误差 1——光电探测器(PD)/跨阻放大器(TIA)的高斯噪声。
    # 以"输出信号的均方根(RMS)为基准"按比例加噪声：value 的单位是 LSB
    # （最低有效位），除以 255 得到相对幅度，噪声强度随信号大小变化。
    rms = torch.sqrt(torch.clamp(output.detach().float().pow(2).mean(), min=1e-12))
    scale = rms * value / 255.0
    return output + torch.randn_like(output) * scale.to(output.dtype)


# ── Effect 2: WDM adjacent-channel crosstalk ─────────────────────────────────
def _perturb_wdm_crosstalk(
    output: torch.Tensor, value: float, site_count: int
) -> torch.Tensor:
    # 中文说明：误差 2——波分复用(WDM)相邻波长通道的串扰。
    # 每个特征值泄漏一部分到相邻通道：输出 = 自己*(1-2α) + 前通道*α + 后通道*α。
    # α 由 value（串扰强度）除以通道数得到并限制在 [0, 0.49]，保证不越界。
    # torch.roll 实现"把整条特征轴平移一位"，从而拿到相邻通道的值。
    # Feature/channel axis is the last dim for Linear outputs
    if output.ndim >= 2:
        feat_axis = -1
    else:
        feat_axis = 0

    if output.shape[feat_axis] <= 1:
        return output

    alpha = min(max(value / max(site_count, 1), 0.0), 0.49)
    return (
        (1.0 - 2.0 * alpha) * output
        + alpha * torch.roll(output, shifts=1, dims=feat_axis)
        + alpha * torch.roll(output, shifts=-1, dims=feat_axis)
    )


# ── Effect 3: MRR process variation ──────────────────────────────────────────
def _perturb_mrr_variation(
    output: torch.Tensor, value: float, site_count: int,
    pattern_cache: dict, module_name: str,
) -> torch.Tensor:
    # 中文说明：误差 3——微环(MRR)制造工艺偏差。
    # 每颗微环的谐振波长略有不同（制造误差），导致权重乘性偏差。
    # sigma 随 value(百分比)增大、随参与器件数开方减小（平均效应）；
    # 每通道的偏差模式用缓存生成一次（同一模块同一形状用同一套随机偏差，
    # 保证可复现、保证"器件偏差是固定的"这一物理事实）。
    sigma = value / (100.0 * (max(site_count, 1) ** 0.5))

    # Static per-channel pattern, cached per module
    axis = -1 if output.ndim >= 2 else 0
    shape = [1] * output.ndim
    shape[axis] = output.shape[axis]
    key = (module_name, tuple(shape), str(output.device), str(output.dtype))
    if key not in pattern_cache:
        pattern_cache[key] = torch.randn(shape, device=output.device, dtype=output.dtype)

    # 输出 = 原输出 * (1 + 固定偏差模式 * sigma)
    return output * (1.0 + pattern_cache[key] * sigma)


# ── Effect 4: Thermal drift (closed-loop compensation proxy) ─────────────────
def _perturb_thermal_drift(
    output: torch.Tensor, value: float,
    pattern_cache: dict, module_name: str,
    compensation_efficiency: float = 0.95,
    sigma_gain_per_c: float = 0.02,
    loss_per_c: float = 0.002,
) -> torch.Tensor:
    # 中文说明：误差 4——温度漂移（用闭环补偿后的"残余"建模）。
    # 假设温控闭环已补偿 95%（compensation_efficiency），只剩 value*5%
    # 的残余温差 residual_c 起作用。残余温差造成：
    #   - 增益扰动：每个通道乘 (1 + 固定模式 * sigma_gain_per_c * 残温差)
    #   - 公共损耗：整体乘 (1 - loss_per_c * 残温差)（温度越高损耗越大）
    residual_c = value * (1.0 - compensation_efficiency)
    if residual_c <= 0.0:
        return output

    axis = -1 if output.ndim >= 2 else 0
    shape = [1] * output.ndim
    shape[axis] = output.shape[axis]
    key = (module_name, tuple(shape), str(output.device), str(output.dtype))
    if key not in pattern_cache:
        pattern_cache[key] = torch.randn(shape, device=output.device, dtype=output.dtype)

    gain = 1.0 + pattern_cache[key] * (sigma_gain_per_c * residual_c)
    common = max(0.0, 1.0 - loss_per_c * residual_c)
    return output * gain * common


# ── Effect 5: Wavelength detuning ────────────────────────────────────────────
def _perturb_wavelength_detuning(output: torch.Tensor, value: float) -> torch.Tensor:
    # 中文说明：误差 5——激光波长失谐。激光波长偏离微环谐振波长越远，
    # 光传输效率越低（传递误差按平方关系增长，封顶 50%）。
    transfer_error = min(0.5, (value / 40.0) ** 2)
    return output * (1.0 - transfer_error)


# ── Main perturbation factory ────────────────────────────────────────────────
def make_perturb_fn(
    effect: str,
    value: float,
    site_count: int = 1,
    compensation_efficiency: float = 0.95,
):
    """按误差名称创建扰动函数，供模型推理时在指定模块输出后调用。

    Parameters match HPAT paper_v1.json defaults.
    """
    # 中文说明：扰动函数工厂。按误差名返回一个 perturb(output, module_name)
    # 函数，方便在模型每个 Linear 的输出后调用。site_count 是注入点位总数
    # （用于按器件数量均摊/平均误差），compensation_efficiency 只对热漂移有效。
    pattern_cache: dict = {}

    def perturb(output: torch.Tensor, module_name: str = "") -> torch.Tensor:
        # 分发到对应的误差函数；未知误差名直接抛异常
        if effect == "gaussian_pd_tia_noise":
            return _perturb_pd_tia(output, value)
        elif effect == "wdm_adjacent_crosstalk":
            return _perturb_wdm_crosstalk(output, value, site_count)
        elif effect == "mrr_variation":
            return _perturb_mrr_variation(output, value, site_count, pattern_cache, module_name)
        elif effect == "thermal_drift":
            return _perturb_thermal_drift(output, value, pattern_cache, module_name, compensation_efficiency)
        elif effect == "wavelength_detuning":
            return _perturb_wavelength_detuning(output, value)
        else:
            raise ValueError(f"Unknown effect: {effect}")

    return perturb


# ── Sweep plan (matching HPAT) ───────────────────────────────────────────────
# 扫描计划：五种误差各配一档严重度取值（与 HPAT 论文保持一致）。
# 每项：(误差名, 扫描变量名, 取值列表[0=无误差, ...])
SWEEP_PLAN = [
    ("gaussian_pd_tia_noise",   "noise_lsb",          [0.0, 0.25, 0.5, 1.0]),
    ("wdm_adjacent_crosstalk",  "crosstalk_alpha",    [0.0, 0.01, 0.03, 0.05]),
    ("mrr_variation",           "sigma_percent",      [0.0, 1.0, 3.0, 5.0]),
    ("thermal_drift",           "delta_c",            [0.0, 2.0, 5.0, 10.0]),
    ("wavelength_detuning",     "delta_lambda_pm",    [0.0, 5.0, 10.0, 20.0]),
]

# 随机类误差（注入随机噪声/偏差，需要多颗种子多次实验取平均）；
# 其余为确定性误差（一次实验即可）
STOCHASTIC_EFFECTS = {"gaussian_pd_tia_noise", "mrr_variation", "thermal_drift"}
