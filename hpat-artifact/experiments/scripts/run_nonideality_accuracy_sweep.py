"""非理想性对精度的影响扫描（run_nonideality_accuracy_sweep.py）。

实验目的：研究 HPAT 光子处理器的各种"非理想性"（噪声、串扰、量化、
插损、波长失谐、工艺偏差、热漂移）如何影响 MobileViT 在固定图片子集
上的分类精度（top-1/top-5 下降、预测变化率等）。支持两种模式：
  1) dataset 模式（默认推荐，但需要 --dataset-root + --subset-file）：
     真实用 torch/timm 跑 MobileViT，在模型模块输出上注入扰动，统计精度变化。
  2) 纯分析模式（无 --dataset-root）：用合成代理公式估算敏感性（浅层检查）。

- 输入：hpat_experiment_config.json、可选 --dataset-root/--subset-file
  （固定标注子集）、--operator-activity-csv（HPAT 映射边界所需算子活动表）。
- 产出（--output-dir 下）：tables/ 下多张 CSV（measured/precision/trials/
  by_severity/safe_region/failure_thresholds）+ raw/ 下原始预测明细与 logits
  + nonideality_accuracy_sweep_manifest.json；开启项目写入权限时同步仓库 tables/。
- 命令：python run_nonideality_accuracy_sweep.py --output-dir <目录>
  [--config <配置>] [--dataset-root <根>] [--subset-file <文件>]
  [--injection-boundary hpat-mapping|reasonable-max|maximal-all-mac|all-linear-smoke]
  [--effects 效果名,逗号分隔] ...
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import random
from collections import defaultdict
from typing import Any

from _common import (
    REPO_ROOT,
    base_manifest,
    ensure_dir,
    load_json,
    project_writes_enabled,
    read_csv,
    relative,
    sha256_file,
    write_csv,
    write_json,
)
from hpat_eval.injection_boundary import all_linear_smoke_boundary, boundary_from_operator_rows, maximal_all_mac_boundary, reasonable_max_boundary
from hpat_eval.mobilevit_loader import select_device, sync_device, variants_from_config
from hpat_eval.nonidealities import NONIDEALITY_FIELDS, nonideality_rows


# 免责声明：本实验只支持"固定子集上的鲁棒性/敏感性"结论，
# 不代表全量 ImageNet 验证、流片验证或实测 HPAT 边缘/移动部署。
DATASET_CLAIM_BOUNDARY = (
    "Fixed-subset robustness/sensitivity only; not full ImageNet validation, "
    "not silicon validation, and not measured HPAT edge/mobile deployment."
)

# 属于"随机性效应"的非理想性：对这些效应会重复多次随机种子取分布，
# 其它效应（确定性效应）只跑一次。
STOCHASTIC_EFFECTS = {"gaussian_pd_tia_noise", "mrr_variation", "thermal_drift"}

# 以下都是 CSV 输出表头（字段名）的定义，供写表时统一使用
MEASURED_NONIDEALITY_FIELDS = NONIDEALITY_FIELDS + [
    "sample_count",
    "label_count",
    "clean_top1",
    "perturbed_top1",
    "top1_delta",
    "clean_top5",
    "perturbed_top5",
    "top5_delta",
    "prediction_change_rate",
    "seed",
    "device",
    "injection_location",
    "injection_boundary",
]

TRIAL_NONIDEALITY_FIELDS = MEASURED_NONIDEALITY_FIELDS + [
    "trial_index",
    "trial_seed",
    "top1_drop",
    "top5_drop",
    "claim_boundary",
]

BY_SEVERITY_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "trial_count",
    "sample_count",
    "label_count",
    "clean_top1_mean",
    "perturbed_top1_mean",
    "top1_delta_mean",
    "top1_drop_mean",
    "top1_drop_p05",
    "top1_drop_p50",
    "top1_drop_p95",
    "clean_top5_mean",
    "perturbed_top5_mean",
    "top5_delta_mean",
    "top5_drop_mean",
    "top5_drop_p05",
    "top5_drop_p50",
    "top5_drop_p95",
    "prediction_change_rate_mean",
    "prediction_change_rate_p05",
    "prediction_change_rate_p50",
    "prediction_change_rate_p95",
    "mean_relative_error_mean",
    "p95_relative_error",
    "seeds",
    "device",
    "metric_kind",
    "evidence_label",
    "claim_boundary",
]

SAFE_REGION_FIELDS = [
    "variant",
    "effect",
    "safe_top1_drop_threshold",
    "safe_top5_drop_threshold",
    "safe_prediction_change_threshold",
    "safe_sweep_variable",
    "safe_sweep_value",
    "safe_direction",
    "safe_status",
    "sample_count",
    "label_count",
    "trial_count",
    "evidence_label",
    "claim_boundary",
]

FAILURE_THRESHOLD_FIELDS = [
    "variant",
    "effect",
    "first_failure_sweep_variable",
    "first_failure_sweep_value",
    "failure_reason",
    "previous_safe_sweep_value",
    "safe_direction",
    "sample_count",
    "label_count",
    "trial_count",
    "evidence_label",
    "claim_boundary",
]

PRECISION_ACCURACY_FIELDS = [
    "variant",
    "adc_dac_bits",
    "sample_count",
    "label_count",
    "clean_top1",
    "perturbed_top1",
    "top1_delta",
    "clean_top5",
    "perturbed_top5",
    "top5_delta",
    "prediction_change_rate",
    "seed",
    "device",
    "injection_location",
    "injection_boundary",
    "evidence_label",
]

PREDICTION_CHANGE_FIELDS = [
    "variant",
    "trial_index",
    "trial_seed",
    "sample_index",
    "image_path",
    "label",
    "effect",
    "sweep_variable",
    "sweep_value",
    "clean_top1",
    "perturbed_top1",
    "changed",
    "clean_correct_top1",
    "perturbed_correct_top1",
]

PREDICTION_TOPK_MARGIN_FIELDS = [
    "variant",
    "trial_index",
    "trial_seed",
    "sample_index",
    "image_path",
    "label",
    "class_name",
    "synset",
    "source_id",
    "effect",
    "sweep_variable",
    "sweep_value",
    "clean_top1",
    "perturbed_top1",
    "changed",
    "clean_top5",
    "perturbed_top5",
    "clean_label_in_top5",
    "perturbed_label_in_top5",
    "clean_correct_top1",
    "perturbed_correct_top1",
    "clean_correct_top5",
    "perturbed_correct_top5",
    "clean_top1_confidence",
    "perturbed_top1_confidence",
    "clean_top1_margin",
    "perturbed_top1_margin",
    "margin_delta",
    "top5_jaccard",
    "injection_boundary",
    "evidence_label",
    "claim_boundary",
]


def _available(name: str) -> bool:
    """判断某个 Python 包是否可被 import。

    参数 name：包名。
    返回：可导入为 True，否则 False。
    """
    return importlib.util.find_spec(name) is not None


def _subset_rows(dataset_root: pathlib.Path, subset_file: pathlib.Path, max_samples: int | None) -> list[dict[str, Any]]:
    """读取"固定图片子集"清单，解析成样本字典列表。

    支持的子集文件格式：
      - CSV：需含 path/image_path/file/filename 列（相对路径），可选 label 等。
      - 纯文本：每行"相对路径 [标签]"，# 开头为注释。
    会做两项校验：样本不能为空；每个样本文件必须真实存在。

    参数：
        dataset_root：数据集根目录。
        subset_file：子集清单文件。
        max_samples：最多取前 N 个样本（None 表示不限制）。
    返回：样本字典列表（含 path/relative_path/label/class_name/synset 等字段）。
    """
    rows: list[dict[str, Any]] = []
    if subset_file.suffix.lower() == ".csv":
        with subset_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                # 兼容不同列名写法（path / image_path / file / filename）
                rel = row.get("path") or row.get("image_path") or row.get("file") or row.get("filename")
                if not rel:
                    raise ValueError("subset CSV must include path, image_path, file, or filename")
                label = row.get("label") or row.get("target") or row.get("class_id") or ""
                sample = dict(row)
                sample.update(
                    {
                        "path": dataset_root / rel,
                        "relative_path": rel,
                        "label": label,
                        "class_name": row.get("class_name", row.get("class", "")),
                        "synset": row.get("synset", ""),
                        "source_id": row.get("source_id", row.get("sha256", "")),
                        "external_data_label": row.get("external_data_label", ""),
                    }
                )
                rows.append(sample)
    else:
        # 纯文本格式：每行 "<相对路径> [标签]"，空行和 # 注释被跳过
        for line in subset_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            rel = parts[0]
            label = parts[1] if len(parts) > 1 else ""
            rows.append(
                {
                    "path": dataset_root / rel,
                    "relative_path": rel,
                    "label": label,
                    "class_name": "",
                    "synset": "",
                    "source_id": "",
                    "external_data_label": "",
                }
            )
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("subset file produced no samples")
    # 校验：每个样本文件必须真实存在，避免后续加载时报晦涩错误
    for index, row in enumerate(rows):
        if not pathlib.Path(row["path"]).exists():
            raise ValueError(f"subset sample {index + 1} does not exist: {row['path']}")
    return rows


def _sweep_plan(config: dict[str, Any]) -> list[tuple[str, str, list[Any]]]:
    """从配置生成"扫描计划"：要注入哪些非理想性、扫描哪个变量、扫哪些值。

    返回：[(效果名, 扫描变量名, 取值列表), ...] 的列表。
    例：("gaussian_pd_tia_noise", "noise_lsb", [0.0, 0.25, 0.5, 1.0])
    """
    cfg = config.get("nonideality", {})
    return [
        ("gaussian_pd_tia_noise", "noise_lsb", cfg.get("noise_lsb", [0.0, 0.25, 0.5, 1.0])),
        ("wdm_adjacent_crosstalk", "crosstalk_alpha", cfg.get("crosstalk_alpha", [0.0, 0.01, 0.03, 0.05])),
        ("uniform_converter_quantization", "quant_bits", cfg.get("quant_bits", [4, 6, 8, 10])),
        ("insertion_loss", "path_db", cfg.get("insertion_loss_db", [0.0, 1.0, 2.0, 3.0])),
        ("wavelength_detuning", "delta_lambda_pm", cfg.get("detuning_pm", [0.0, 5.0, 10.0, 20.0])),
        ("mrr_variation", "sigma_percent", cfg.get("mrr_variation_percent", [0.0, 1.0, 3.0, 5.0])),
        ("thermal_drift", "delta_c", cfg.get("thermal_drift_c", [0.0, 2.0, 5.0, 10.0])),
    ]


def _selected_sweep_plan(config: dict[str, Any], selected_effects: set[str] | None) -> list[tuple[str, str, list[Any]]]:
    """按用户选择的效果名过滤扫描计划。

    参数：
        config：实验配置。
        selected_effects：要保留的效果名集合；None 或空表示全部保留。
    返回：过滤后的扫描计划；遇到未知效果名会报错。
    """
    plan = _sweep_plan(config)
    if not selected_effects:
        return plan
    known = {effect for effect, _variable, _values in plan}
    unknown = selected_effects - known
    if unknown:
        raise ValueError(f"Unknown non-ideality effects requested: {', '.join(sorted(unknown))}")
    return [entry for entry in plan if entry[0] in selected_effects]


def _sanitize_key(value: Any) -> str:
    """把任意值转成"适合做文件名/键"的安全字符串。

    把 -、.、/、空格替换成 m、p、_、_，避免路径解析歧义。
    例：_sanitize_key("4.0") == "4p0"
    """
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_").replace(" ", "_")


def _stable_offset(*parts: Any) -> int:
    """根据若干字符串算一个稳定的小偏移量（0~99999）。

    用途：给不同 (模型, 效果) 组合分配不同的随机扰动方向偏移。
    同一个输入永远得到同一个偏移，保证可复现。
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100000


def _trial_torch_seed(trial_seed: int, variant: str, effect: str) -> int:
    """Return one seed per trial/effect so severity points share a perturbation direction.

    为每次试验/每个效果生成独立种子，使同一效果的不同严重度取值共享
    相同的随机扰动方向，曲线才不会因为种子不同而抖动。
    """
    return int(trial_seed) + _stable_offset(variant, effect)


def _configure_torch_determinism(torch: Any) -> None:
    """Require deterministic kernels instead of silently accepting a nondeterministic path.

    强制 torch 使用确定性算子，保证同样输入产出同样结果（可复现）。
    若某算子没有确定版本会直接报错，而不是静默接受不确定结果。
    """
    torch.use_deterministic_algorithms(True)


def _seed_torch_rng(torch: Any, seed: int, device: Any) -> None:
    """Synchronize and seed both the global and selected-device RNG streams.

    同步所选设备并给 torch 全局随机源与 MPS（Apple GPU）随机源分别播种子，
    确保模型初始化与扰动生成都是可复现的。
    """
    sync_device(torch, device)
    torch.manual_seed(int(seed))
    if str(device).startswith("mps"):
        torch.mps.manual_seed(int(seed))


def _jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    """把配置里的元组转成列表，使其可直接 JSON 序列化。

    参数 config：任意字典。
    返回：可 JSON 序列化的字典副本。
    """
    out: dict[str, Any] = {}
    for key, value in config.items():
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


def _prepare_image(path: pathlib.Path, resolution: int, torch: Any, transform: Any | None = None):
    """把单张图片加载成模型输入的张量。

    参数：
        path：图片文件路径。
        resolution：正方形分辨率（如 256/384）。
        torch：torch 模块（延迟注入，便于无 torch 环境也能 import 本脚本）。
        transform：可选的预处理函数；None 时用内置的 resize+归一化。
    返回：形状为 (1, 3, resolution, resolution) 的 float32 张量（已归一化）。
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if transform is not None:
        return transform(image)
    # 兜底预处理：缩放 → 像素值转张量 → 按 ImageNet 均值/方差归一化
    image = image.resize((resolution, resolution))
    data = list(image.getdata())
    tensor = torch.tensor(data, dtype=torch.float32).view(resolution, resolution, 3).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _labels(samples: list[dict[str, Any]]) -> list[int | None]:
    """从样本列表中提取标签；无标签的样本返回 None。

    参数 samples：样本字典列表。
    返回：与样本等长的标签列表（int 或 None）。
    """
    labels: list[int | None] = []
    for row in samples:
        value = str(row.get("label", "")).strip()
        labels.append(int(value) if value else None)
    return labels


def _topk(logits: Any, k: int) -> list[list[int]]:
    """取每个样本 logits 的前 k 个预测类别索引。

    参数：
        logits：形状 (N, C) 的模型输出。
        k：要取多少个。
    返回：[[索引...], ...]，每行是一个样本的 top-k 类别列表。
    """
    values = logits.topk(k=min(k, logits.shape[-1]), dim=-1).indices.detach().cpu().tolist()
    return [[int(v) for v in row] for row in values]


def _topk_detail(logits: Any, torch: Any, k: int = 5) -> tuple[list[list[int]], list[list[float]], list[float], list[float]]:
    """取 top-k 的详细信息：索引、概率值、top-1 置信度、top-1 与 top-2 的间隔。

    返回：(类别索引, 概率值, top1置信度列表, top1间隔列表)。
    间隔（margin）越大说明预测越稳，扰动后越不容易翻转。
    """
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(k=min(k, logits.shape[-1]), dim=-1)
    indices = [[int(v) for v in row] for row in top.indices.detach().cpu().tolist()]
    values = [[float(v) for v in row] for row in top.values.detach().cpu().tolist()]
    top1_conf = [row[0] if row else 0.0 for row in values]
    margins = [(row[0] - row[1]) if len(row) > 1 else (row[0] if row else 0.0) for row in values]
    return indices, values, top1_conf, margins


def _accuracy(topk: list[list[int]], labels: list[int | None]) -> tuple[float, int]:
    """计算 top-k 命中率（只统计有标签的样本）。

    参数：
        topk：每样本的预测类别列表。
        labels：每样本的真实标签（None 表示无标签样本）。
    返回：(正确率百分比, 有标签样本数)。
    """
    labelled = [(preds, label) for preds, label in zip(topk, labels) if label is not None]
    if not labelled:
        return 0.0, 0
    correct = sum(1 for preds, label in labelled if int(label) in preds)
    return 100.0 * correct / len(labelled), len(labelled)


def _collapse_nested_module_names(names: set[str]) -> set[str]:
    """Keep one outermost injection site when both a module and its descendants are selected.

    若同时选中了某个模块及其子模块，只保留最外层注入点，
    避免对同一输出重复注入扰动。
    """
    kept: list[str] = []
    for name in sorted(names, key=lambda item: (item.count("."), len(item), item)):
        if any(name.startswith(parent + ".") for parent in kept):
            continue
        kept.append(name)
    return set(kept)


def _feature_axis(output: Any) -> int:
    # Conv2d uses NCHW; Linear/attention outputs keep features in the last dimension.
    # （中文说明）特征轴位置：卷积输出为 NCHW 时特征在轴 1；
    # Linear/注意力输出特征在最后一维（轴 -1）。
    return 1 if getattr(output, "ndim", 0) == 4 else -1


def _make_perturb_fn(
    effect: str,
    value: Any,
    torch: Any,
    rng: random.Random,
    profile_config: dict[str, Any] | None = None,
):
    """构造一个"扰动函数"perturb(output, module_name)，模拟指定非理想性。

    该函数会被注册成模型模块的 forward hook：每层模块前向计算出 output 时，
    自动对 output 施加扰动，模拟光子硬件中的信号劣化。

    参数：
        effect：非理想性效果名（如 gaussian_pd_tia_noise、mrr_variation）。
        value：该效果的严重度扫描值。
        torch：torch 模块。
        rng：随机源（本函数暂未直接使用，保留参数以统一接口）。
        profile_config：注入画像配置（含注入点数、热补偿效率等假设）。
    返回：perturb 函数（可被 register_forward_hook 使用）。
    """
    _ = rng
    numeric = float(value)
    cfg = profile_config or {}
    profile = str(cfg.get("injection_profile", "calibrated_proxy_v2"))
    site_count = max(int(cfg.get("_injection_site_count", 1)), 1)
    # 缓存"每模块/每形状"的随机模式，保证同一注入点在多次前向中扰动方向不变
    pattern_cache: dict[tuple[str, tuple[int, ...], str, str], Any] = {}
    def channel_pattern(output: Any, module_name: str) -> Any:
        """生成一个沿特征轴的随机模式张量（用于 mrr_variation 等静态失配）。

        按模块名+形状+设备+类型缓存，同一站点重复调用返回同一模式，
        模拟"工艺偏差是固定的"这一物理特性。
        """
        axis = _feature_axis(output)
        axis = axis if axis >= 0 else output.ndim + axis
        shape = [1] * output.ndim
        shape[axis] = output.shape[axis]
        key = (module_name, tuple(shape), str(output.device), str(output.dtype))
        if key not in pattern_cache:
            pattern_cache[key] = torch.randn(shape, device=output.device, dtype=output.dtype)
        return pattern_cache[key]

    def perturb(output: Any, module_name: str = ""):
        # 按效果种类对模块输出施加不同扰动：
        if effect == "gaussian_pd_tia_noise":
            # LSB-referenced noise follows each tensor's RMS rather than a hard unit floor.
            # （中文说明）加性高斯噪声：噪声幅度以该张量 RMS 的 LSB 比例缩放，
            # 而不是固定某个绝对幅度，更接近真实读出链路（PD/TIA）噪声行为。
            rms = torch.sqrt(torch.clamp(output.detach().float().pow(2).mean(), min=1e-12)).to(output.dtype)
            scale = rms * numeric / 255.0
            return output + torch.randn_like(output) * scale
        if effect == "wdm_adjacent_crosstalk":
            # Symmetric nearest-neighbour leakage on the feature/channel axis, with DC gain preserved.
            # The sweep value is a mapped-subgraph-equivalent budget, distributed across sites to
            # avoid applying the same end-to-end crosstalk allowance at every optical conversion.
            # （中文说明）相邻波长通道串扰：沿特征轴做"左右邻居加权泄漏"，
            # 保持直流增益不变；总串扰预算按注入点数量分摊，避免重复叠加。
            axis = _feature_axis(output)
            if output.shape[axis] <= 1:
                return output
            alpha = min(max(numeric / site_count, 0.0), 0.49)
            return (
                (1.0 - 2.0 * alpha) * output
                + alpha * torch.roll(output, shifts=1, dims=axis)
                + alpha * torch.roll(output, shifts=-1, dims=axis)
            )
        if effect == "uniform_converter_quantization":
            # ADC/DAC 均匀量化：裁剪到 [-1,1] 后按 bits 位取整量化
            bits = max(int(numeric), 1)
            levels = (1 << bits) - 1
            clipped = torch.clamp(output, -1.0, 1.0)
            return (torch.round((clipped + 1.0) * levels / 2.0) * 2.0 / levels) - 1.0
        if effect == "insertion_loss":
            # 插入损耗：光功率按 dB 数线性衰减（-numeric/20 换算成幅度比）
            return output * (10.0 ** (-numeric / 20.0))
        if effect == "wavelength_detuning":
            # 波长失谐：按失谐量(pm)的平方折算传递误差，封顶 50%
            transfer_error = min(0.5, (numeric / 40.0) ** 2)
            return output * (1.0 - transfer_error)
        if effect == "mrr_variation":
            # Process variation is static per mapped site/channel, not fresh activation noise per batch.
            # Independent site variation is apportioned by sqrt(N), so the x-axis remains an
            # equivalent mapped-subgraph RMS budget rather than N repeated copies of that budget.
            # （中文说明）工艺偏差（微环尺寸差异）：是静态的站点间增益失配，
            # 不是每批随机的激活噪声。独立站点偏差按 sqrt(N) 分摊，
            # 使横轴仍是"等效子图 RMS 预算"而不是把同一预算重复 N 次。
            sigma = numeric / (100.0 * (site_count ** 0.5))
            return output * (1.0 + channel_pattern(output, module_name) * sigma)
        if effect == "thermal_drift":
            # Closed-loop proxy: ambient change is reduced to a residual error, then expressed as
            # static channel gain mismatch plus a small common-mode loss. Assumptions remain explicit
            # and uncalibrated until a layout/controller transfer matrix is available.
            # （中文说明）热漂移的闭环代理：环境温度变化被闭环补偿掉大部分，
            # 只留残差误差，再把残差表达成"静态通道增益失配 + 小量共模损耗"。
            # 在获得版图/控制器传递矩阵前，这些假设都是显式且未标定的。
            compensation = min(max(float(cfg.get("thermal_compensation_efficiency", 0.98)), 0.0), 1.0)
            residual_c = numeric * (1.0 - compensation)
            sigma_per_c = max(float(cfg.get("thermal_residual_gain_sigma_per_c", 0.02)), 0.0)
            loss_per_c = max(float(cfg.get("thermal_common_loss_per_residual_c", 0.002)), 0.0)
            gain = 1.0 + channel_pattern(output, module_name) * (sigma_per_c * residual_c)
            common = max(0.0, 1.0 - loss_per_c * residual_c)
            return output * gain * common
        return output

    # 把注入画像名挂到函数属性上，便于后续检查用了哪种扰动模型
    setattr(perturb, "_injection_profile", profile)

    return perturb


def _forward_logits(model: Any, inputs: Any, torch: Any, device: Any, perturb_fn: Any | None = None) -> Any:
    """对一批输入做前向推理，可选地注入扰动，返回 CPU 上的 logits。

    参数：
        model：torch 模型。
        inputs：输入张量（已在目标设备上）。
        torch：torch 模块。
        device：目标设备。
        perturb_fn：扰动函数；None 表示不注入（清洁推理）。
    返回：形状 (N, C) 的 logits（已移到 CPU）。
    """
    hooks = []
    if perturb_fn is not None:
        # 在指定模块输出上注册 forward hook：allowed 集合为空时默认只注入 Linear
        allowed = getattr(perturb_fn, "_allowed_module_names", None)
        for name, module in model.named_modules():
            if (allowed is not None and name in allowed) or (allowed is None and isinstance(module, torch.nn.Linear)):
                hooks.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output, module_name=name: perturb_fn(output, module_name)
                    )
                )
    with torch.no_grad():
        logits = model(inputs)
        sync_device(torch, device)
    # 前向结束立即移除 hook，避免污染后续推理
    for hook in hooks:
        hook.remove()
    return logits.detach().cpu()


def _prepare_input_batches(
    samples: list[dict[str, Any]],
    resolution: int,
    torch: Any,
    transform: Any,
    batch_size: int,
) -> list[tuple[int, Any]]:
    """把样本按批切分并预处理成输入张量。

    参数：
        samples：样本字典列表。
        resolution：输入分辨率。
        torch：torch 模块。
        transform：预处理函数。
        batch_size：每批样本数。
    返回：[(批起始索引, 输入张量), ...] 列表。
    """
    batches: list[tuple[int, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch_rows = samples[start : start + batch_size]
        batch = torch.stack(
            [_prepare_image(pathlib.Path(row["path"]), resolution, torch, transform=transform) for row in batch_rows]
        )
        batches.append((start, batch))
    return batches


def _parse_repeat_seeds(value: str | list[int] | None, seed: int) -> list[int]:
    """解析"重复试验种子"参数（逗号分隔字符串或列表）。

    参数：
        value：None/空 → 只用一个种子；字符串 → 按逗号拆成多个；列表 → 直接用。
        seed：默认种子。
    返回：种子整数列表（至少一个）。
    """
    if value is None or value == "":
        return [seed]
    if isinstance(value, list):
        seeds = [int(v) for v in value]
    else:
        seeds = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    return seeds or [seed]


def _trial_seeds(effect: str, repeat_seeds: list[int], seed: int) -> list[int]:
    """决定某个效果要用哪些种子跑试验。

    随机性效应（噪声/工艺偏差/热漂移）用用户给的重复种子列表取分布；
    确定性效应只用单个种子（跑多次没有意义）。
    """
    if effect in STOCHASTIC_EFFECTS:
        return repeat_seeds
    return [seed]


def _float(value: Any, default: float = 0.0) -> float:
    """把任意值转成浮点数；空值/None 返回默认值。"""
    if value in ("", None):
        return default
    return float(value)


def _fmt(value: float, digits: int = 4) -> str:
    """把浮点数格式化成固定小数位的字符串。"""
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float:
    """求均值（空列表返回 0.0）。"""
    return sum(values) / max(len(values), 1)


def _percentile(values: list[float], percentile: float) -> float:
    """计算百分位数（线性插值，空列表返回 0.0）。

    参数：
        values：数值列表。
        percentile：百分位（0~100）。
    返回：对应百分位的值。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_relpath(sample_path: pathlib.Path, dataset_root: pathlib.Path) -> str:
    """把样本绝对路径转成相对数据集根的路径；不在根下则原样返回。"""
    try:
        return str(sample_path.relative_to(dataset_root))
    except ValueError:
        return str(sample_path)


def _model_state_sha256(model: Any) -> str:
    """Hash model parameters without depending on torch serialization details.

    直接对模型参数张量的原始字节计算 SHA-256，不依赖 torch 的序列化格式，
    用于记录"用哪个权重的模型跑的实验"。
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _topk_string(values: list[int]) -> str:
    """把 top-k 索引列表拼成分号分隔的字符串，便于写进 CSV。"""
    return ";".join(str(int(value)) for value in values)


def _jaccard(a: list[int], b: list[int]) -> float:
    """计算两个 top-k 集合的 Jaccard 相似度（交集/并集）。

    用于衡量扰动前后预测类别重叠程度；空并集按 1.0（完全一致）处理。
    """
    left = set(int(v) for v in a)
    right = set(int(v) for v in b)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _measured_row(
    *,
    variant: str,
    effect: str,
    variable: str,
    value: Any,
    trials: int,
    mean_relative_error: float | None,
    p95_relative_error: float | None,
    metric_kind: str,
    evidence_label: str,
    sample_count: int,
    label_count: int,
    clean_top1: float | None,
    perturbed_top1: float | None,
    clean_top5: float | None,
    perturbed_top5: float | None,
    prediction_change_rate: float,
    seed: str | int,
    device: str,
    injection_location: str,
    injection_boundary: str,
) -> dict[str, Any]:
    """组装一行"测量结果"记录（对应 CSV 中一行）。

    各参数即 CSV 字段值；无标签样本相关字段（top1/top5 等）填空字符串。
    返回：可直接写入 CSV 的字典。
    """
    return {
        "variant": variant,
        "effect": effect,
        "sweep_variable": variable,
        "sweep_value": value,
        "trials": trials,
        "mean_relative_error": f"{mean_relative_error:.8f}" if mean_relative_error is not None else "",
        "p95_relative_error": f"{p95_relative_error:.8f}" if p95_relative_error is not None else "",
        # 以下三个代理字段在 dataset 模式下不填，保留给纯分析模式
        "proxy_score_percent": "",
        "laser_energy_multiplier": "",
        "retuning_overhead_percent": "",
        "metric_kind": metric_kind,
        "evidence_label": evidence_label,
        "sample_count": sample_count,
        "label_count": label_count,
        "clean_top1": _fmt(clean_top1) if clean_top1 is not None and label_count else "",
        "perturbed_top1": _fmt(perturbed_top1) if perturbed_top1 is not None and label_count else "",
        "top1_delta": _fmt((perturbed_top1 or 0.0) - (clean_top1 or 0.0)) if clean_top1 is not None and perturbed_top1 is not None and label_count else "",
        "clean_top5": _fmt(clean_top5) if clean_top5 is not None and label_count else "",
        "perturbed_top5": _fmt(perturbed_top5) if perturbed_top5 is not None and label_count else "",
        "top5_delta": _fmt((perturbed_top5 or 0.0) - (clean_top5 or 0.0)) if clean_top5 is not None and perturbed_top5 is not None and label_count else "",
        "prediction_change_rate": _fmt(prediction_change_rate),
        "seed": seed,
        "device": device,
        "injection_location": injection_location,
        "injection_boundary": injection_boundary,
    }


def derive_nonideality_summary_tables(
    trial_rows: list[dict[str, Any]],
    safe_top1_drop_threshold: float = 5.0,
    safe_top5_drop_threshold: float = 5.0,
    safe_prediction_change_threshold: float = 10.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """由逐次试验明细行汇总出五张结果表。

    按 (模型, 效果, 扫描变量, 取值) 分组，对每组内多次试验做统计：
    - measured_rows：每组的"代表性测量行"（均值精度、变化率等）。
    - precision_rows：只含量化效果（uniform_converter_quantization）的精度行。
    - by_severity_rows：含分位数（p05/p50/p95）的按严重度汇总。
    - safe_rows / failure_rows：安全区间与首个失效阈值（见 _derive_safe_regions）。

    参数：
        trial_rows：逐次试验明细行列表。
        三个阈值：判定"安全"的 top-1/top-5 下降与预测变化率上限。
    返回：(measured_rows, precision_rows, by_severity_rows, safe_rows, failure_rows)。
    """
    # 按 (variant, effect, variable, value) 分组，把同一严重度的多次试验聚在一起
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["variant"]), str(row["effect"]), str(row["sweep_variable"]), str(row["sweep_value"]))].append(row)

    by_severity: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    for (variant, effect, variable, value), rows in sorted(grouped.items()):
        # 收集该组各列数值（过滤空值），用于均值/分位数统计
        clean_top1 = [_float(row.get("clean_top1")) for row in rows if row.get("clean_top1") not in ("", None)]
        pert_top1 = [_float(row.get("perturbed_top1")) for row in rows if row.get("perturbed_top1") not in ("", None)]
        top1_delta = [_float(row.get("top1_delta")) for row in rows if row.get("top1_delta") not in ("", None)]
        top1_drop = [_float(row.get("top1_drop")) for row in rows if row.get("top1_drop") not in ("", None)]
        clean_top5 = [_float(row.get("clean_top5")) for row in rows if row.get("clean_top5") not in ("", None)]
        pert_top5 = [_float(row.get("perturbed_top5")) for row in rows if row.get("perturbed_top5") not in ("", None)]
        top5_delta = [_float(row.get("top5_delta")) for row in rows if row.get("top5_delta") not in ("", None)]
        top5_drop = [_float(row.get("top5_drop")) for row in rows if row.get("top5_drop") not in ("", None)]
        change = [_float(row.get("prediction_change_rate")) for row in rows]
        relative = [
            _float(row.get("mean_relative_error"))
            for row in rows
            if row.get("mean_relative_error") not in ("", None)
        ]
        p95_relative = [
            _float(row.get("p95_relative_error"))
            for row in rows
            if row.get("p95_relative_error") not in ("", None)
        ]
        seeds = sorted({str(row.get("trial_seed") or row.get("seed")) for row in rows if row.get("trial_seed") or row.get("seed")})
        sample_count = int(_float(rows[0].get("sample_count"), 0.0))
        label_count = int(_float(rows[0].get("label_count"), 0.0))
        evidence_label = str(rows[0].get("evidence_label", ""))
        # 有多次试验时，在证据标签里注明是随机种子包络还是重复试验
        if len(rows) > 1:
            evidence_label += "; repeated stochastic seed envelope" if effect in STOCHASTIC_EFFECTS else "; repeated trials"
        severity_row = {
            "variant": variant,
            "effect": effect,
            "sweep_variable": variable,
            "sweep_value": value,
            "trial_count": len(rows),
            "sample_count": sample_count,
            "label_count": label_count,
            "clean_top1_mean": _fmt(_mean(clean_top1)) if clean_top1 else "",
            "perturbed_top1_mean": _fmt(_mean(pert_top1)) if pert_top1 else "",
            "top1_delta_mean": _fmt(_mean(top1_delta)) if top1_delta else "",
            "top1_drop_mean": _fmt(_mean(top1_drop)) if top1_drop else "",
            "top1_drop_p05": _fmt(_percentile(top1_drop, 5)) if top1_drop else "",
            "top1_drop_p50": _fmt(_percentile(top1_drop, 50)) if top1_drop else "",
            "top1_drop_p95": _fmt(_percentile(top1_drop, 95)) if top1_drop else "",
            "clean_top5_mean": _fmt(_mean(clean_top5)) if clean_top5 else "",
            "perturbed_top5_mean": _fmt(_mean(pert_top5)) if pert_top5 else "",
            "top5_delta_mean": _fmt(_mean(top5_delta)) if top5_delta else "",
            "top5_drop_mean": _fmt(_mean(top5_drop)) if top5_drop else "",
            "top5_drop_p05": _fmt(_percentile(top5_drop, 5)) if top5_drop else "",
            "top5_drop_p50": _fmt(_percentile(top5_drop, 50)) if top5_drop else "",
            "top5_drop_p95": _fmt(_percentile(top5_drop, 95)) if top5_drop else "",
            "prediction_change_rate_mean": _fmt(_mean(change)),
            "prediction_change_rate_p05": _fmt(_percentile(change, 5)),
            "prediction_change_rate_p50": _fmt(_percentile(change, 50)),
            "prediction_change_rate_p95": _fmt(_percentile(change, 95)),
            "mean_relative_error_mean": f"{_mean(relative):.8f}" if relative else "",
            "p95_relative_error": f"{_percentile(p95_relative, 95):.8f}" if p95_relative else "",
            "seeds": ",".join(seeds),
            "device": str(rows[0].get("device", "")),
            "metric_kind": str(rows[0].get("metric_kind", "")),
            "evidence_label": evidence_label,
            "claim_boundary": DATASET_CLAIM_BOUNDARY,
        }
        by_severity.append(severity_row)
        # 用均值生成"代表性测量行"
        measured = _measured_row(
            variant=variant,
            effect=effect,
            variable=variable,
            value=value,
            trials=sample_count,
            mean_relative_error=(
                _float(severity_row["mean_relative_error_mean"])
                if severity_row["mean_relative_error_mean"]
                else None
            ),
            p95_relative_error=(
                _float(severity_row["p95_relative_error"])
                if severity_row["p95_relative_error"]
                else None
            ),
            metric_kind=severity_row["metric_kind"],
            evidence_label=evidence_label,
            sample_count=sample_count,
            label_count=label_count,
            clean_top1=_float(severity_row["clean_top1_mean"]) if severity_row["clean_top1_mean"] else None,
            perturbed_top1=_float(severity_row["perturbed_top1_mean"]) if severity_row["perturbed_top1_mean"] else None,
            clean_top5=_float(severity_row["clean_top5_mean"]) if severity_row["clean_top5_mean"] else None,
            perturbed_top5=_float(severity_row["perturbed_top5_mean"]) if severity_row["perturbed_top5_mean"] else None,
            prediction_change_rate=_float(severity_row["prediction_change_rate_mean"]),
            seed=severity_row["seeds"],
            device=severity_row["device"],
            injection_location=str(rows[0].get("injection_location", "")),
            injection_boundary=str(rows[0].get("injection_boundary", "")),
        )
        measured_rows.append(measured)
        # 单独抽出"量化"效果：ADC/DAC 位宽是论文最关注的精度图之一
        if effect == "uniform_converter_quantization":
            precision_rows.append(
                {
                    "variant": variant,
                    "adc_dac_bits": value,
                    "sample_count": sample_count,
                    "label_count": label_count,
                    "clean_top1": measured["clean_top1"],
                    "perturbed_top1": measured["perturbed_top1"],
                    "top1_delta": measured["top1_delta"],
                    "clean_top5": measured["clean_top5"],
                    "perturbed_top5": measured["perturbed_top5"],
                    "top5_delta": measured["top5_delta"],
                    "prediction_change_rate": measured["prediction_change_rate"],
                    "seed": measured["seed"],
                    "device": measured["device"],
                    "injection_location": measured["injection_location"],
                    "injection_boundary": measured["injection_boundary"],
                    "evidence_label": measured["evidence_label"],
                }
            )

    # 安全区间与失效阈值：基于按严重度汇总的行进一步推导
    safe_rows, failure_rows = _derive_safe_regions(
        by_severity,
        safe_top1_drop_threshold=safe_top1_drop_threshold,
        safe_top5_drop_threshold=safe_top5_drop_threshold,
        safe_prediction_change_threshold=safe_prediction_change_threshold,
    )
    return measured_rows, precision_rows, by_severity, safe_rows, failure_rows


def _derive_safe_regions(
    by_severity: list[dict[str, Any]],
    safe_top1_drop_threshold: float,
    safe_top5_drop_threshold: float,
    safe_prediction_change_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从按严重度汇总中推出每个效果的"安全区间"与"首个失效阈值"。

    思路：对每个 (模型, 效果)，把扫描值按"严重度递增"排序（量化效果例外，
    位宽越大越安全，故按位宽递减排），从最安全端开始检查；一旦某点超过任一
    阈值，就记下"首个失效点"及"它之前最后一个安全点"。

    参数：
        by_severity：按严重度汇总的行列表。
        三个阈值：top-1/top-5 下降、预测变化率的"安全上限"。
    返回：(safe_rows, failure_rows)，分别记录安全区间和失效阈值。
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_severity:
        grouped[(str(row["variant"]), str(row["effect"]))].append(row)
    safe_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for (variant, effect), rows in sorted(grouped.items()):
        # 量化效果：位宽越高越安全，所以按取值从大到小检查；
        # 其它效果：扫描值越大越严重，按从小到大检查。
        higher_value_safer = effect == "uniform_converter_quantization"
        sorted_rows = sorted(rows, key=lambda row: _float(row["sweep_value"]), reverse=higher_value_safer)
        direction = "higher_value_safer_minimum_safe_value_reported" if higher_value_safer else "higher_value_more_severe_maximum_safe_value_reported"
        safe_candidates: list[dict[str, Any]] = []
        first_failure: dict[str, Any] | None = None
        previous_safe_value = ""
        for row in sorted_rows:
            # 用"均值"字段判定：超过任一阈值即视为该点不安全
            top1_drop = _float(row.get("top1_drop_mean"))
            top5_drop = _float(row.get("top5_drop_mean"))
            change = _float(row.get("prediction_change_rate_mean"))
            failures: list[str] = []
            if top1_drop > safe_top1_drop_threshold:
                failures.append("top1_drop")
            if top5_drop > safe_top5_drop_threshold:
                failures.append("top5_drop")
            if change > safe_prediction_change_threshold:
                failures.append("prediction_change_rate")
            if failures:
                # 记录第一个失效点（及其前一个安全值）
                if first_failure is None:
                    first_failure = {**row, "failure_reason": ";".join(failures), "previous_safe_sweep_value": previous_safe_value}
            else:
                safe_candidates.append(row)
                previous_safe_value = str(row["sweep_value"])
        # 安全值取最接近失效点的那一个（即扫描范围内允许的最大安全值）
        if safe_candidates:
            safe_value = safe_candidates[-1]["sweep_value"]
            safe_status = "safe_region_observed_within_sweep"
        else:
            safe_value = ""
            safe_status = "no_sweep_value_safe_within_threshold"
        if first_failure is None:
            failure_reason = ""
            first_failure_value = ""
            previous_value = previous_safe_value
        else:
            failure_reason = str(first_failure["failure_reason"])
            first_failure_value = str(first_failure["sweep_value"])
            previous_value = str(first_failure.get("previous_safe_sweep_value", ""))
        template = rows[0]
        safe_rows.append(
            {
                "variant": variant,
                "effect": effect,
                "safe_top1_drop_threshold": safe_top1_drop_threshold,
                "safe_top5_drop_threshold": safe_top5_drop_threshold,
                "safe_prediction_change_threshold": safe_prediction_change_threshold,
                "safe_sweep_variable": template["sweep_variable"],
                "safe_sweep_value": safe_value,
                "safe_direction": direction,
                "safe_status": safe_status if first_failure is not None else "all_sweep_values_safe_within_threshold",
                "sample_count": template["sample_count"],
                "label_count": template["label_count"],
                "trial_count": max(int(_float(row.get("trial_count"), 1)) for row in rows),
                "evidence_label": template["evidence_label"],
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
        failure_rows.append(
            {
                "variant": variant,
                "effect": effect,
                "first_failure_sweep_variable": template["sweep_variable"],
                "first_failure_sweep_value": first_failure_value,
                "failure_reason": failure_reason,
                "previous_safe_sweep_value": previous_value,
                "safe_direction": direction,
                "sample_count": template["sample_count"],
                "label_count": template["label_count"],
                "trial_count": max(int(_float(row.get("trial_count"), 1)) for row in rows),
                "evidence_label": template["evidence_label"],
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return safe_rows, failure_rows


def _run_dataset_mode(
    output_dir: pathlib.Path,
    config: dict[str, Any],
    dataset_root: pathlib.Path,
    subset_file: pathlib.Path,
    max_samples: int | None,
    seed: int,
    device_request: str,
    pretrained: bool,
    model_variant: str,
    operator_activity_csv: pathlib.Path | None,
    injection_boundary: str,
    min_boundary_coverage: float,
    batch_size: int,
    save_logits: str,
    save_prediction_detail: str,
    repeat_seeds: list[int],
    safe_top1_drop_threshold: float,
    safe_top5_drop_threshold: float,
    safe_prediction_change_threshold: float,
    selected_effects: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """真实跑 MobileViT 的数据集模式：注入非理想性并统计精度变化（核心函数）。

    流程：
    1) 检查依赖并选定设备，固定随机种子；
    2) 加载固定图片子集，逐模型变体加载 timm 预训练模型；
    3) 先做"清洁推理"得到基准 logits 与 top-1/top-5 精度；
    4) 对每个效果×严重度×试验种子，在 HPAT 映射层输出上注入扰动再推理，
       逐样本记录预测变化明细（写入 CSV），并计算精度/变化率等指标；
    5) 汇总出五张结果表 + 环境元信息。

    参数：详见各参数名；其中 injection_boundary 决定在哪些层注入
    （hpat-mapping 用 HPAT 真实映射层，all-linear-smoke 只在全部 Linear 层）。
    返回：(measured_rows, precision_rows, trial_rows, by_severity_rows,
          safe_rows, failure_rows, meta)。
    """
    # 先检查运行所需依赖是否齐全，缺失则直接报错
    missing = [name for name in ["torch", "timm", "PIL", "numpy"] if not _available(name)]
    if missing:
        raise RuntimeError(f"Missing Python packages for dataset mode: {', '.join(missing)}")

    import numpy as np  # type: ignore
    import timm  # type: ignore
    import torch  # type: ignore
    from timm.data import create_transform, resolve_data_config  # type: ignore

    rng = random.Random(seed)
    # 选择设备（cpu/mps/cuda），并启用确定性算子 + 统一播种
    device, backend, device_reason = select_device(torch, device_request)
    _configure_torch_determinism(torch)
    _seed_torch_rng(torch, seed, device)
    samples = _subset_rows(dataset_root, subset_file, max_samples)
    labels = _labels(samples)
    raw_dir = ensure_dir(output_dir / "raw")
    prediction_csv = raw_dir / "nonideality_prediction_changes.csv"
    topk_margin_csv = raw_dir / "nonideality_prediction_topk_margin.csv"
    trial_rows: list[dict[str, Any]] = []
    clean_npz: dict[str, Any] = {}
    perturbed_npz: dict[str, Any] = {}
    # 只保留用户选择的模型变体（"all" 表示全部）
    variants = [v for v in variants_from_config(config) if model_variant == "all" or v["variant"] == model_variant]
    if not variants:
        raise ValueError(f"Unknown model variant requested: {model_variant}")

    boundary_records: list[dict[str, Any]] = []
    preprocess_records: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))

    # 若要求 topk-margin 明细，则同时打开第二个明细文件；否则用空上下文占位
    topk_margin_context = (
        topk_margin_csv.open("w", encoding="utf-8", newline="")
        if save_prediction_detail == "topk-margin"
        else contextlib.nullcontext(None)
    )
    with prediction_csv.open("w", encoding="utf-8", newline="") as f_prediction, topk_margin_context as f_topk_margin:
        prediction_writer = csv.DictWriter(
            f_prediction, fieldnames=PREDICTION_CHANGE_FIELDS, lineterminator="\n"
        )
        prediction_writer.writeheader()
        topk_margin_writer = None
        if f_topk_margin is not None:
            topk_margin_writer = csv.DictWriter(
                f_topk_margin, fieldnames=PREDICTION_TOPK_MARGIN_FIELDS, lineterminator="\n"
            )
            topk_margin_writer.writeheader()
        # 外层循环：每个模型变体
        for variant in variants:
            resolution = int(variant["input_resolution"])
            # 从 timm 加载预训练模型，记录其权重哈希（可复现性）
            model = timm.create_model(variant["timm_model"], pretrained=pretrained).eval()
            checkpoint_sha256 = _model_state_sha256(model)
            pretrained_cfg = _jsonable_config(getattr(model, "pretrained_cfg", {}) or {})
            model = model.to(device)
            # 按模型配置确定预处理方式（timm 官方 transform）
            data_config = resolve_data_config({}, model=model)
            transform = create_transform(**data_config, is_training=False)
            preprocess_records.append(
                {
                    "model_variant": variant["variant"],
                    "timm_model": variant["timm_model"],
                    "preprocess_source": "timm.data.resolve_data_config/create_transform",
                    "data_config": _jsonable_config(data_config),
                    "batch_size": batch_size,
                    "pretrained": pretrained,
                    "pretrained_cfg": pretrained_cfg,
                    "checkpoint_state_sha256": checkpoint_sha256,
                }
            )
            module_names = {name for name, _module in model.named_modules() if name}
            linear_names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
            # 根据注入边界类型构造"要注入扰动的模块集合"：
            if injection_boundary in {"hpat-mapping", "reasonable-max", "maximal-all-mac"}:
                # 这三种边界需要算子活动表来计算 HPAT 映射层
                op_csv = operator_activity_csv or REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"
                if not op_csv.exists():
                    raise ValueError(f"HPAT mapping boundary requires operator activity CSV: {op_csv}")
                boundary_builder = (
                    boundary_from_operator_rows
                    if injection_boundary == "hpat-mapping"
                    else (reasonable_max_boundary if injection_boundary == "reasonable-max" else maximal_all_mac_boundary)
                )
                boundary = boundary_builder(read_csv(op_csv), model_variant=variant["variant"], available_module_names=module_names)
                boundary["operator_activity_csv"] = relative(op_csv)
                boundary["operator_activity_csv_sha256"] = sha256_file(op_csv)
                # 映射覆盖率不足时不继续，防止"只注入了极少数层"的误导结果
                if boundary["matched_layer_count"] <= 0 or float(boundary["coverage"]) < min_boundary_coverage:
                    raise ValueError(
                        "HPAT mapping injection boundary coverage is insufficient for "
                        f"{variant['variant']}: coverage={boundary['coverage']:.4f}, "
                        f"matched={boundary['matched_layer_count']}, required>={min_boundary_coverage:.4f}"
                    )
                raw_allowed_module_names = set(boundary["matched_layers"])
                # reasonable-max / maximal-all-mac 会折叠嵌套模块名，只保留最外层注入点
                allowed_module_names = (
                    _collapse_nested_module_names(raw_allowed_module_names)
                    if injection_boundary in {"reasonable-max", "maximal-all-mac"}
                    else raw_allowed_module_names
                )
                boundary["raw_matched_layer_count"] = len(raw_allowed_module_names)
                boundary["injection_site_count"] = len(allowed_module_names)
                boundary["injection_site_policy"] = (
                    "outermost-selected-module; descendants suppressed to avoid nested duplicate injection"
                    if injection_boundary in {"reasonable-max", "maximal-all-mac"}
                    else "matched module outputs"
                )
                boundary["injection_sites"] = sorted(allowed_module_names)
                evidence_kind = (
                    "dataset-coupled MobileViT accuracy fixed subset"
                    if injection_boundary == "hpat-mapping"
                    else "calibrated-proxy-v2 non-overlapping module-output diagnostic; not internal optical-operator accuracy evidence"
                )
            else:
                # all-linear-smoke：只在全部 Linear 层注入（工程自检，不算 HPAT 映射证据）
                boundary = all_linear_smoke_boundary(variant["variant"], module_names, linear_names)
                allowed_module_names = set(boundary["matched_layers"])
                evidence_kind = "all-linear smoke perturbation; not HPAT-mapping accuracy evidence"
            boundary_records.append(boundary)

            # 清洁推理：得到基准 logits、top-1/top-5 与置信度/间隔
            input_batches = _prepare_input_batches(samples, resolution, torch, transform, batch_size)
            clean_logits_batches = []
            clean_top1: list[list[int]] = []
            clean_top5: list[list[int]] = []
            clean_top1_confidence: list[float] = []
            clean_top1_margin: list[float] = []
            for _start, cpu_inputs in input_batches:
                logits = _forward_logits(model, cpu_inputs.to(device), torch, device)
                clean_logits_batches.append(logits)
                batch_top5, _batch_probs, batch_confidence, batch_margin = _topk_detail(logits, torch, 5)
                clean_top5.extend(batch_top5)
                clean_top1.extend([[row[0]] for row in batch_top5])
                clean_top1_confidence.extend(batch_confidence)
                clean_top1_margin.extend(batch_margin)
            clean_logits = torch.cat(clean_logits_batches, dim=0)
            variant_key = _sanitize_key(variant["variant"])
            # 按需保存清洁 logits（summary 模式只保存干净输出）
            if save_logits in {"summary", "full"}:
                clean_npz[f"{variant_key}_clean_logits"] = clean_logits.numpy()
            clean_top1_acc, label_count = _accuracy(clean_top1, labels)
            clean_top5_acc, _ = _accuracy(clean_top5, labels)
            metric_kind = "top-1/top-5 accuracy delta on fixed labeled subset" if label_count else "logit drift on fixed unlabeled subset"
            evidence_label = evidence_kind if label_count else evidence_kind + "; unlabeled logit-drift only"
            injection_location = (
                "HPAT mapped module outputs"
                if injection_boundary in {"hpat-mapping", "reasonable-max", "maximal-all-mac"}
                else "all torch.nn.Linear outputs; smoke only"
            )

            # 中层循环：每个非理想性效果 × 每个严重度取值
            for effect, variable, values in _selected_sweep_plan(config, selected_effects):
                for value in values:
                    seeds_for_effect = _trial_seeds(effect, repeat_seeds, seed)
                    # 内层循环：随机性效果跑多次种子取分布
                    for trial_index, trial_seed in enumerate(seeds_for_effect):
                        # Common random numbers: a seed follows one perturbation direction across
                        # the severity sweep. Including `value` here made adjacent dose points use
                        # unrelated noise/device patterns and produced visually erratic curves.
                        # （中文说明）固定种子只依赖"模型+效果"，不依赖扫描取值：
                        # 这样相邻严重度共享同一条噪声/工艺模式，曲线才平滑可读。
                        perturbation_seed = _trial_torch_seed(trial_seed, variant["variant"], effect)
                        _seed_torch_rng(torch, perturbation_seed, device)
                        perturb_config = dict(config.get("nonideality", {}))
                        perturb_config["_injection_site_count"] = len(allowed_module_names)
                        perturb_fn = _make_perturb_fn(effect, value, torch, rng, perturb_config)
                        setattr(perturb_fn, "_allowed_module_names", allowed_module_names)
                        pert_top1: list[list[int]] = []
                        pert_top5: list[list[int]] = []
                        numerator = 0.0
                        denominator = 0.0
                        perturbed_batches = []
                        # 扰动推理：逐批前向并对比干净 logits，记录相对漂移
                        for start, cpu_inputs in input_batches:
                            perturbed_logits = _forward_logits(model, cpu_inputs.to(device), torch, device, perturb_fn)
                            clean_slice = clean_logits[start : start + perturbed_logits.shape[0]]
                            numerator += torch.sum(torch.abs(perturbed_logits - clean_slice)).item()
                            denominator += torch.sum(torch.abs(clean_slice)).item()
                            batch_top5, _batch_probs, batch_confidence, batch_margin = _topk_detail(perturbed_logits, torch, 5)
                            batch_top1 = [[row[0]] for row in batch_top5]
                            pert_top1.extend(batch_top1)
                            pert_top5.extend(batch_top5)
                            if save_logits == "full":
                                perturbed_batches.append(perturbed_logits.numpy())
                            # 逐样本写预测变化明细：clean vs perturbed 的 top-1 是否翻转
                            for offset, preds in enumerate(batch_top1):
                                sample_index = start + offset
                                label = labels[sample_index]
                                sample = samples[sample_index]
                                clean_top5_for_sample = clean_top5[sample_index]
                                pert_top5_for_sample = batch_top5[offset]
                                clean_label_in_top5 = "" if label is None else int(label in clean_top5_for_sample)
                                pert_label_in_top5 = "" if label is None else int(label in pert_top5_for_sample)
                                prediction_writer.writerow(
                                    {
                                        "variant": variant["variant"],
                                        "trial_index": trial_index,
                                        "trial_seed": trial_seed,
                                        "sample_index": sample_index,
                                        "image_path": _sample_relpath(pathlib.Path(sample["path"]), dataset_root),
                                        "label": "" if label is None else label,
                                        "effect": effect,
                                        "sweep_variable": variable,
                                        "sweep_value": value,
                                        "clean_top1": clean_top1[sample_index][0],
                                        "perturbed_top1": preds[0],
                                        "changed": int(clean_top1[sample_index][0] != preds[0]),
                                        "clean_correct_top1": "" if label is None else int(clean_top1[sample_index][0] == label),
                                        "perturbed_correct_top1": "" if label is None else int(preds[0] == label),
                                    }
                                )
                                # 可选：更详细的 top-5 置信度/间隔/Jaccard 明细
                                if topk_margin_writer is not None:
                                    topk_margin_writer.writerow(
                                        {
                                            "variant": variant["variant"],
                                            "trial_index": trial_index,
                                            "trial_seed": trial_seed,
                                            "sample_index": sample_index,
                                            "image_path": _sample_relpath(pathlib.Path(sample["path"]), dataset_root),
                                            "label": "" if label is None else label,
                                            "class_name": sample.get("class_name", ""),
                                            "synset": sample.get("synset", ""),
                                            "source_id": sample.get("source_id", ""),
                                            "effect": effect,
                                            "sweep_variable": variable,
                                            "sweep_value": value,
                                            "clean_top1": clean_top1[sample_index][0],
                                            "perturbed_top1": preds[0],
                                            "changed": int(clean_top1[sample_index][0] != preds[0]),
                                            "clean_top5": _topk_string(clean_top5_for_sample),
                                            "perturbed_top5": _topk_string(pert_top5_for_sample),
                                            "clean_label_in_top5": clean_label_in_top5,
                                            "perturbed_label_in_top5": pert_label_in_top5,
                                            "clean_correct_top1": "" if label is None else int(clean_top1[sample_index][0] == label),
                                            "perturbed_correct_top1": "" if label is None else int(preds[0] == label),
                                            "clean_correct_top5": clean_label_in_top5,
                                            "perturbed_correct_top5": pert_label_in_top5,
                                            "clean_top1_confidence": f"{clean_top1_confidence[sample_index]:.8f}",
                                            "perturbed_top1_confidence": f"{batch_confidence[offset]:.8f}",
                                            "clean_top1_margin": f"{clean_top1_margin[sample_index]:.8f}",
                                            "perturbed_top1_margin": f"{batch_margin[offset]:.8f}",
                                            "margin_delta": f"{batch_margin[offset] - clean_top1_margin[sample_index]:.8f}",
                                            "top5_jaccard": f"{_jaccard(clean_top5_for_sample, pert_top5_for_sample):.8f}",
                                            "injection_boundary": injection_boundary,
                                            "evidence_label": evidence_label,
                                            "claim_boundary": DATASET_CLAIM_BOUNDARY,
                                        }
                                    )
                        if save_logits == "full":
                            # 完整保存扰动后 logits（供事后离线分析）
                            perturbed_key = f"{variant_key}_{_sanitize_key(effect)}_{_sanitize_key(value)}_seed_{trial_seed}"
                            perturbed_npz[perturbed_key] = np.concatenate(perturbed_batches, axis=0)
                        # 计算本次 trial 的精度、变化率与 logit 相对漂移
                        pert_top1_acc, _ = _accuracy(pert_top1, labels)
                        pert_top5_acc, _ = _accuracy(pert_top5, labels)
                        changed = [int(a[0] != b[0]) for a, b in zip(clean_top1, pert_top1)]
                        change_rate = 100.0 * sum(changed) / max(len(changed), 1)
                        mean_delta = float(numerator / max(denominator, 1e-12))
                        trial_row = _measured_row(
                            variant=variant["variant"],
                            effect=effect,
                            variable=variable,
                            value=value,
                            trials=len(samples),
                            mean_relative_error=mean_delta if not label_count else None,
                            p95_relative_error=mean_delta if not label_count else None,
                            metric_kind=metric_kind,
                            evidence_label=evidence_label,
                            sample_count=len(samples),
                            label_count=label_count,
                            clean_top1=clean_top1_acc if label_count else None,
                            perturbed_top1=pert_top1_acc if label_count else None,
                            clean_top5=clean_top5_acc if label_count else None,
                            perturbed_top5=pert_top5_acc if label_count else None,
                            prediction_change_rate=change_rate,
                            seed=trial_seed,
                            device=backend,
                            injection_location=injection_location,
                            injection_boundary=injection_boundary,
                        )
                        # 补充 trial 专属字段：索引、种子、top-1/top-5 降幅
                        trial_row.update(
                            {
                                "trial_index": trial_index,
                                "trial_seed": trial_seed,
                                "top1_drop": _fmt(max(0.0, clean_top1_acc - pert_top1_acc)) if label_count else "",
                                "top5_drop": _fmt(max(0.0, clean_top5_acc - pert_top5_acc)) if label_count else "",
                                "claim_boundary": DATASET_CLAIM_BOUNDARY,
                            }
                        )
                        trial_rows.append(trial_row)

            # 释放大对象，避免多个变体叠加占满内存
            del model
            del input_batches
            del clean_logits_batches
            del clean_logits
    # 由逐次试验明细汇总出五张结果表
    measured_rows, precision_rows, by_severity_rows, safe_rows, failure_rows = derive_nonideality_summary_tables(
        trial_rows,
        safe_top1_drop_threshold=safe_top1_drop_threshold,
        safe_top5_drop_threshold=safe_top5_drop_threshold,
        safe_prediction_change_threshold=safe_prediction_change_threshold,
    )

    # 按需写出原始产物：预测明细必写；topk-margin / logits 按开关写
    raw_outputs = [relative(prediction_csv)]
    if save_prediction_detail == "topk-margin":
        raw_outputs.append(relative(topk_margin_csv))
    if save_logits in {"summary", "full"}:
        import numpy as np  # type: ignore

        clean_npz_path = raw_dir / "nonideality_logits_clean.npz"
        np.savez_compressed(clean_npz_path, **clean_npz)
        raw_outputs.append(relative(clean_npz_path))
    if save_logits == "full":
        import numpy as np  # type: ignore

        perturbed_npz_path = raw_dir / "nonideality_logits_perturbed.npz"
        np.savez_compressed(perturbed_npz_path, **perturbed_npz)
        raw_outputs.append(relative(perturbed_npz_path))
    # 注入边界清单：记录本次注入了哪些层、覆盖率多少，判定是否可主张
    boundary_manifest = output_dir / "nonideality_injection_boundary_manifest.json"
    write_json(
        boundary_manifest,
        {
            "injection_boundary": injection_boundary,
            "min_boundary_coverage": min_boundary_coverage,
            "records": boundary_records,
            "claim_eligible": injection_boundary == "hpat-mapping",
            "claim_boundary": DATASET_CLAIM_BOUNDARY,
        },
    )
    raw_outputs.append(relative(boundary_manifest))
    # 汇总环境与运行元信息（设备、版本、随机策略、安全阈值等）
    meta = {
        "device": backend,
        "device_reason": device_reason,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "timm_version": getattr(timm, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "rng_policy": (
            "synchronize selected device, then explicitly seed torch global and MPS RNG streams "
            "once before model construction and before every variant/effect/trial perturbation"
        ),
        "floating_diagnostic_policy": (
            "For labelled paper runs, accuracy/transition metrics are canonical. Logit-relative-error summary "
            "fields are omitted, and per-sample confidence/margin detail is retained only as a noncanonical "
            "local diagnostic because long MPS perturbed forwards are not byte-stable at that float granularity."
        ),
        "sample_count": len(samples),
        "label_count": sum(label is not None for label in labels),
        "batch_size": batch_size,
        "save_logits": save_logits,
        "save_prediction_detail": save_prediction_detail,
        "repeat_seeds": repeat_seeds,
        "safe_thresholds": {
            "top1_drop": safe_top1_drop_threshold,
            "top5_drop": safe_top5_drop_threshold,
            "prediction_change_rate": safe_prediction_change_threshold,
        },
        "raw_outputs": raw_outputs,
        "injection_boundary": injection_boundary,
        "boundary_records": boundary_records,
        "preprocess_records": preprocess_records,
        "claim_boundary": DATASET_CLAIM_BOUNDARY,
    }
    return measured_rows, precision_rows, trial_rows, by_severity_rows, safe_rows, failure_rows, meta


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    dataset_root: pathlib.Path | None = None,
    subset_file: pathlib.Path | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    device: str = "auto",
    pretrained: bool = False,
    model_variant: str = "all",
    operator_activity_csv: pathlib.Path | None = None,
    injection_boundary: str = "hpat-mapping",
    min_boundary_coverage: float = 0.5,
    batch_size: int = 32,
    save_logits: str = "full",
    save_prediction_detail: str = "basic",
    repeat_seeds: str | list[int] | None = None,
    safe_top1_drop_threshold: float = 5.0,
    safe_top5_drop_threshold: float = 5.0,
    safe_prediction_change_threshold: float = 10.0,
    selected_effects: str | list[str] | set[str] | None = None,
) -> dict[str, pathlib.Path]:
    """非理想性精度扫描主入口：根据参数决定走"数据集模式"还是"分析模式"。

    数据集模式需要同时提供 --dataset-root 与 --subset-file；否则走纯分析
    （合成代理公式）模式。两种模式产出不同的 CSV 集合。

    参数：见各参数名（命令行动画脚本逐一传入）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    seed = int(seed if seed is not None else config.get("seed", 1))
    repeat_seed_values = _parse_repeat_seeds(repeat_seeds, seed)
    if isinstance(selected_effects, str):
        selected_effect_values = {item.strip() for item in selected_effects.split(",") if item.strip()}
    else:
        selected_effect_values = set(selected_effects or [])
    # 参数合法性校验：save_logits / save_prediction_detail 必须是允许的取值
    save_logits = save_logits.lower()
    if save_logits not in {"summary", "none", "full"}:
        raise ValueError("--save-logits must be one of summary, none, or full")
    save_prediction_detail = save_prediction_detail.lower()
    if save_prediction_detail not in {"basic", "topk-margin"}:
        raise ValueError("--save-prediction-detail must be one of basic or topk-margin")
    # 是否进入数据集模式：只要给了数据集根或子集文件就进入
    dataset_mode = dataset_root is not None or subset_file is not None
    out_csv = tables_dir / "nonideality_accuracy_sweep.csv"
    project_csv = REPO_ROOT / "tables" / "nonideality_accuracy_sweep.csv"
    write_project = project_writes_enabled()
    manifest_path = output_dir / "nonideality_accuracy_sweep_manifest.json"
    manifest = base_manifest(
        "nonideality_accuracy_sweep",
        "dataset-coupled accuracy sweep" if dataset_mode else "modelled/analytical non-ideality sensitivity",
    )
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset_root": relative(dataset_root) if dataset_root else "",
            "subset_file": relative(subset_file) if subset_file else "",
            "subset_file_sha256": sha256_file(subset_file) if subset_file else None,
            "max_samples": max_samples,
            "seed": seed,
            "repeat_seeds": repeat_seed_values,
            "selected_effects": sorted(selected_effect_values),
            "requested_device": device,
            "pretrained": pretrained,
            "model_variant": model_variant,
            "operator_activity_csv": relative(operator_activity_csv) if operator_activity_csv else "",
            "operator_activity_csv_sha256": sha256_file(operator_activity_csv) if operator_activity_csv else None,
            "injection_boundary": injection_boundary,
            "min_boundary_coverage": min_boundary_coverage,
            "batch_size": batch_size,
            "save_logits": save_logits,
            "save_prediction_detail": save_prediction_detail,
            "safe_thresholds": {
                "top1_drop": safe_top1_drop_threshold,
                "top5_drop": safe_top5_drop_threshold,
                "prediction_change_rate": safe_prediction_change_threshold,
            },
            "claim_boundary": DATASET_CLAIM_BOUNDARY if dataset_mode else manifest["claim_boundary"],
        }
    )
    if dataset_mode:
        # 数据集模式：两个路径参数必须都提供，缺一则阻塞
        if not dataset_root or not subset_file:
            manifest.update(
                {
                    "status": "blocked",
                    "blocked_reason": "Dataset mode requires both --dataset-root and --subset-file.",
                    "promotion_note": "Provide a fixed subset before collecting accuracy-coupled non-ideality evidence.",
                }
            )
            write_json(manifest_path, manifest)
            raise ValueError(manifest["blocked_reason"])
        try:
            rows, precision_rows, trial_rows, by_severity_rows, safe_rows, failure_rows, dataset_meta = _run_dataset_mode(
                output_dir,
                config,
                dataset_root,
                subset_file,
                max_samples,
                seed,
                device,
                pretrained,
                model_variant,
                operator_activity_csv,
                injection_boundary,
                min_boundary_coverage,
                batch_size,
                save_logits,
                save_prediction_detail,
                repeat_seed_values,
                safe_top1_drop_threshold,
                safe_top5_drop_threshold,
                safe_prediction_change_threshold,
                selected_effect_values,
            )
        except ValueError as exc:
            # 数据集模式失败（如依赖缺失、覆盖率不足）→ 记录 blocked 后终止
            manifest.update(
                {
                    "status": "blocked",
                    "blocked_reason": str(exc),
                    "promotion_note": "Dataset mode is blocked until the injection boundary is aligned with HPAT mapping coverage requirements.",
                    "caffeinate_used": False,
                    "caffeinate_reason": "Not used; blocked before a long dataset run.",
                }
            )
            write_json(manifest_path, manifest)
            raise
        # 定义并写出数据集模式的七张结果表
        measured_csv = tables_dir / "nonideality_accuracy_measured.csv"
        project_measured_csv = REPO_ROOT / "tables" / "nonideality_accuracy_measured.csv"
        precision_csv = tables_dir / "precision_accuracy_sweep.csv"
        project_precision_csv = REPO_ROOT / "tables" / "precision_accuracy_sweep.csv"
        trials_csv = tables_dir / "nonideality_accuracy_trials.csv"
        project_trials_csv = REPO_ROOT / "tables" / "nonideality_accuracy_trials.csv"
        by_severity_csv = tables_dir / "nonideality_accuracy_by_severity.csv"
        project_by_severity_csv = REPO_ROOT / "tables" / "nonideality_accuracy_by_severity.csv"
        safe_csv = tables_dir / "nonideality_accuracy_safe_region.csv"
        project_safe_csv = REPO_ROOT / "tables" / "nonideality_accuracy_safe_region.csv"
        failure_csv = tables_dir / "nonideality_accuracy_failure_thresholds.csv"
        project_failure_csv = REPO_ROOT / "tables" / "nonideality_accuracy_failure_thresholds.csv"
        write_csv(out_csv, rows, MEASURED_NONIDEALITY_FIELDS)
        write_csv(measured_csv, rows, MEASURED_NONIDEALITY_FIELDS)
        write_csv(precision_csv, precision_rows, PRECISION_ACCURACY_FIELDS)
        write_csv(trials_csv, trial_rows, TRIAL_NONIDEALITY_FIELDS)
        write_csv(by_severity_csv, by_severity_rows, BY_SEVERITY_FIELDS)
        write_csv(safe_csv, safe_rows, SAFE_REGION_FIELDS)
        write_csv(failure_csv, failure_rows, FAILURE_THRESHOLD_FIELDS)
        project_outputs: list[pathlib.Path] = []
        if write_project:
            # 开启项目写入权限时，把七张表同步到仓库公共 tables/
            project_rows = [
                (project_csv, rows, MEASURED_NONIDEALITY_FIELDS),
                (project_measured_csv, rows, MEASURED_NONIDEALITY_FIELDS),
                (project_precision_csv, precision_rows, PRECISION_ACCURACY_FIELDS),
                (project_trials_csv, trial_rows, TRIAL_NONIDEALITY_FIELDS),
                (project_by_severity_csv, by_severity_rows, BY_SEVERITY_FIELDS),
                (project_safe_csv, safe_rows, SAFE_REGION_FIELDS),
                (project_failure_csv, failure_rows, FAILURE_THRESHOLD_FIELDS),
            ]
            for project_path, project_rows_to_write, project_fields in project_rows:
                write_csv(project_path, project_rows_to_write, project_fields)
                project_outputs.append(project_path)
        # 按注入边界类型给出运行状态与"能否主张结论"的判定：
        if injection_boundary == "hpat-mapping":
            run_status = "ok"
            evidence_tier = "fixed-subset local accuracy"
            claim_eligible = True
            promotion_note = (
                "Supports fixed-subset accuracy-coupled claims only within the fixed subset, model, "
                "pretrained/checkpoint state, device, and hook-based injection boundary recorded here. "
                "This is not full ImageNet robustness, silicon validation, or edge/mobile deployment evidence."
            )
        elif injection_boundary == "reasonable-max":
            run_status = "paper_eligible_with_limitations"
            evidence_tier = "local/modelled module-output diagnostic"
            claim_eligible = True
            promotion_note = (
                "Supports only the paper's bounded reasonable-high module-output sensitivity diagnostic. "
                "It is not physical optical-operator injection, full ImageNet robustness, silicon validation, "
                "or measured edge/mobile deployment evidence."
            )
        else:
            run_status = "smoke"
            evidence_tier = "engineering diagnostic"
            claim_eligible = False
            promotion_note = (
                "The selected injection boundary is engineering-only and cannot support HPAT-mapped claims."
            )

        manifest.update(
            {
                "status": run_status,
                "evidence_tier": evidence_tier,
                "claim_eligible": claim_eligible,
                "outputs": [
                    relative(path)
                    for path in [
                        out_csv,
                        measured_csv,
                        precision_csv,
                        trials_csv,
                        by_severity_csv,
                        safe_csv,
                        failure_csv,
                    ]
                ]
                + [relative(path) for path in project_outputs]
                + dataset_meta["raw_outputs"],
                "project_write_performed": write_project,
                "row_count": len(rows),
                "trial_row_count": len(trial_rows),
                "by_severity_row_count": len(by_severity_rows),
                "safe_region_row_count": len(safe_rows),
                "failure_threshold_row_count": len(failure_rows),
                "precision_row_count": len(precision_rows),
                "dataset": dataset_meta,
                "promotion_note": promotion_note,
                # 长数据集运行建议用 caffeinate 防休眠，这里只做记录
                "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED") == "1",
                "caffeinate_reason": (
                    "Caller set HPAT_CAFFEINATE_USED=1; command was expected to be wrapped in caffeinate."
                    if os.environ.get("HPAT_CAFFEINATE_USED") == "1"
                    else "Not used by this script directly; wrap long dataset runs in caffeinate."
                ),
            }
        )
        write_json(manifest_path, manifest)
        result = {
            "csv": out_csv,
            "measured_csv": measured_csv,
            "precision_csv": precision_csv,
            "trials_csv": trials_csv,
            "by_severity_csv": by_severity_csv,
            "safe_region_csv": safe_csv,
            "failure_thresholds_csv": failure_csv,
            "manifest": manifest_path,
        }
        if write_project:
            result["project_csv"] = project_csv
        return result

    # 分析模式（无数据集）：用合成代理公式做快速敏感性估算
    rows = nonideality_rows(config)
    write_csv(out_csv, rows, NONIDEALITY_FIELDS)
    project_outputs = []
    if write_project:
        write_csv(project_csv, rows, NONIDEALITY_FIELDS)
        project_outputs.append(project_csv)
    manifest.update(
        {
            "status": "proxy",
            "outputs": [relative(out_csv)] + [relative(path) for path in project_outputs],
            "project_write_performed": write_project,
            "row_count": len(rows),
            "promotion_note": "Proxy transfer sensitivity only unless connected to real MobileViT logits or simulator transfer matrices.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled sweep.",
        }
    )
    write_json(manifest_path, manifest)
    result = {"csv": out_csv, "manifest": manifest_path}
    if write_project:
        result["project_csv"] = project_csv
    return result


def main() -> None:
    """命令行入口：把命令行参数逐一传给 run() 并打印产出路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subset-file", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model-variant", default="all")
    parser.add_argument("--operator-activity-csv", default="")
    parser.add_argument("--injection-boundary", default="hpat-mapping", choices=["hpat-mapping", "reasonable-max", "maximal-all-mac", "all-linear-smoke"])
    parser.add_argument("--min-boundary-coverage", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-logits", default="full", choices=["summary", "none", "full"])
    parser.add_argument("--save-prediction-detail", default="basic", choices=["basic", "topk-margin"])
    parser.add_argument("--repeat-seeds", default="")
    parser.add_argument("--safe-top1-drop-threshold", type=float, default=5.0)
    parser.add_argument("--safe-top5-drop-threshold", type=float, default=5.0)
    parser.add_argument("--safe-prediction-change-threshold", type=float, default=10.0)
    parser.add_argument("--effects", default="", help="Comma-separated effect names; default runs the full configured sweep")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.dataset_root) if args.dataset_root else None,
        pathlib.Path(args.subset_file) if args.subset_file else None,
        args.max_samples,
        args.seed,
        args.device,
        args.pretrained,
        args.model_variant,
        pathlib.Path(args.operator_activity_csv) if args.operator_activity_csv else None,
        args.injection_boundary,
        args.min_boundary_coverage,
        args.batch_size,
        args.save_logits,
        args.save_prediction_detail,
        args.repeat_seeds,
        args.safe_top1_drop_threshold,
        args.safe_top5_drop_threshold,
        args.safe_prediction_change_threshold,
        args.effects,
    )
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
