"""hpat-artifact 论文证据图渲染脚本（matplotlib 版）。

【这个脚本做什么】
    把 hpat-artifact 可复现实验库（论文《HPAT：光子张量处理器》的配套实验）
    中已经算好的实验结果数据（CSV / JSON 表格）用 matplotlib 渲染成论文插图
    （PNG 位图），包括：能耗对比图、能耗不确定度/包络图、非理想性误差对精度
    影响图、扩展性（scalability）曲线图、面积/热应力代理图等，并生成对应图
    片的元数据 sidecar（JSON）与清单 manifest。

【数据从哪来】
    数据源大多是实验产物，位于仓库根目录 tables/ 下，例如：
        - fig6_evidence_repaired.csv / hpat_energy_by_component*.csv：能耗证据
        - mobilevit_mapping_scenarios.csv / e2e_speedup_bound.csv：算子覆盖与阿姆达尔上界
        - nonideality_accuracy_*.csv：非理想性误差对精度的扫描结果
        - scalability_*.csv / layout_area_feasibility_proxy.csv：扩展性与面积代理
    本脚本不会自己跑实验，只负责"读表 -> 画图 -> 落盘"。

【输出到哪】
    - 渲染出的 PNG 图片：figures/data/matplotlib/ 目录
    - 供 figure sidecar 使用的派生 CSV：同样写到 figures/data/matplotlib/
    - 渲染清单：由 run(output_dir) 写到传入的 output_dir 下的
      render_matplotlib_evidence_figures_manifest.json

【怎么运行】
    通常由 run_all 之类的主控脚本调用，也可以通过命令行单独运行：
        python render_matplotlib_evidence_figures.py --output-dir <输出目录>
    注意：脚本会一次性渲染约 21 张图，运行会生成大量图片文件，请勿在调试时直接执行。

【证据分级说明】
    脚本里大量出现的 evidence_tier / evidence_label 字段用于标注每张图的数据是
    "measured reference"（实测参照）、"modelled estimate"（建模估计）还是
    "provisional"（暂定），避免把建模代理值误当成真实硅片测量结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

# _common 是本实验库的公共工具模块；hpat_eval 包提供能耗/证据强度的计算函数。
# 这里导入的 read_csv/write_csv/write_json 用于读写表格，REPO_ROOT 是仓库根目录。
from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, read_csv, relative, write_csv, write_json
from hpat_eval.e_local import energy_break_even_rows, energy_sensitivity_rank_rows
from hpat_eval.evidence_strength import energy_uncertainty_rows, energy_uncertainty_summary_rows
from hpat_eval.manifests import figure_sidecar


# 渲染出的 PNG / 派生 CSV 统一输出到仓库的 figures/data/matplotlib/ 目录
OUT_DIR = REPO_ROOT / "figures" / "data" / "matplotlib"

# 本文档围绕的三个 MobileViT 骨干网络变体（从小到大）
MODELS = ["MobileViT-XXS", "MobileViT-XS", "MobileViT-S"]
# 图上显示的短标签：XXS / XS / S
MODEL_LABELS = {"MobileViT-XXS": "XXS", "MobileViT-XS": "XS", "MobileViT-S": "S"}
MODEL_SHORT = ["XXS", "XS", "S"]

# 全脚本统一使用的配色板（论文图表常用色盲友好的八色方案）
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "teal": "#56B4E9",
    "gold": "#F0E442",
    "gray": "#595959",
    "darkgray": "#333333",
    "lightgray": "#D9D9D9",
    "lavender": "#8E7CC3",
}

# 三个模型在柱状图/折线图中依次使用的颜色
MODEL_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
# 随图附带的最基本证据来源文件（用于 sidecar 记录数据出处）
EVIDENCE_SOURCES = [
    REPO_ROOT / "tables" / "p0_readiness_summary.json",
    REPO_ROOT / "tables" / "no_silicon_evidence_gate.md",
]
# 能耗分解图里，把过长的组件名压缩成图上能放得下的短标签
COMPONENT_LABELS = {
    "Optical source and passive loss budget": "Optical source/loss",
    "DAC and input modulation": "DAC/input mod.",
    "MRR programming/hold tuning": "MRR tuning",
    "Thermal tuning/tracking": "Thermal tracking",
    "O/E readout": "O/E readout",
    "SRAM/eDRAM local buffers": "Local buffers",
    "Electrical bus/data movement": "Electrical bus",
    "Digital accumulation/control": "Digital control",
    "LayerNorm/softmax/nonlinear electronics": "Nonlinear/norm",
    "Amortized calibration": "Calibration",
}


def _mpl():
    """惰性导入并配置 matplotlib，返回 pyplot 模块。

    只在需要画图时才调用，避免 import 开销影响其它流程。
    - 使用 "Agg" 后端：不弹窗口、不依赖 GUI，直接把图渲染成 PNG 文件（服务器环境必需）。
    - 通过 rcParams 统一定义论文插图风格：衬线字体、600 dpi 高清输出、
      去掉多余边框、浅色网格线等，保证所有图风格一致。
    """
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端：只允许保存图片，不显示窗口
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.6,
            "axes.spines.top": False,  # 去掉上、右两侧的坐标轴边框，更符合论文风格
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return plt


def _path(name: str) -> pathlib.Path:
    """返回输出目录 OUT_DIR 下名为 name 的文件路径（自动创建目录）。"""
    ensure_dir(OUT_DIR)
    return OUT_DIR / name


def _save(fig: Any, path: pathlib.Path) -> pathlib.Path:
    """统一保存并关闭一张 matplotlib 图。

    - 若该图设置了自定义布局矩形 _hpat_tight_rect（供带整体图题/脚注的大图使用），
      则按该矩形做 tight_layout，否则用默认 padding。
    - 以 600 dpi、白底保存为 PNG 后立即关闭 figure 释放内存。
    """
    tight_rect = getattr(fig, "_hpat_tight_rect", None)  # 读取图对象上可能附加的自定义布局
    tight_kwargs = getattr(fig, "_hpat_tight_kwargs", {})
    if tight_rect is None:
        fig.tight_layout(pad=0.35, **tight_kwargs)
    else:
        fig.tight_layout(pad=0.35, rect=tight_rect, **tight_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=600, facecolor="white")  # 裁剪空白边距并高清保存
    import matplotlib.pyplot as plt

    plt.close(fig)  # 关闭当前图，防止多张图累积占用内存
    return path


def _write_source(name: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> pathlib.Path:
    """把派生数据写到输出目录，作为图片的"数据源"CSV 一并归档。

    派生 CSV 与 PNG 同目录存放，便于审稿/复现时核对图上每个数字的来源。
    """
    path = _path(name)
    write_csv(path, rows, fieldnames)
    return path


def _safe_float(value: Any) -> float:
    """把 CSV 里的字符串安全转成 float，空值或 None 一律按 0.0 处理。"""
    if value in ("", None):
        return 0.0
    return float(value)


def _safe_float_or_none(value: Any) -> float | None:
    """把 CSV 字符串安全转成 float；遇到缺失/无效标记则返回 None。

    与 _safe_float 不同：这里保留"缺数据"的语义（None），
    方便绘图代码据此决定是否画 "n/a" 标记，而不是误当 0 画上去。
    无效标记包括空串、"n/a"、"nan" 等。
    """
    if value in ("", None):
        return None
    text = str(value).strip().lower()
    if text in {"n/a", "na", "nan", "not_collected", "none"}:
        return None
    try:
        if math.isnan(float(text)):
            return None
        return float(text)
    except ValueError:
        return None


def _first_existing(paths: list[pathlib.Path]) -> pathlib.Path:
    """返回 paths 中第一个存在且非空的文件；都不满足则返回最后一个（兜底）。

    用于在不同实验结果命名之间做向后兼容：新命名优先，找不到就退回旧命名。
    """
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return paths[-1]


def _existing(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """过滤出 paths 中真实存在且非空的文件（用于 sidecar 记录可选数据出处）。"""
    return [path for path in paths if path.exists() and path.stat().st_size > 0]


def _resolved_energy_component_source() -> pathlib.Path:
    """解析能耗分量数据文件：优先用 trace 驱动（逐算子模拟）版本，否则退回旧版。

    返回的 CSV 描述 HPAT 推理能耗按组件（光源/DAC/MRR 调谐/读出等）的分解。
    """
    return _first_existing(
        [
            REPO_ROOT / "tables" / "hpat_energy_by_component_trace_driven.csv",
            REPO_ROOT / "tables" / "hpat_energy_by_component.csv",
        ]
    )


def _fig6_envelope_normalized_energy_rows(
    source: pathlib.Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[pathlib.Path], bool]:
    """把逐组件能耗按实验配置里的能量包络（envelope）做整体归一化。

    Fig. 6 要求所有组件（如光源、DAC、MRR 调谐等）的能耗之和与论文保留的
    整体能耗估计一致，所以这里给每个模型的组件能耗统一乘一个缩放因子 factor，
    使总和恰好等于 config 中 energy_envelope_mj 的目标值。

    参数：
        source: 原始能耗分量 CSV 路径
        rows:   原始能耗分量行
    返回：
        (normalized, 数据源路径列表, changed)
        - normalized: 归一化后的行（含 raw_energy_mj / normalization_factor 等证据列）
        - 数据源路径列表：若发生了缩放，会额外写入一份派生 CSV 一并记录
        - changed: 是否真的发生了缩放（用于通知调用方需要重算不确定度）
    """
    config = load_json(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json")
    envelope = config.get("energy_envelope_mj", {})  # 每个模型的目标总能耗（mJ）
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("model_variant", "")].append(row)

    normalized: list[dict[str, Any]] = []
    changed = False
    for model in MODELS:
        model_rows = grouped.get(model, [])
        total = sum(_safe_float(row.get("energy_mj")) for row in model_rows)  # 当前组件能耗总和
        target = _safe_float(envelope.get(model, total))  # 目标总能耗，缺配置则保持原值
        factor = target / total if total > 0 and target > 0 else 1.0  # 统一缩放系数
        if abs(factor - 1.0) > 0.01:
            changed = True
        for row in model_rows:
            updated = dict(row)
            raw_energy = _safe_float(row.get("energy_mj"))
            energy = raw_energy * factor  # 缩放后即满足包络约束的组件能耗
            updated["energy_mj"] = f"{energy:.6f}"
            updated["share_percent"] = f"{100.0 * energy / target if target else 0.0:.4f}"
            updated["raw_energy_mj"] = row.get("raw_energy_mj") or f"{raw_energy:.9f}"  # 保留原始值便于追溯
            existing_factor = _safe_float(row.get("normalization_factor") or 1.0)
            updated["normalization_factor"] = f"{existing_factor * factor:.6f}"  # 累积缩放因子
            updated["normalization_target_mj"] = f"{target:.6f}" if target else row.get("normalization_target_mj", "")
            updated["evidence_label"] = (
                row.get("evidence_label", "")
                + "; envelope-normalized plotting source for Fig. 6"  # 标记数据已做过包络归一化
            ).strip("; ")
            updated["evidence_tier"] = (
                row.get("evidence_tier", "")
                + "; envelope-normalized modelled proxy"  # 强调是建模代理值，非实测
            ).strip("; ")
            normalized.append(updated)

    fieldnames = list(rows[0].keys()) if rows else ["status"]
    if changed:
        derived = _write_source("mpl_fig6_energy_envelope_normalized_source.csv", normalized, fieldnames)
        return normalized, [source, derived], True
    return rows, [source], False


def _resolved_uncertainty_rows(
    component_rows: list[dict[str, Any]], force_recompute: bool = False, source_prefix: str = "mpl"
) -> tuple[list[dict[str, Any]], list[pathlib.Path]]:
    """解析（必要时重新计算）能耗不确定度扫描行。

    论文 Fig. 6 的 (b) 面板需要每个模型的 p05/nominal/p50/p95 四种能耗情景。
    若 tables/energy_uncertainty_summary.csv 里四种情景齐全且未被要求强制重算，
    则直接复用已有结果；否则调用 hpat_eval 里的 energy_uncertainty_rows 按
    能耗分量重新扫描，并把重算结果写盘归档。

    参数：
        component_rows: 能耗分量行（作为扫描的输入基线）
        force_recompute: True 时无视已有汇总文件，强制重算
        source_prefix: 派生 CSV 的文件名前缀（避免多张图互相覆盖）
    返回：
        (summary_rows, 数据源路径列表)
    """
    summary_source = REPO_ROOT / "tables" / "energy_uncertainty_summary.csv"
    rows = read_csv(summary_source) if summary_source.exists() and summary_source.stat().st_size > 0 else []
    models_in_components = {row.get("model_variant", "") for row in component_rows if row.get("model_variant")}
    expected_cases = {"p05", "nominal", "p50", "p95"}  # 四种不确定度情景名
    complete = bool(rows) and all(
        expected_cases.issubset({row.get("case", "") for row in rows if row.get("model") == model})
        for model in models_in_components
    )
    if complete and not force_recompute:
        return rows, [summary_source, REPO_ROOT / "tables" / "energy_uncertainty_sweep.csv"]
    config = load_json(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json")
    sweep_rows = energy_uncertainty_rows(component_rows, config)  # 逐情景扫描得到能耗
    summary_rows = energy_uncertainty_summary_rows(sweep_rows)  # 汇总成 p05/p50/p95 等统计行
    derived_sweep = _write_source(
        f"{source_prefix}_energy_uncertainty_sweep_resolved.csv",
        sweep_rows,
        list(sweep_rows[0].keys()) if sweep_rows else ["status"],
    )
    derived_summary = _write_source(
        f"{source_prefix}_energy_uncertainty_summary_resolved.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else ["status"],
    )
    source_paths = [] if force_recompute else [summary_source]
    return summary_rows, source_paths + [derived_sweep, derived_summary]


def _panel_label(ax: Any, label: str) -> None:
    """在子图 (subplot) 左上角外侧写面板编号，如 (a)、(b)，用于多面板论文图。"""
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _fig6_platform_label(label: str) -> str:
    """把 Fig. 6 数据源里的平台长标签规整成图上显示的短标签。

    例如 "CPU desktop reference, draft-extracted/provisional" -> "Intel i9-13900\n(provisional)"。
    换行是为了在图上分行显示，标注 "provisional"（暂定）表明该参照数据可靠度有限。
    """
    normalized = (
        label.replace(", draft-extracted/provisional", " provisional")
        .replace(", draft-extracted provenance", " provisional")
        .replace(", measured reference", " measured")
        .replace("Local MPS measured desktop reference", "Local MPS measured")
        .replace("HPAT estimated/modelled", "HPAT modelled")
    )
    if normalized.startswith("CPU desktop reference"):
        if "provisional" in normalized:
            return "Intel i9-13900\n(provisional)"
        return "Intel i9-13900"
    if normalized.startswith("GPU desktop reference") or normalized.startswith("GPU measured desktop reference"):
        if "provisional" in normalized:
            return "NVIDIA RTX 4060\n(provisional)"
        return "NVIDIA RTX 4060"
    if normalized.startswith("Apple M5 Pro MPS") or normalized.startswith("Local MPS measured"):
        return "Apple M5 Pro\nMPS"
    if normalized.startswith("HPAT modelled"):
        return "HPAT\nmodelled"
    return normalized


def _fig6_platform_rank(label: str) -> int:
    """给平台定一个固定排序序号，保证柱状图里平台从左到右顺序稳定。

    顺序按"证据由弱到强"排列：CPU 参照 -> GPU 参照 -> M5 Pro -> HPAT（建模）。
    """
    if label.startswith("CPU desktop reference"):
        return 0
    if label.startswith("GPU desktop reference") or label.startswith("GPU measured desktop reference"):
        return 1
    if label.startswith("Apple M5 Pro MPS") or label.startswith("Local MPS measured"):
        return 2
    if label.startswith("HPAT"):
        return 3
    return 10


def _fig6_platform_color(label: str) -> str:
    """按平台类别返回柱状图颜色，方便跨图一眼区分 CPU/GPU/M5/HPAT。"""
    if label.startswith("CPU"):
        return COLORS["orange"]
    if label.startswith("GPU"):
        return COLORS["blue"]
    if label.startswith("HPAT"):
        return COLORS["green"]
    if label.startswith("Apple M5 Pro MPS") or label.startswith("Local MPS"):
        return COLORS["purple"]
    return COLORS["gray"]


def _fig6_missing_y(metric: str, metric_rows: list[dict[str, Any]]) -> float:
    """计算缺数据平台时 "n/a" 文字应放置的纵坐标高度。

    取该指标现存正值的较小者乘以系数（能耗取 0.55、其它取 0.35），
    并设下限（能耗 0.8、其它 0.25），保证 "n/a" 文字画在柱状图底部附近、
    不与真实柱体重叠。
    """
    values = [_safe_float_or_none(row.get("value")) for row in metric_rows]
    positive = [value for value in values if value is not None and value > 0]
    if not positive:
        return 0.0
    if metric == "energy":
        return max(0.8, min(positive) * 0.55)
    return max(0.25, min(positive) * 0.35)


def _fig7_mixed_evidence_context() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 7：多个参照平台与 HPAT 建模估计的延迟/能效对比图。

    这是"混合证据背景图"：把保留的实测参照（i9-13900H CPU、RTX 4060 Laptop GPU、
    M5 Pro CPU/GPU 的 powermetrics 本地实测）与 A16 手机 Core ML 全策略行、
    HPAT 建模估计并排比较。左面板画延迟（ms，对数轴），右面板画能效
    （inf./J，由 1000 / 每推理能耗 mJ 推导，对数轴，越大越好）。
    数据来源包括 fig6_evidence_repaired.csv、本地 MPS 功率探针结果、
    以及先前图 7 上下文 CSV。HPAT 行画斜线填充以强调其建模（非实测）属性。

    返回：与其他图构建函数一致的 (png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    # ---- 汇总四个数据来源：论文证据修复表、A16 上下文、M5 Pro 的 GPU 与 CPU 功率探针 ----
    hpat_source = REPO_ROOT / "tables" / "fig6_evidence_repaired.csv"
    a16_source = REPO_ROOT / "figures" / "data" / "matplotlib" / "mpl_fig7_a16_hpat_context.csv"
    m5_gpu_source = REPO_ROOT / "experiments" / "results" / "mps_power_probe_latest" / "tables" / "mobilevit_mps_power_measured.csv"
    m5_cpu_source = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "m5pro_cpu_power_probe_fig7_admin_20260709"
        / "tables"
        / "mobilevit_mps_power_measured.csv"
    )
    hpat_rows = read_csv(hpat_source)
    a16_rows = read_csv(a16_source) if a16_source.exists() else []
    m5_cpu_rows = read_csv(m5_cpu_source) if m5_cpu_source.exists() else []
    # 统一列结构：每条记录都带 evidence_tier 与 safe_interpretation，便于追溯数据性质
    fieldnames = [
        "model_variant",
        "metric",
        "platform_key",
        "platform_label",
        "value",
        "unit",
        "evidence_tier",
        "source",
        "aggregation_level",
        "status",
        "safe_interpretation",
        "must_not_imply",
    ]
    source_rows: list[dict[str, Any]] = []

    # 第一轮：从 fig6_evidence_repaired.csv 提取 CPU/GPU/HPAT 三类的延迟、能耗、功率行
    for row in hpat_rows:
        label = row.get("platform_label", "")
        metric = row.get("metric", "")
        if metric not in {"latency", "energy", "power"}:
            continue
        platform: tuple[str, str] | None = None
        if label == "CPU desktop reference, measured reference":
            platform = ("i9_13900h", "i9-13900H CPU")
        elif label == "GPU measured desktop reference":
            platform = ("rtx4060_laptop", "RTX 4060 Laptop")
        elif label == "HPAT estimated/modelled":
            platform = ("hpat", "HPAT")
        if platform is None:
            continue
        key, display_label = platform
        source_rows.append(
            {
                "model_variant": row["model_variant"],
                "metric": metric,
                "platform_key": key,
                "platform_label": display_label,
                "value": row["value"],
                "unit": "mJ/inference" if metric == "energy" else row.get("unit", ""),
                "evidence_tier": row.get("evidence_tier", ""),
                "source": relative(hpat_source) if key == "hpat" else row.get("source", ""),
                "aggregation_level": "modelled" if key == "hpat" else "retained_measured_reference",
                "status": "",
                "safe_interpretation": (
                    "HPAT architecture-level modelled estimate."  # HPAT 行明确标注为架构级建模估计
                    if key == "hpat"
                    else f"{display_label} retained measured reference; source provenance is preserved in this row."
                ),
                "must_not_imply": row.get("must_not_imply", ""),
            }
        )

    # 第二轮：同一张证据表里单独抽出 Apple M5 Pro MPS/GPU 的本地实测行
    for row in hpat_rows:
        if row.get("platform_label") != "Apple M5 Pro MPS desktop reference" or row.get("metric") not in {"latency", "energy", "power"}:
            continue
        source_rows.append(
            {
                "model_variant": row["model_variant"],
                "metric": row["metric"],
                "platform_key": "m5_gpu",
                "platform_label": "M5 Pro GPU",
                "value": row["value"],
                "unit": "mJ/inference" if row["metric"] == "energy" else row.get("unit", ""),
                "evidence_tier": row.get("evidence_tier", ""),
                "source": row.get("source", ""),
                "aggregation_level": "mps_gpu_reference",
                "status": "",
                "safe_interpretation": "Apple M5 Pro MPS/GPU local desktop reference with powermetrics energy when available.",
                "must_not_imply": row.get("must_not_imply", ""),
            }
        )

    # 第三轮：M5 Pro CPU 本地 powermetrics 探针结果（空闲扣除后的窗口功耗）
    m5_cpu_metric_fields = {
        "latency": ("latency_ms_mean", "ms"),
        "energy": ("energy_mj_per_inference_mean", "mJ/inference"),
        "power": ("idle_subtracted_power_w_mean", "W"),
    }
    for row in m5_cpu_rows:
        if row.get("status") != "ok" or row.get("backend") != "cpu":
            continue
        for metric, (field, unit) in m5_cpu_metric_fields.items():
            value = row.get(field, "")
            if _safe_float_or_none(value) is None:
                continue
            source_rows.append(
                {
                    "model_variant": row["variant"],
                    "metric": metric,
                    "platform_key": "m5_cpu",
                    "platform_label": "M5 Pro CPU",
                    "value": value,
                    "unit": unit,
                    "evidence_tier": "measured desktop reference",
                    "source": relative(m5_cpu_source),
                    "aggregation_level": "cpu_powermetrics_idle_subtracted",
                    "status": "",
                    "safe_interpretation": "Apple M5 Pro CPU local desktop reference with powermetrics idle-subtracted window power.",
                    "must_not_imply": "Mobile/edge deployment power, calibrated CPU rail power, HPAT silicon power, or cross-platform energy-efficiency superiority.",
                }
            )

    # 第四轮：A16 手机 Core ML 全策略（all-policy）行，作为移动端能耗参照上下文
    for row in a16_rows:
        if row.get("platform_key") != "a16_all" or row.get("metric") not in {"latency", "energy", "power"}:
            continue
        source_rows.append(
            {
                "model_variant": row["model_variant"].replace("-duration-allocated", ""),
                "metric": row["metric"],
                "platform_key": "a16_all",
                "platform_label": "A16 all",
                "value": row["value"],
                "unit": row.get("unit", ""),
                "evidence_tier": row.get("evidence_tier", ""),
                "source": row.get("source", relative(a16_source)),
                "aggregation_level": row.get("aggregation_level", ""),
                "status": row.get("status", ""),
                "safe_interpretation": row.get("safe_interpretation", "A16 Core ML all-policy context."),
                "must_not_imply": row.get("must_not_imply", "HPAT mobile deployment or calibrated per-rail power."),
            }
        )

    display_rows: list[dict[str, Any]] = []
    for row in source_rows:
        metric = row.get("metric", "")
        if metric == "latency":
            display_rows.append(row)
        elif metric == "energy":
            # 把每推理能耗（mJ）换算成能效指标 inf./J = 1000 / mJ，便于与延迟并列比较
            energy_mj = _safe_float_or_none(row.get("value"))
            if energy_mj is None or energy_mj <= 0:
                continue
            display_rows.append(
                {
                    **row,
                    "metric": "energy_efficiency",
                    "value": f"{1000.0 / energy_mj:.6f}",
                    "unit": "inf./J",
                    "aggregation_level": f"derived_from_{row.get('aggregation_level', 'energy')}_energy_mj_per_inference",
                    "safe_interpretation": (
                        f"Energy efficiency derived as 1000 / energy_mJ_per_inference from {row.get('platform_label', 'this row')}."
                    ),
                    "must_not_imply": (
                        row.get("must_not_imply", "")
                        + " Do not treat derived inf./J as a separately measured power counter."
                    ).strip(),
                }
            )
    source_rows = display_rows

    # 把拼好的统一结构数据写盘，供 sidecar（图数据出处说明）记录
    source_csv = _write_source(
        "mpl_fig7_requested_hpat_context.csv",
        source_rows,
        fieldnames,
    )

    metrics = ["latency", "energy_efficiency"]  # 左右两个面板分别画这两个指标
    units = {"latency": "ms (log)", "energy_efficiency": "inf./J (log, higher better)"}
    platforms = ["a16_all", "i9_13900h", "rtx4060_laptop", "m5_cpu", "m5_gpu", "hpat"]
    # 只保留真实有数据的平台，避免画出全空的图例
    platforms = [platform for platform in platforms if any(row["platform_key"] == platform for row in source_rows)]
    platform_labels = {
        "a16_all": "A16 all",
        "i9_13900h": "i9-13900H",
        "rtx4060_laptop": "RTX4060 Laptop",
        "m5_cpu": "M5 Pro CPU",
        "m5_gpu": "M5 Pro GPU",
        "hpat": "HPAT",
    }
    platform_colors = {
        "a16_all": COLORS["orange"],
        "i9_13900h": COLORS["gray"],
        "rtx4060_laptop": COLORS["blue"],
        "m5_cpu": COLORS["teal"],
        "m5_gpu": COLORS["purple"],
        "hpat": COLORS["green"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.05))  # 1 行 2 列：左延迟、右能效
    for ax, metric in zip(axes, metrics):
        metric_rows = [r for r in source_rows if r["metric"] == metric]
        x = list(range(len(MODELS)))
        width = 0.12
        for idx, platform in enumerate(platforms):
            # 逐个平台取三个模型的指标值，缺失的数据记录为 None
            values = []
            for model in MODELS:
                match = [r for r in metric_rows if r["model_variant"] == model and r["platform_key"] == platform]
                values.append(_safe_float_or_none(match[0]["value"]) if match else None)
            offset = (idx - (len(platforms) - 1) / 2.0) * width  # 把多平台柱子错开
            xpos = [v + offset for v in x]
            bar_x = [xv for xv, yv in zip(xpos, values) if yv is not None]  # 只画有值的柱子
            bar_y = [yv for yv in values if yv is not None]
            if bar_x:
                # HPAT 建模柱用斜线填充 + 深色描边，和实测参照区分开
                ax.bar(
                    bar_x,
                    bar_y,
                    width=width,
                    label=platform_labels[platform],
                    color=platform_colors[platform],
                    hatch="//" if platform == "hpat" else None,
                    edgecolor="#222222" if platform == "hpat" else "white",
                    linewidth=0.45,
                )
            missing_x = [xv for xv, yv in zip(xpos, values) if yv is None]
            if missing_x:
                # 缺数据的平台在该位置画旋转的 "n/a" 标记，表示无该数据而非为 0
                missing_y = _fig6_missing_y(metric, metric_rows)
                for mx in missing_x:
                    ax.text(
                        mx,
                        missing_y,
                        "n/a",
                        color=platform_colors[platform],
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6.1,
                        fontweight="bold",
                    )
        ax.set_title({"latency": "Latency", "energy_efficiency": "Energy efficiency"}[metric])
        ax.set_xticks(x, MODEL_SHORT)
        ax.set_ylabel(units[metric])
        ax.set_yscale("log")  # 两个指标跨度都很大，用对数轴便于比较
        ax.set_axisbelow(True)
    from matplotlib.patches import Patch

    # 手动构造统一图例（Patch 矩形色块），避免只出现有数据平台的零散图例
    legend_handles = [
        Patch(
            facecolor=platform_colors[platform],
            edgecolor="#222222" if platform == "hpat" else "white",
            hatch="//" if platform == "hpat" else None,
            label=platform_labels[platform],
        )
        for platform in platforms
    ]
    fig.legend(
        legend_handles,
        [platform_labels[platform] for platform in platforms],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.115),
    )
    fig.text(
        0.5,
        0.025,
        "Bars are ordered approximately from weaker to stronger context. A16 uses Core ML all-policy rows; "
        "M5 Pro CPU/GPU use local CPU and MPS/powermetrics rows. Energy efficiency = 1000 / energy per inference (mJ). "
        "HPAT rows are modelled architecture estimates.",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    fig.suptitle("Latency and energy-efficiency context vs. HPAT modelled estimates", y=0.99, fontsize=11)
    fig._hpat_tight_rect = (0.0, 0.22, 1.0, 0.91)  # 给底部注释行留出空间的自定义布局矩形
    png = _save(fig, _path("mpl_fig7_mixed_evidence_context.png"))
    # sidecar 需要记录所有数据出处，包括那些"可能存在但已被清理"的可选文件
    source_paths = [source_csv, hpat_source, a16_source, m5_gpu_source, m5_cpu_source] + _existing(
        [
            REPO_ROOT / "tables" / "mobilevit_desktop_baseline_measured.csv",
            REPO_ROOT / "experiments" / "results" / "iphone14promax_a16_compute_units_full_matrix_4h_20260709_031715" / "analysis" / "iphone14promax_a16_coreml_compute_units_latency.csv",
        ]
    )
    return (
        png,
        source_paths,
        "mixed: retained measured references / author-measured A16 phone all-policy rows / local M5 Pro CPU-GPU rows / modelled HPAT",
        "Matplotlib-rendered requested Fig. 7 context using grouped latency and derived energy-efficiency panels. "
        "Compared rows are i9-13900H display label from retained CPU reference, RTX 4060 Laptop display label from retained GPU reference, "
        "M5 Pro CPU local powermetrics reference, M5 Pro GPU/MPS local powermetrics reference, "
        "A16 Core ML all-policy rows, and HPAT modelled architecture estimates. "
        "Energy efficiency is derived as 1000 / energy per inference in mJ from each source row. "
        "A16 phone energy remains a coarse battery-discharge diagnostic rather than a calibrated rail measurement; "
        "HPAT rows are modelled and not measured silicon or mobile deployment results.",
    )


def _fig5_operator_amdahl_combo() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 5：映射优化机会与阿姆达尔（Amdahl）设计界组合图。

    (a) 左面板：不同映射边界（Linear 仅线性层 / +PW+attn 加入逐点卷积与注意力 /
        All-conv ceiling 全部卷积）可映射 MAC（乘累加）占比柱状图；
    (b) 右面板：以"映射子图加速假设"为横轴（log2 刻度）的整模型端到端加速比
        上界曲线，说明即使某段被光计算加速，整体加速受阿姆达尔定律限制。
    数据来自 mobilevit_mapping_scenarios.csv 与 mapping_scenario_speedup_bound.csv，
    均为本地 trace/形状推导的建模结果，不是实测 HPAT 加速比。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    operator_source = REPO_ROOT / "tables" / "mobilevit_mapping_scenarios.csv"
    bound_source = REPO_ROOT / "tables" / "mapping_scenario_speedup_bound.csv"
    operator_rows = read_csv(operator_source)
    bound_rows = read_csv(bound_source)

    # 三种映射边界场景及图上显示名/颜色
    scenarios = [
        ("linear_only", "Linear", COLORS["green"]),
        ("plus_pointwise_attention", "+PW+attn", "#2A9D8F"),
        ("plus_all_convolution", "All-conv ceiling", "#E9A000"),
    ]
    source_rows: list[dict[str, Any]] = []
    for row in operator_rows:
        source_rows.append(
            {
                "panel": "operator_coverage",
                "model": row["model"],
                "scenario": row["scenario"],
                "metric": "mapped_mac_share_percent",
                "assumption": "",
                "value": row["mapped_mac_share_percent"],
                "evidence_label": f"{row.get('evidence_tier', '')}; {row.get('claim_status', '')}",
            }
        )

    assumptions = sorted({int(float(r["optical_speedup_assumption"])) for r in bound_rows})
    for row in bound_rows:
        source_rows.append(
            {
                "panel": "amdahl_bound",
                "model": row["model"],
                "scenario": row["scenario"],
                "metric": "e2e_speedup_upper_bound",
                "assumption": row["optical_speedup_assumption"],
                "value": row["e2e_speedup_upper_bound"],
                "evidence_label": row.get("evidence_label", ""),
            }
        )
    # 把两张子图各自的数据合并成一张派生 CSV，随图归档便于核对
    derived = _write_source(
        "mpl_fig5_operator_amdahl_combo.csv",
        source_rows,
        ["panel", "model", "scenario", "metric", "assumption", "value", "evidence_label"],
    )

    # 1 行 2 列子图：左侧略窄画映射占比柱状图，右侧略宽画加速比上界曲线
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), gridspec_kw={"width_ratios": [0.92, 1.08]})
    ax0, ax1 = axes
    x = list(range(len(MODELS)))
    width = 0.24
    for scenario_index, (scenario, label, color) in enumerate(scenarios):
        # 取三种映射边界下三个模型的可映射 MAC 占比
        values = [
            _safe_float(next(r for r in operator_rows if r["model"] == model and r["scenario"] == scenario)["mapped_mac_share_percent"])
            for model in MODELS
        ]
        bars = ax0.bar(
            [value + (scenario_index - 1) * width for value in x],  # 三个场景的柱子并排错开
            values,
            width,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
            hatch="///" if scenario == "plus_all_convolution" else None,  # 全卷积"天花板"用斜线区分
            label=label,
        )
        # 柱顶标注数值；全卷积场景因是上界估计，加 * 号表示"天花板"
        for bar, value in zip(bars, values):
            text_value = f"{value:.0f}*" if scenario == "plus_all_convolution" else f"{value:.1f}"
            ax0.text(bar.get_x() + bar.get_width() / 2, value + 1.5, text_value, ha="center", va="bottom", fontsize=6.8)
    ax0.set_xticks(x, MODEL_SHORT)
    ax0.set_ylim(0, 106)
    ax0.set_ylabel("Mapped MAC share (%)")
    ax0.set_title("Mapping opportunity by boundary")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=5.8, columnspacing=0.8, handlelength=1.8)
    _panel_label(ax0, "(a)")

    # 右面板：每个模型画两条阿姆达尔上界曲线（Linear 实线、Extension 虚线）
    for model, color in zip(MODELS, MODEL_COLORS):
        for scenario, scenario_label, linestyle in [
            ("linear_only", "Linear", "-"),
            ("plus_pointwise_attention", "Extension", "--"),
        ]:
            vals = sorted(
                [r for r in bound_rows if r["model"] == model and r["scenario"] == scenario],
                key=lambda r: _safe_float(r["optical_speedup_assumption"]),  # 按加速假设从小到大排序
            )
            ax1.plot(
                [_safe_float(r["optical_speedup_assumption"]) for r in vals],
                [_safe_float(r["e2e_speedup_upper_bound"]) for r in vals],
                marker="o",
                markersize=3.0,
                color=color,
                linestyle=linestyle,
                label=f"{MODEL_LABELS[model]} {scenario_label}",
            )
    ax1.set_xscale("log", base=2)  # 加速假设覆盖 2^n 量级，用 log2 刻度
    ax1.set_xticks(assumptions)
    ax1.set_xticklabels([f"{v}x" for v in assumptions])
    ax1.set_xlabel("Mapped-subgraph speedup assumption")
    ax1.set_ylabel("Whole-model design bound")
    ax1.set_title("Amdahl-style scenario bounds")
    ax1.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.2)
    _panel_label(ax1, "(b)")

    png = _save(fig, _path("mpl_fig5_operator_amdahl_combo.png"))
    return (
        png,
        [operator_source, bound_source, derived]
        + _existing(EVIDENCE_SOURCES + [REPO_ROOT / "tables" / "p1_readiness_summary.json"]),
        "local trace/shape-derived mapping scenarios and modelled bounds; not measured HPAT speedup",
        "Mapping-optimization Fig. 5: conservative Linear, pointwise/attention extension, and all-convolution MAC ceiling with Amdahl design bounds. Not measured HPAT silicon, edge deployment, or speedup.",
    )


def _fig6_energy_uncertainty_combo() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 6：HPAT 能耗分解 + 能耗不确定度范围组合图。

    (a) 左面板：三个模型的组件能耗堆叠柱状图（光源/损耗、DAC/输入调制、
        MRR 调谐、热调谐、O/E 读出、本地缓存、电总线、数字控制、非线性/归一化、
        校准等组件逐层堆叠），并统一归一化到论文保留的整体能耗包络（envelope）。
    (b) 右面板：p05 / nominal / p50 / p95 四种能耗情景的误差棒图，展示模型化
        能耗估计的不确定度范围。
    数据来自能耗分量 CSV 与不确定度扫描结果，全部为建模代理值而非实测硅片能耗。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    energy_source = _resolved_energy_component_source()
    raw_energy_rows = read_csv(energy_source)
    energy_rows, energy_plot_sources, normalized_to_envelope = _fig6_envelope_normalized_energy_rows(
        energy_source, raw_energy_rows
    )
    uncertainty_rows, uncertainty_sources = _resolved_uncertainty_rows(
        energy_rows,
        force_recompute=normalized_to_envelope,
        source_prefix="mpl_fig6_energy_envelope",
    )

    # 收集全部能耗分量组件名（保持 CSV 中出现顺序），作为堆叠柱的图例顺序
    components = []
    for row in energy_rows:
        if row["component_group"] not in components:
            components.append(row["component_group"])
    component_palette = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["red"],
        COLORS["teal"],
        COLORS["gold"],
        COLORS["gray"],
        COLORS["lavender"],
        "#7AA650",
    ]

    source_rows: list[dict[str, Any]] = []
    for row in energy_rows:
        source_rows.append(
            {
                "panel": "component_energy",
                "model": row["model_variant"],
                "metric": row["component_group"],
                "case": "",
                "value": row["energy_mj"],
                "evidence_label": row.get("evidence_label", ""),
            }
        )
    for row in uncertainty_rows:
        source_rows.append(
            {
                "panel": "uncertainty_envelope",
                "model": row["model"],
                "metric": "energy_mj",
                "case": row["case"],
                "value": row["energy_mj"],
                "evidence_label": row.get("evidence_label", ""),
            }
        )
    derived = _write_source(
        "mpl_fig6_energy_uncertainty_combo.csv",
        source_rows,
        ["panel", "model", "metric", "case", "value", "evidence_label"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25), gridspec_kw={"width_ratios": [1.18, 0.82]})
    ax0, ax1 = axes
    # (a) 面板：按组件逐层堆叠的能耗分解柱状图（三个模型并排）
    bottoms = [0.0, 0.0, 0.0]
    handles = []
    for idx, component in enumerate(components):
        values = []
        for model in MODELS:
            match = [r for r in energy_rows if r["model_variant"] == model and r["component_group"] == component]
            values.append(_safe_float(match[0]["energy_mj"]) if match else 0.0)
        bars = ax0.bar(MODEL_SHORT, values, bottom=bottoms, color=component_palette[idx % len(component_palette)], label=COMPONENT_LABELS.get(component, component))
        handles.append(bars[0])
        bottoms = [a + b for a, b in zip(bottoms, values)]  # 累计各组件高度实现堆叠
    ax0.set_ylabel("Modelled energy (mJ)")
    ax0.set_title("Component energy breakdown")
    _panel_label(ax0, "(a)")

    # (b) 面板：四种不确定度情景（p05/nominal/p50/p95）下的整体能耗
    x = list(range(len(MODELS)))
    p05 = []
    nominal = []
    p50 = []
    p95 = []
    for model in MODELS:
        by_case = {r["case"]: _safe_float(r["energy_mj"]) for r in uncertainty_rows if r["model"] == model}
        p05.append(by_case["p05"])  # 悲观情景（第 5 百分位）
        nominal.append(by_case["nominal"])  # 默认估计
        p50.append(by_case["p50"])  # 中间情景
        p95.append(by_case["p95"])  # 乐观情景（第 95 百分位）
    lower = [max(0.0, n - lo) for n, lo in zip(nominal, p05)]  # 名义值向下界长度
    upper = [hi - n for n, hi in zip(nominal, p95)]  # 名义值向上界长度
    # 用误差棒（errorbar）画出 nominal ± 不确定度区间，菱形点单独标出中间情景
    ax1.errorbar(
        x,
        nominal,
        yerr=[lower, upper],
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        elinewidth=1.1,
        capsize=3.0,
        label="default estimate, uncertainty range",
    )
    ax1.scatter(x, p50, marker="D", s=20, color=COLORS["orange"], zorder=3, label="middle scenario")
    ax1.set_xticks(x, MODEL_SHORT)
    ax1.set_ylabel("Modelled energy (mJ)")
    ax1.set_title("Energy uncertainty range")
    ax1.legend(frameon=False, loc="upper left")
    _panel_label(ax1, "(b)")

    fig.legend(
        handles,
        [COMPONENT_LABELS.get(component, component) for component in components],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Model-based HPAT energy and uncertainty", y=1.02, fontsize=10)
    png = _save(fig, _path("mpl_fig6_energy_uncertainty_combo.png"))
    normalization_note = (
        " Component values and uncertainty range are envelope-normalized to the retained HPAT energy estimates so Fig. 6 uses one mJ scale."
        if normalized_to_envelope
        else ""
    )
    return (
        png,
        [*energy_plot_sources, *uncertainty_sources, derived]
        + _existing(
            EVIDENCE_SOURCES
            + [
                REPO_ROOT / "tables" / "energy_unit_cost_source_ledger.csv",
                REPO_ROOT / "tables" / "p1_readiness_summary.json",
            ]
        ),
        "model-based energy/uncertainty; not silicon energy",
        "Matplotlib-rendered evidence-strengthening Fig. 6: modelled HPAT component energy plus uncertainty range."
        + normalization_note
        + " Supports retained energy discussion; values are modelled proxies, not HPAT silicon or edge measurements.",
    )


def _supp_energy_break_even_sensitivity() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染补充图：能耗盈亏平衡压力扫描（heatmap）+ 能耗敏感度排名（横向条形图）。

    作为反驳审稿意见（rebuttal）的备用证据：
    (a) 左面板：以"DAC 转换器能耗乘子"和"光源能耗乘子"为两个轴的热力图，
        每个格子的数值是某组乘子下整模型能耗相对参照能耗的百分比，展示
        关键组件能耗恶化到多少倍时 HPAT 能耗优势才被"吃掉"（盈亏平衡点）；
    (b) 右面板：按"能量波动占名义值的百分比"排序的各因素敏感度排名。
    若 tables 下已有现成扫描结果就直接读表，否则现场用 hpat_eval 重新计算。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    break_even_source = REPO_ROOT / "tables" / "energy_break_even_sweep.csv"
    sensitivity_source = REPO_ROOT / "tables" / "energy_sensitivity_rank.csv"
    fig6_source = REPO_ROOT / "tables" / "fig6_evidence_repaired.csv"
    component_source = _first_existing(
        [
            REPO_ROOT / "tables" / "hpat_energy_by_component_trace_driven.csv",
            REPO_ROOT / "tables" / "hpat_energy_by_component.csv",
        ]
    )
    sources = [fig6_source, component_source]
    if break_even_source.exists() and sensitivity_source.exists():
        break_even_rows = read_csv(break_even_source)  # 优先复用已有扫描结果
        sensitivity_rows = read_csv(sensitivity_source)
        sources.extend([break_even_source, sensitivity_source])
    else:
        # 否则现场按能耗分量与证据表重算，保证实验可复现
        break_even_rows = energy_break_even_rows(read_csv(component_source), read_csv(fig6_source))
        sensitivity_rows = energy_sensitivity_rank_rows(read_csv(component_source))
    derived_break_even = _write_source(
        "mpl_supp_energy_break_even_source.csv",
        break_even_rows,
        list(break_even_rows[0].keys()) if break_even_rows else ["status"],
    )
    derived_sensitivity = _write_source(
        "mpl_supp_energy_sensitivity_source.csv",
        sensitivity_rows,
        list(sensitivity_rows[0].keys()) if sensitivity_rows else ["status"],
    )

    model = "MobileViT-S" if any(row.get("model_variant") == "MobileViT-S" for row in break_even_rows) else break_even_rows[0]["model_variant"]
    # 只保留"其它变量全部取名义值、8bit、权重常驻"的干净子集，便于做二维扫描热力图
    heat_rows = [
        row
        for row in break_even_rows
        if row["model_variant"] == model
        and abs(_safe_float(row["memory_bus_multiplier"]) - 1.0) < 1e-9
        and abs(_safe_float(row["thermal_calibration_multiplier"]) - 1.0) < 1e-9
        and abs(_safe_float(row["digital_remainder_multiplier"]) - 1.0) < 1e-9
        and abs(_safe_float(row["calibration_overhead_multiplier"]) - 1.0) < 1e-9
        and int(float(row["precision_bits"])) == 8
        and row["weight_mode"] == "resident"
    ]
    x_vals = sorted({_safe_float(row["converter_multiplier"]) for row in heat_rows})  # 热力图横轴取值
    y_vals = sorted({_safe_float(row["optical_source_multiplier"]) for row in heat_rows})  # 热力图纵轴取值
    grid = [[0.0 for _ in x_vals] for _ in y_vals]
    for row in heat_rows:
        x_idx = x_vals.index(_safe_float(row["converter_multiplier"]))
        y_idx = y_vals.index(_safe_float(row["optical_source_multiplier"]))
        # 优先用现成的相对能耗百分比列；缺失时用场景能耗/名义能耗自行换算
        if row.get("energy_vs_reference_percent"):
            grid[y_idx][x_idx] = _safe_float(row["energy_vs_reference_percent"])
        else:
            grid[y_idx][x_idx] = 100.0 * _safe_float(row["scenario_energy_mj"]) / max(_safe_float(row["nominal_energy_mj"]), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax0, ax1 = axes
    # (a) 热力图：viridis_r 颜色越深表示能耗恶化越严重，origin="lower" 让 y 轴从小到大
    image = ax0.imshow(grid, origin="lower", cmap="viridis_r", aspect="auto")
    ax0.set_xticks(range(len(x_vals)), [f"{value:g}x" for value in x_vals])
    ax0.set_yticks(range(len(y_vals)), [f"{value:g}x" for value in y_vals])
    ax0.set_xlabel("Converter energy multiplier")
    ax0.set_ylabel("Optical-source multiplier")
    ax0.set_title(f"Break-even stress ({MODEL_LABELS.get(model, model)})")
    # 每个格子内写数值，便于精确读数
    for yi, _y in enumerate(y_vals):
        for xi, _x in enumerate(x_vals):
            ax0.text(xi, yi, f"{grid[yi][xi]:.0f}", ha="center", va="center", color="white" if grid[yi][xi] > 60 else "black", fontsize=6.8)
    cbar = fig.colorbar(image, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Energy/reference (%)")
    _panel_label(ax0, "(a)")

    # (b) 敏感度排名：取该模型内排名前 6 的因素，按波动幅度从小到大画横向条形图
    rank_rows = [
        row for row in sensitivity_rows if row["model_variant"] == model and int(float(row["rank_within_model"])) <= 6
    ]
    rank_rows = sorted(rank_rows, key=lambda row: _safe_float(row["swing_percent_of_nominal"]))
    labels = [
        row["factor"]  # 把机器可读的因子名（如 thermal_calibration_multiplier）转成图例短名
        .replace("_multiplier", "")
        .replace("_overhead", "")
        .replace("thermal_calibration", "thermal/calib.")
        .replace("digital_remainder", "digital rem.")
        .replace("optical_source", "optical src.")
        .replace("memory_bus", "mem/bus")
        .replace("precision_bits", "precision")
        .replace("weight_mode", "weight mode")
        for row in rank_rows
    ]
    values = [_safe_float(row["swing_percent_of_nominal"]) for row in rank_rows]
    ax1.barh(range(len(rank_rows)), values, color=COLORS["blue"])
    ax1.set_yticks(range(len(rank_rows)), labels)
    ax1.set_xlabel("Energy swing (% of nominal)")
    ax1.set_title("Sensitivity rank")
    _panel_label(ax1, "(b)")

    fig.suptitle("Evidence strengthening: modelled energy break-even and sensitivity", y=1.02, fontsize=10)
    png = _save(fig, _path("mpl_supp_energy_break_even_sensitivity.png"))
    return (
        png,
        sources
        + [derived_break_even, derived_sensitivity]
        + _existing(EVIDENCE_SOURCES + [REPO_ROOT / "tables" / "e_local_readiness_summary.json"]),
        "evidence-strengthening modelled break-even/sensitivity; not measured deployment energy",
        "Matplotlib-rendered supplemental/rebuttal-reserve modelled energy break-even stress and sensitivity rank. Reference context uses explicit evidence labels; this is not measured HPAT deployment energy.",
    )


def _energy_breakdown() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染证据加强用的 HPAT 组件能耗堆叠分解图（单张图）。

    与 Fig. 6 (a) 类似，但只画一张独立大图：按组件（光源、DAC、MRR 调谐等）
    逐层堆叠三个模型的能耗。若数据来自模拟器导出（simulator_export）且未做
    归一化缩放，则标注为"由模拟活动计数与显式单位成本建模"；否则标注为
    "建模/归一化的能耗分配"。本图不涉及实测硅片能耗。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = _resolved_energy_component_source()
    rows = read_csv(source)
    # 判断数据是否为"模拟器导出 + 无归一化缩放"，决定图注用哪种措辞
    calibrated = any(r.get("activity_source") == "simulator_export" for r in rows) and all(
        _safe_float(r.get("normalization_factor") or 1.0) == 1.0 for r in rows
    )
    components = []
    for r in rows:
        if r["component_group"] not in components:
            components.append(r["component_group"])  # 保持组件出现顺序
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bottoms = [0.0, 0.0, 0.0]
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"], COLORS["teal"], COLORS["gold"], COLORS["gray"], "#7AA650", "#6A74B8"]
    for idx, comp in enumerate(components):
        values = []
        for model in MODELS:
            match = [r for r in rows if r["model_variant"] == model and r["component_group"] == comp]
            values.append(_safe_float(match[0]["energy_mj"]) if match else 0.0)
        ax.bar(MODEL_SHORT, values, bottom=bottoms, label=comp, color=palette[idx % len(palette)])  # 逐层堆叠
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_ylabel("Energy (mJ)")
    ax.set_title("Evidence-strengthening modelled energy breakdown")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)  # 图例放右侧外侧避免遮挡
    png = _save(fig, _path("mpl_fig_energy_breakdown_stacked.png"))
    return (
        png,
        [source],
        "modelled from simulator activity counts and explicit unit costs"
        if calibrated
        else "modelled / normalized energy allocation",
        "Matplotlib-rendered evidence-strengthening HPAT component energy breakdown from simulator activity and explicit unit costs. Not silicon-measured energy."
        if calibrated
        else "Matplotlib-rendered evidence-strengthening HPAT component energy breakdown. Values are normalized/modelled unless HPAT simulator activity is supplied.",
    )


def _operator_coverage() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染证据加强用的算子域覆盖图：PDPU 候选 MAC 占比 vs 电子剩余占比。

    PDPU（光子点积处理单元）能加速的算子（MAC）占比越高，光子计算就越有机会；
    剩余部分（电子算力）是 Amdahl 上界的主要瓶颈。数据来自本地 torch/timm
    hook（钩子）抓取的算子级 trace 汇总，属于"有界的映射证据"，不构成端侧或
    硅片实测证据。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "mobilevit_operator_domain_summary.csv"
    rows = read_csv(source)
    pdpu = []  # 可映射到 PDPU 的 MAC 占比
    elec = []  # 必须留在电子域的 MAC 占比
    for model in MODELS:
        match = [r for r in rows if r["model"] == model][0]
        pdpu.append(_safe_float(match["pdpu_candidate_mac_share_percent"]))
        elec.append(_safe_float(match["electronic_remainder_mac_share_percent"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(MODEL_SHORT, pdpu, color=COLORS["green"], label="PDPU-candidate MAC share")
    ax.bar(MODEL_SHORT, elec, bottom=pdpu, color=COLORS["gray"], label="Electronic remainder")  # 剩余部分堆在下方
    for i, value in enumerate(pdpu):
        ax.text(i, value / 2, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=8)  # 柱内标数值
    ax.set_ylim(0, 100)
    ax.set_ylabel("Traced MAC share (%)")
    ax.set_title("Evidence-strengthening operator-domain coverage")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    png = _save(fig, _path("mpl_fig_operator_domain_coverage.png"))
    return (
        png,
        [source],
        "local torch/timm hook trace summary; not edge evidence",
        "Matplotlib-rendered evidence-strengthening operator coverage summary from local torch/timm hooks. It supports bounded mapping evidence, not edge or silicon claims.",
    )


def _speedup_bound() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染证据加强用的端到端加速比上界曲线（阿姆达尔定律风格）。

    横轴是"光学映射子图加速假设"（取 log2），纵轴是整模型加速比上界。
    由于电子剩余部分不能被光学部分替代，曲线必然随假设增大而趋于饱和，
    直观展示 HPAT 加速比的真实上限。数据来自本地 hook trace 推导，不是实测加速比。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "e2e_speedup_bound.csv"
    rows = read_csv(source)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for model, color in zip(MODELS, [COLORS["blue"], COLORS["orange"], COLORS["green"]]):
        vals = [r for r in rows if r["model"] == model]
        x = [math.log2(_safe_float(r["optical_speedup_assumption"])) for r in vals]  # 加速假设转 log2 刻度
        y = [_safe_float(r["e2e_speedup_upper_bound"]) for r in vals]
        ax.plot(x, y, marker="o", label=MODEL_LABELS[model], color=color, linewidth=1.8, markersize=4)
    ax.set_xlabel("log2(optical speedup assumption)")
    ax.set_ylabel("E2E speedup upper bound")
    ax.set_title("Evidence-strengthening Amdahl-style speedup bound")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_e2e_speedup_bound.png"))
    return (
        png,
        [source],
        "Amdahl-style bound from local hook trace",
        "Matplotlib-rendered evidence-strengthening Amdahl-style upper bound from traced electronic remainder. This is not measured HPAT speedup.",
    )


def _qkv_traffic() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染证据加强用的 Q/K/V 数据搬移流量（traffic）堆叠柱状图。

    自注意力中的 Q/K/V 投影、注意力打分、Value 乘加等环节会产生大量片上数据
    搬移流量，这是光处理器能否带来收益的关键权衡点。图里按输入、Q/K/V 输出、
    注意力、Value、电子剩余、权重流、编程、总线等 8 类流量逐层堆叠，横轴是
    "模型 x 权重模式（常驻/流式/重编程）"组合，固定 8-bit、196 token 序列。
    数据来自本地 trace 校准，属于 P1（论文第一阶段）本地/建模证据，不代表实测
    访存能耗。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = _first_existing(
        [REPO_ROOT / "tables" / "qkv_traffic_calibrated.csv", REPO_ROOT / "tables" / "qkv_traffic_sensitivity.csv"]
    )
    # 固定 8bit 量化、196 token 的标准设定，保证各组数据可比
    rows = [r for r in read_csv(source) if int(float(r["bit_width"])) == 8 and int(float(r["n_tokens"])) == 196]
    # 每类流量：(图例名, 字节数字段, 兼容旧文件的 bit 数字段)
    components = [
        ("Input", "activation_input_bytes", "input_bits"),
        ("Q/K/V output", "qkv_projection_output_bytes", "qkv_output_bits"),
        ("Attention", "attention_score_bytes", ""),
        ("Value", "value_product_bytes", ""),
        ("Electronic", "electronic_remainder_bytes", ""),
        ("Weight stream", "weight_stream_bytes", "weight_stream_bits"),
        ("Programming", "programming_bytes", "programming_bits"),
        ("Bus", "bus_bytes", "bus_bits"),
    ]
    cases = []  # 三个模型 x 三种权重模式，共 9 组柱子
    for model in MODELS:
        for mode in ["resident", "streamed", "reprogrammed"]:
            cases.append((model, mode))
    labels = [f"{MODEL_LABELS[m]}\n{mode}" for m, mode in cases]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    bottoms = [0.0 for _ in cases]
    colors = [COLORS["blue"], COLORS["green"], COLORS["teal"], COLORS["purple"], COLORS["gray"], COLORS["orange"], COLORS["red"], COLORS["gold"]]
    for idx, (label, byte_key, legacy_bits_key) in enumerate(components):
        values = []
        for model, mode in cases:
            match = [r for r in rows if r["variant"] == model and r["weight_mode"] == mode][0]
            if byte_key in match:
                values.append(_safe_float(match[byte_key]) / 1024.0)  # 字节 -> KiB
            elif legacy_bits_key and legacy_bits_key in match:
                values.append(_safe_float(match[legacy_bits_key]) / 8.0 / 1024.0)  # 兼容旧版本 bit 单位
            else:
                values.append(0.0)
        ax.bar(range(len(cases)), values, bottom=bottoms, color=colors[idx], label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]  # 累计高度实现堆叠
    ax.set_xticks(range(len(cases)), labels)
    ax.set_ylabel("Traffic (KiB)")
    ax.set_title("Evidence-strengthening Q/K/V traffic at 8-bit, N=196")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    # 把每组的流量总量写盘，方便后续引用/核对
    derived = _write_source(
        "mpl_fig_qkv_traffic_sensitivity.csv",
        [{"case": label, "total_kib": f"{total:.6f}"} for label, total in zip(labels, bottoms)],
        ["case", "total_kib"],
    )
    png = _save(fig, _path("mpl_fig_qkv_traffic_sensitivity.png"))
    return (
        png,
        [source, derived]
        + _existing(
            [
                REPO_ROOT / "tables" / "mobilevit_operator_activity.csv",
                REPO_ROOT / "tables" / "p1_readiness_summary.json",
            ]
        ),
        "local/modelled P1 Q/K/V traffic calibration",
        "Matplotlib-rendered evidence-strengthening Q/K/V traffic calibration. Traffic is local/modelled and not measured memory energy.",
    )


def _nonideality() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染证据加强用的非理想性敏感度横向条形图。

    逐项扫描光子处理器的非理想效应（PD/TIA 噪声、WDM 相邻串扰、转换器量化、
    插入损耗、波长失谐、MRR 工艺偏差、热漂移），每个效应取扫描中最坏的影响值：
    若有带标注子集则画 top-1 精度损失（%），否则画平均相对误差。展示哪些
    非理想效应对精度/误差影响最大。这是建模/分析敏感度，不是硅片鲁棒性或
    ImageNet 完整精度证据。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_accuracy_sweep.csv"
    # 把机器可读的非理想性效应名映射为图上短标签
    short = {
        "gaussian_pd_tia_noise": "PD/TIA noise",
        "wdm_adjacent_crosstalk": "WDM crosstalk",
        "uniform_converter_quantization": "Quantization",
        "insertion_loss": "Insertion loss",
        "wavelength_detuning": "Detuning",
        "mrr_variation": "MRR variation",
        "thermal_drift": "Thermal drift",
    }
    source_rows = read_csv(source)
    # 若扫描带有标注子集上的 top-1 精度损失，优先画精度损失；否则退回画平均相对误差
    use_accuracy = any(int(float(r.get("label_count") or 0)) > 0 and r.get("top1_delta") not in ("", None) for r in source_rows)
    buckets: dict[tuple[str, str], float] = defaultdict(float)
    for r in source_rows:
        key = (short.get(r["effect"], r["effect"]), r["variant"])
        value = abs(_safe_float(r.get("top1_delta"))) if use_accuracy else _safe_float(r["mean_relative_error"])
        buckets[key] = max(buckets[key], value)  # 取各效应在扫描中最大的（最坏）影响
    effects = list(dict.fromkeys(k[0] for k in buckets.keys()))  # 保持效应出现顺序去重
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = list(range(len(effects)))
    height = 0.23
    # 横向条形图：每个效应画三个模型的三条细条
    for idx, model in enumerate(MODELS):
        values = [buckets.get((effect, model), 0.0) for effect in effects]
        ypos = [v + (idx - 1) * height for v in y]  # 三条条错开
        ax.barh(ypos, values, height=height, label=MODEL_LABELS[model], color=[COLORS["blue"], COLORS["orange"], COLORS["green"]][idx])
    ax.set_yticks(y, effects)
    ax.set_xlabel("Worst top-1 accuracy drop (%)" if use_accuracy else "Worst mean relative error across sweep")
    ax.set_title("Evidence-strengthening selected non-ideality sensitivity" if use_accuracy else "Evidence-strengthening non-ideality sensitivity proxy")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_nonideality_sensitivity.png"))
    # 把最坏影响值导出为 CSV，作为该图的派生数据源
    derived = _write_source(
        "mpl_fig_nonideality_sensitivity.csv",
        [
            {
                "effect": effect,
                "model": model,
                "top1_accuracy_drop_percent" if use_accuracy else "max_mean_relative_error": f"{buckets.get((effect, model), 0.0):.8f}",
            }
            for effect in effects
            for model in MODELS
        ],
        ["effect", "model", "top1_accuracy_drop_percent" if use_accuracy else "max_mean_relative_error"],
    )
    return (
        png,
        [source, derived],
        "dataset-coupled MobileViT accuracy subset" if use_accuracy else "modelled selected non-ideality / analytical sensitivity",
        "Matplotlib-rendered evidence-strengthening top-1 accuracy drop on a fixed labeled subset. It is not silicon robustness."
        if use_accuracy
        else "Matplotlib-rendered evidence-strengthening non-ideality sensitivity proxy. It is not silicon robustness or ImageNet accuracy evidence.",
    )


def _fixed_subset_dose_response() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 8：固定子集（Imagenette 校验集）上的"剂量-响应"曲线图。

    2x2 四个子图，分别对应四种低风险可解释扰动：PD/TIA 噪声、WDM 串扰、
    MRR 工艺偏差、热漂移。横轴是扰动量（剂量），纵轴是 top-1 精度下降百分比；
    每个模型一条曲线，若存在多随机种子结果则画出 p05-p95 阴影包络。
    红色虚线标出 5% 阈值，直观判断哪些扰动在多大剂量下会明显掉点。
    数据来自 Imagenette 外部公开校验子集，不是完整 ImageNet 或设备/硅片鲁棒性证据。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_accuracy_by_severity.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)  # 缺数据直接报错，避免画空图
    rows = read_csv(source)
    if not rows:
        raise ValueError("nonideality_accuracy_by_severity.csv has no rows")
    selected = {
        "gaussian_pd_tia_noise": "PD/TIA noise",
        "wdm_adjacent_crosstalk": "WDM crosstalk",
        "mrr_variation": "MRR variation",
        "thermal_drift": "Thermal drift",
    }
    xlabels = {
        "gaussian_pd_tia_noise": "noise (LSB)",
        "wdm_adjacent_crosstalk": "crosstalk alpha",
        "mrr_variation": "MRR sigma (%)",
        "thermal_drift": "delta T (C)",
    }
    xticks = {  # 每个效应的固定横轴刻度位置，保证子图间可读性
        "gaussian_pd_tia_noise": [0.0, 0.25, 0.5, 1.0],
        "wdm_adjacent_crosstalk": [0.0, 0.01, 0.03, 0.05],
        "mrr_variation": [0.0, 1.0, 3.0, 5.0],
        "thermal_drift": [0.0, 2.0, 5.0, 10.0],
    }
    # 只画数据中实际存在的效应子图
    effects = [effect for effect in selected if any(row["effect"] == effect for row in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.85), squeeze=False, sharey=True)
    derived_rows: list[dict[str, Any]] = []
    for effect_index, effect in enumerate(effects):
        ax = axes[effect_index // 2][effect_index % 2]  # 按行优先填充 2x2 子图
        effect_rows = [row for row in rows if row["effect"] == effect]
        variable = effect_rows[0]["sweep_variable"]
        for model, color in zip(MODELS, MODEL_COLORS):
            vals = sorted(
                [row for row in effect_rows if row["variant"] == model],
                key=lambda row: _safe_float(row["sweep_value"]),  # 按扰动量从小到大画线
            )
            if not vals:
                continue
            x = [_safe_float(row["sweep_value"]) for row in vals]
            y = [_safe_float(row["top1_drop_mean"]) for row in vals]
            lo = [_safe_float(row["top1_drop_p05"]) for row in vals]
            hi = [_safe_float(row["top1_drop_p95"]) for row in vals]
            ax.plot(x, y, marker="o", color=color, label=MODEL_LABELS[model])
            # p05/p95 不同时才画半透明包络（多随机种子不确定性带）
            if any(abs(a - b) > 1e-9 for a, b in zip(lo, hi)):
                ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
            for row, yy in zip(vals, y):
                derived_rows.append(
                    {
                        "effect": effect,
                        "sweep_variable": variable,
                        "sweep_value": row["sweep_value"],
                        "model": model,
                        "top1_drop_mean": f"{yy:.6f}",
                        "top1_drop_p05": row.get("top1_drop_p05", ""),
                        "top1_drop_p95": row.get("top1_drop_p95", ""),
                        "trial_count": row.get("trial_count", ""),
                        "evidence_label": row.get("evidence_label", ""),
                    }
                )
        ax.set_title(selected.get(effect, effect))
        ax.set_xlabel(xlabels.get(effect, variable.replace("_", " ")))
        if effect_index % 2 == 0:
            ax.set_ylabel("Top-1 drop (%)")
        ax.set_xticks(xticks.get(effect, []))
        ax.set_ylim(-0.35, 10.25)
        ax.axhline(5.0, color=COLORS["red"], linewidth=0.8, linestyle="--", alpha=0.75)  # 5% 掉点阈值参考线
        if effect_index == 0:
            ax.text(
                0.98,
                5.15,
                "5% threshold",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=COLORS["red"],
            )
    for empty_index in range(len(effects), 4):
        axes[empty_index // 2][empty_index % 2].axis("off")  # 没有数据的子图直接关掉
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Imagenette fixed-subset HPAT-mapping sensitivity", y=0.99, fontsize=10)
    fig._hpat_tight_rect = (0.0, 0.10, 1.0, 0.94)  # 底部给整图图例留空间
    fig._hpat_tight_kwargs = {"h_pad": 2.0}
    # 把每条曲线的点导出为 CSV，作为 Fig. 8 的派生数据源
    derived = _write_source(
        "mpl_fig8_fixed_subset_dose_response.csv",
        derived_rows,
        [
            "effect",
            "sweep_variable",
            "sweep_value",
            "model",
            "top1_drop_mean",
            "top1_drop_p05",
            "top1_drop_p95",
            "trial_count",
            "evidence_label",
        ],
    )
    png = _save(fig, _path("mpl_fig8_fixed_subset_dose_response.png"))
    return (
        png,
        [source, derived]
        + _existing(
            [
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid.csv",
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid_source_ledger.csv",
                REPO_ROOT / "tables" / "e_local_readiness_summary.json",
            ]
        ),
        "Imagenette validation external-public fixed-subset robustness; not full ImageNet or device/silicon robustness",
        "Matplotlib-rendered Fig. 8: selected HPAT-mapping sensitivity on the Imagenette validation external-public fixed subset. It shows top-1 drop for low-risk interpreted perturbations, with stochastic seed envelopes where applicable; not full ImageNet validation, not silicon/device robustness, and not measured edge/mobile device robustness.",
    )


def _supp_fixed_subset_stress_modes() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染补充图：固定子集上的三种"压力/失效模式"诊断曲线。

    1x3 三个子图，对应三种更苛刻的失效模式：
    - 单位范围截断 + 量化（横轴是 ADC/DAC 位数，从左到右位数递减）；
    - 未补偿的插入损耗压力（横轴是路径损耗 dB）；
    - 未补偿的波长失谐压力（横轴是波长偏移 pm）。
    每个子图画三个模型的 top-1 精度下降随剂量变化的曲线（带 p05-p95 包络），
    并画 5% 阈值线。注意这些诊断不代表标定后的量化鲁棒性，也不代表完整
    ImageNet 或设备/硅片鲁棒性。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_accuracy_by_severity.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    rows = read_csv(source)
    if not rows:
        raise ValueError("nonideality_accuracy_by_severity.csv has no rows")
    stress = {
        "uniform_converter_quantization": "Unit-range clipping +\nquantization stress",
        "insertion_loss": "Uncompensated\ninsertion-loss stress",
        "wavelength_detuning": "Uncompensated\nwavelength-detuning stress",
    }
    xlabels = {
        "uniform_converter_quantization": "ADC/DAC bits",
        "insertion_loss": "path loss (dB)",
        "wavelength_detuning": "delta lambda (pm)",
    }
    xticks = {
        "uniform_converter_quantization": [12.0, 10.0, 8.0, 6.0, 4.0],
        "insertion_loss": [0.0, 1.0, 2.0, 3.0],
        "wavelength_detuning": [0.0, 5.0, 10.0, 20.0],
    }
    effects = [effect for effect in stress if any(row["effect"] == effect for row in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), squeeze=False)
    derived_rows: list[dict[str, Any]] = []
    for effect_index, effect in enumerate(effects):
        ax = axes[0][effect_index]
        effect_rows = [row for row in rows if row["effect"] == effect]
        variable = effect_rows[0]["sweep_variable"]
        for model, color in zip(MODELS, MODEL_COLORS):
            vals = sorted(
                [row for row in effect_rows if row["variant"] == model],
                key=lambda row: _safe_float(row["sweep_value"]),
                reverse=effect == "uniform_converter_quantization",  # 量化位数越大越好，排序取反
            )
            if not vals:
                continue
            x = [_safe_float(row["sweep_value"]) for row in vals]
            y = [_safe_float(row["top1_drop_mean"]) for row in vals]
            lo = [_safe_float(row["top1_drop_p05"]) for row in vals]
            hi = [_safe_float(row["top1_drop_p95"]) for row in vals]
            ax.plot(x, y, marker="o", color=color, label=MODEL_LABELS[model])
            if any(abs(a - b) > 1e-9 for a, b in zip(lo, hi)):
                ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)  # 随机种子不确定度包络
            for row, yy in zip(vals, y):
                derived_rows.append(
                    {
                        "effect": effect,
                        "sweep_variable": variable,
                        "sweep_value": row["sweep_value"],
                        "model": model,
                        "top1_drop_mean": f"{yy:.6f}",
                        "top1_drop_p05": row.get("top1_drop_p05", ""),
                        "top1_drop_p95": row.get("top1_drop_p95", ""),
                        "trial_count": row.get("trial_count", ""),
                        "evidence_label": row.get("evidence_label", ""),
                    }
                )
        ax.set_title(stress.get(effect, effect))
        ax.set_xlabel(xlabels.get(effect, variable.replace("_", " ")))
        ax.set_ylabel("Top-1 drop (%)" if effect_index == 0 else "")
        ax.set_xticks(xticks.get(effect, []))
        if effect == "uniform_converter_quantization":
            ax.invert_xaxis()  # 量化子图让"位数越多"在右侧，语义更直观
        ax.axhline(5.0, color=COLORS["red"], linewidth=0.8, linestyle="--", alpha=0.75)  # 5% 阈值参考线
    for empty_index in range(len(effects), 3):
        axes[0][empty_index].axis("off")  # 无数据子图直接关闭
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Supplemental fixed-subset stress and failure-mode diagnostics", y=0.99, fontsize=10)
    fig._hpat_tight_rect = (0.0, 0.18, 1.0, 0.90)  # 底部为整图图例预留空间
    derived = _write_source(
        "mpl_supp_fixed_subset_stress_modes.csv",
        derived_rows,
        [
            "effect",
            "sweep_variable",
            "sweep_value",
            "model",
            "top1_drop_mean",
            "top1_drop_p05",
            "top1_drop_p95",
            "trial_count",
            "evidence_label",
        ],
    )
    png = _save(fig, _path("mpl_supp_fixed_subset_stress_modes.png"))
    return (
        png,
        [source, derived]
        + _existing(
            [
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid.csv",
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid_source_ledger.csv",
                REPO_ROOT / "tables" / "e_local_readiness_summary.json",
            ]
        ),
        "fixed-subset stress/failure-mode diagnostics; not full ImageNet or device/silicon robustness",
        "Matplotlib-rendered supplemental fixed-subset stress modes: unit-range clipping plus quantization, uncompensated insertion loss, and uncompensated wavelength detuning. These diagnostics are not calibrated converter quantization robustness, not full ImageNet validation, not silicon/device robustness, and not measured edge/mobile robustness.",
    )


def _fixed_subset_safe_region() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 9：固定子集上的"安全区间"热力图。

    横轴是七种非理想性效应，纵轴是三个模型；每个格子颜色表示该模型在
    该效应的扫描网格中"安全区间"覆盖的比例（0~1，YlGnBu 颜色映射），
    格子内文字直接写出安全阈值（如 "<=5"、">=8b"、"none"、"n/a"）。
    安全区间指在 5% 掉点阈值内仍保持可接受精度的扰动范围。颜色只能在同一
    效应的网格内比较，不能跨效应比大小；且不是完整 ImageNet 或设备/硅片
    鲁棒性证据。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    safe_source = REPO_ROOT / "tables" / "nonideality_accuracy_safe_region.csv"
    severity_source = REPO_ROOT / "tables" / "nonideality_accuracy_by_severity.csv"
    if not safe_source.exists() or safe_source.stat().st_size == 0:
        raise FileNotFoundError(safe_source)
    safe_rows = read_csv(safe_source)
    severity_rows = read_csv(severity_source) if severity_source.exists() else []
    if not safe_rows:
        raise ValueError("nonideality_accuracy_safe_region.csv has no rows")
    # 固定七种效应的展示顺序，保证图与正文一致
    effect_order = [
        "gaussian_pd_tia_noise",
        "wdm_adjacent_crosstalk",
        "uniform_converter_quantization",
        "insertion_loss",
        "wavelength_detuning",
        "mrr_variation",
        "thermal_drift",
    ]
    effect_order = [effect for effect in effect_order if any(row["effect"] == effect for row in safe_rows)]
    short = {
        "gaussian_pd_tia_noise": "PD/TIA\nnoise",
        "wdm_adjacent_crosstalk": "WDM\ncrosstalk",
        "uniform_converter_quantization": "Quant.\nbits",
        "insertion_loss": "Insertion\nloss",
        "wavelength_detuning": "Wave.\ndetune",
        "mrr_variation": "MRR\nvar.",
        "thermal_drift": "Thermal\ndrift",
    }
    max_index: dict[tuple[str, str], int] = {}
    safe_index: dict[tuple[str, str], float] = {}
    for model in MODELS:
        for effect in effect_order:
            # 该模型在该效应下的全部扫描值，作为安全网格的分母
            values = sorted({_safe_float(row["sweep_value"]) for row in severity_rows if row["variant"] == model and row["effect"] == effect})
            if effect == "uniform_converter_quantization":
                values = sorted(values, reverse=True)  # 位数越大越安全，排序方向取反
            max_index[(model, effect)] = max(len(values) - 1, 1)
            safe_value_rows = [row for row in safe_rows if row["variant"] == model and row["effect"] == effect]
            if not safe_value_rows or safe_value_rows[0].get("safe_sweep_value") in ("", None):
                safe_index[(model, effect)] = 0.0  # 没有安全阈值说明该组合最差，覆盖度为 0
            else:
                safe_value = _safe_float(safe_value_rows[0]["safe_sweep_value"])
                try:
                    idx = values.index(safe_value)  # 安全阈值在扫描网格中的位置
                except ValueError:
                    idx = 0
                safe_index[(model, effect)] = idx / max_index[(model, effect)]  # 归一化为覆盖比例

    matrix = [[safe_index.get((model, effect), 0.0) for effect in effect_order] for model in MODELS]
    fig, ax = plt.subplots(figsize=(7.16, 2.65))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")  # 越蓝覆盖越高
    ax.set_xticks(range(len(effect_order)), [short.get(effect, effect) for effect in effect_order])
    ax.set_yticks(range(len(MODELS)), MODEL_SHORT)
    ax.set_title("Imagenette fixed-subset safe region within sweep grid")
    derived_rows: list[dict[str, Any]] = []
    for y, model in enumerate(MODELS):
        for x, effect in enumerate(effect_order):
            row = next((r for r in safe_rows if r["variant"] == model and r["effect"] == effect), None)
            if row is None:
                label = "n/a"  # 数据缺失
                safe_value = ""
                status = "missing"
            else:
                safe_value = row.get("safe_sweep_value", "")
                status = row.get("safe_status", "")
                if safe_value in ("", None):
                    label = "none"  # 网格内不存在安全区间
                elif effect == "uniform_converter_quantization":
                    label = f">={safe_value}b"  # 位数大于等于该值时安全
                else:
                    label = f"<={safe_value}"  # 扰动量小于等于该值时安全
            ax.text(x, y, label, ha="center", va="center", fontsize=7.0, color="#0B1F2A")  # 格子内写安全阈值
            derived_rows.append(
                {
                    "model": model,
                    "effect": effect,
                    "safe_sweep_value": safe_value,
                    "safe_grid_fraction": f"{safe_index.get((model, effect), 0.0):.6f}",
                    "safe_status": status,
                    "evidence_label": row.get("evidence_label", "") if row else "",
                }
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Safe grid coverage")
    ax.set_xlabel("Non-ideality effect")
    ax.set_ylabel("Model")
    derived = _write_source(
        "mpl_fig9_fixed_subset_safe_region.csv",
        derived_rows,
        ["model", "effect", "safe_sweep_value", "safe_grid_fraction", "safe_status", "evidence_label"],
    )
    png = _save(fig, _path("mpl_fig9_fixed_subset_safe_region.png"))
    return (
        png,
        [safe_source, severity_source, derived]
        + _existing(
            [
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid.csv",
                REPO_ROOT / "tables" / "fixed_subset_imagenette160_valid_source_ledger.csv",
                REPO_ROOT / "tables" / "nonideality_accuracy_failure_thresholds.csv",
                REPO_ROOT / "tables" / "e_local_readiness_summary.json",
            ]
        ),
        "Imagenette validation external-public fixed-subset safe-region summary; not full ImageNet or device/silicon robustness",
        "Matplotlib-rendered Fig. 9: safe severity region on the Imagenette validation external-public fixed subset. Color denotes safe grid coverage within the declared sweep, not cross-effect unit comparability; not full ImageNet validation, silicon robustness, or measured device robustness.",
    )


def _fixed_subset_worst_class_drop() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    """渲染论文 Fig. 10：固定子集上"最差类别"的精度掉点柱状图。

    从各类别级别的鲁棒性统计中挑出 top-1 掉点最严重的 12 个（模型 x 效应 x 类别）
    组合，按掉点幅度从大到小排列画柱状图，柱子颜色对应所属模型，并画 5% 阈值线，
    用于定位最脆弱的类别与扰动组合。数据来自 Imagenette 外部公开校验子集，
    不是完整 ImageNet 或设备/硅片鲁棒性证据。

    返回：(png 路径, 数据源路径列表, 证据标签, 图注) 四元组。
    """
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_accuracy_worst_class.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    rows = read_csv(source)
    if not rows:
        raise ValueError("nonideality_accuracy_worst_class.csv has no rows")
    # 按最差类别掉点均值降序取前 12 名（最脆弱的组合）
    rows = sorted(rows, key=lambda row: _safe_float(row.get("worst_top1_drop_mean")), reverse=True)[:12]
    # 横轴标签：模型短名 + 效应名 + 最差类别名，三行显示
    labels = [
        f"{row['variant'].replace('MobileViT-', '')}\n{row['effect'].replace('_', ' ')}\n{row.get('worst_class_name', '')}"
        for row in rows
    ]
    values = [_safe_float(row.get("worst_top1_drop_mean")) for row in rows]
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    # 柱子颜色按模型着色，便于区分
    colors = [MODEL_COLORS[MODELS.index(row["variant"])] if row["variant"] in MODELS else COLORS["gray"] for row in rows]
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("Worst class top-1 drop (%)")
    ax.set_title("Imagenette fixed-subset worst-class stress cases")
    ax.axhline(5.0, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.75)  # 5% 阈值线
    derived = _write_source(
        "mpl_fig10_fixed_subset_worst_class_drop.csv",
        rows,
        list(rows[0].keys()),
    )
    png = _save(fig, _path("mpl_fig10_fixed_subset_worst_class_drop.png"))
    return (
        png,
        [source, derived] + _existing([REPO_ROOT / "tables" / "nonideality_accuracy_classwise.csv"]),
        "Imagenette validation external-public fixed-subset class-wise robustness; not full ImageNet or device/silicon robustness",
        "Matplotlib-rendered fixed-subset worst-class drop figure. It is derived from Imagenette validation external-public subset statistics and is not full ImageNet validation, silicon robustness, or measured edge/mobile robustness.",
    )


def _fixed_subset_margin_drift() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_margin_drift_by_severity.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    rows = read_csv(source)
    if not rows:
        raise ValueError("nonideality_margin_drift_by_severity.csv has no rows")
    effects = [
        "gaussian_pd_tia_noise",
        "wdm_adjacent_crosstalk",
        "uniform_converter_quantization",
        "insertion_loss",
        "wavelength_detuning",
        "mrr_variation",
        "thermal_drift",
    ]
    effects = [effect for effect in effects if any(row["effect"] == effect for row in rows)]
    ncols = 4
    nrows = math.ceil(len(effects) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.55 * nrows + 0.35), squeeze=False)
    derived_rows: list[dict[str, Any]] = []
    for effect_index, effect in enumerate(effects):
        ax = axes[effect_index // ncols][effect_index % ncols]
        effect_rows = [row for row in rows if row["effect"] == effect]
        variable = effect_rows[0]["sweep_variable"]
        for model, color in zip(MODELS, MODEL_COLORS):
            vals = sorted(
                [row for row in effect_rows if row["variant"] == model],
                key=lambda row: _safe_float(row["sweep_value"]),
                reverse=effect == "uniform_converter_quantization",
            )
            if not vals:
                continue
            x = [_safe_float(row["sweep_value"]) for row in vals]
            y = [_safe_float(row["margin_delta_mean"]) for row in vals]
            ax.plot(x, y, marker="o", color=color, label=MODEL_LABELS[model])
            for row, yy in zip(vals, y):
                derived_rows.append(
                    {
                        "effect": effect,
                        "sweep_variable": variable,
                        "sweep_value": row["sweep_value"],
                        "model": model,
                        "margin_delta_mean": f"{yy:.8f}",
                        "margin_delta_p05": row.get("margin_delta_p05", ""),
                        "margin_delta_p95": row.get("margin_delta_p95", ""),
                        "top5_jaccard_mean": row.get("top5_jaccard_mean", ""),
                    }
                )
        ax.set_title(effect.replace("_", " "))
        ax.set_xlabel(variable.replace("_", " "))
        ax.set_ylabel("Margin drift")
        ax.axhline(0.0, color=COLORS["darkgray"], linewidth=0.7)
        if effect == "uniform_converter_quantization":
            ax.invert_xaxis()
    for empty_index in range(len(effects), nrows * ncols):
        axes[empty_index // ncols][empty_index % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Imagenette fixed-subset margin drift", y=1.02, fontsize=10)
    derived = _write_source(
        "mpl_fig11_fixed_subset_margin_drift.csv",
        derived_rows,
        [
            "effect",
            "sweep_variable",
            "sweep_value",
            "model",
            "margin_delta_mean",
            "margin_delta_p05",
            "margin_delta_p95",
            "top5_jaccard_mean",
        ],
    )
    png = _save(fig, _path("mpl_fig11_fixed_subset_margin_drift.png"))
    return (
        png,
        [source, derived] + _existing([REPO_ROOT / "tables" / "nonideality_accuracy_bootstrap_ci.csv"]),
        "Imagenette validation external-public fixed-subset confidence/margin sensitivity; not full ImageNet or silicon robustness",
        "Matplotlib-rendered fixed-subset margin drift figure. It supports sensitivity interpretation only and is not full ImageNet validation, measured device robustness, or silicon validation.",
    )


def _boundary_ablation() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "nonideality_boundary_ablation_summary.csv"
    delta_source = REPO_ROOT / "tables" / "nonideality_boundary_ablation_delta.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    rows = read_csv(source)
    if not rows:
        raise ValueError("nonideality_boundary_ablation_summary.csv has no rows")
    selected = [row for row in rows if row["effect"] in {"gaussian_pd_tia_noise", "mrr_variation", "uniform_converter_quantization"}]
    selected = selected[:]
    cases = []
    for row in selected:
        key = (row["variant"], row["effect"], row["sweep_value"])
        if key not in cases:
            cases.append(key)
    cases = cases[:12]
    labels = [f"{model.replace('MobileViT-', '')}\n{effect.replace('_', ' ')}\n{value}" for model, effect, value in cases]
    hpat = []
    all_linear = []
    for model, effect, value in cases:
        hpat_row = next(
            (
                row
                for row in rows
                if row["boundary"] == "hpat-mapping"
                and row["variant"] == model
                and row["effect"] == effect
                and row["sweep_value"] == value
            ),
            {},
        )
        all_row = next(
            (
                row
                for row in rows
                if row["boundary"] == "all-linear-smoke"
                and row["variant"] == model
                and row["effect"] == effect
                and row["sweep_value"] == value
            ),
            {},
        )
        hpat.append(_safe_float(hpat_row.get("top1_drop_mean")))
        all_linear.append(_safe_float(all_row.get("top1_drop_mean")))
    x = list(range(len(cases)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    ax.bar([v - width / 2 for v in x], hpat, width=width, color=COLORS["green"], label="HPAT mapping")
    ax.bar([v + width / 2 for v in x], all_linear, width=width, color=COLORS["gray"], label="All-linear diagnostic")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Top-1 drop (%)")
    ax.set_title("Boundary ablation on fixed-subset non-ideality injection")
    ax.legend(frameon=False)
    derived_rows = [
        {
            "case": label,
            "hpat_top1_drop": f"{h:.6f}",
            "all_linear_top1_drop": f"{a:.6f}",
            "diagnostic_delta": f"{a - h:.6f}",
        }
        for label, h, a in zip(labels, hpat, all_linear)
    ]
    derived = _write_source(
        "mpl_fig12_boundary_ablation.csv",
        derived_rows,
        ["case", "hpat_top1_drop", "all_linear_top1_drop", "diagnostic_delta"],
    )
    png = _save(fig, _path("mpl_fig12_boundary_ablation.png"))
    sources = [source, derived] + _existing([delta_source])
    return (
        png,
        sources,
        "fixed-subset boundary diagnostic; all-linear is not HPAT-mapping claim evidence",
        "Matplotlib-rendered boundary ablation figure. HPAT-mapping rows retain fixed-subset limitations; all-linear rows are diagnostic only and do not support HPAT-mapped execution claims.",
    )


def _scalability() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = _first_existing(
        [REPO_ROOT / "tables" / "scalability_physical_proxy.csv", REPO_ROOT / "tables" / "scalability_sweep.csv"]
    )
    rows = []
    for r in read_csv(source):
        adc = r.get("adc_parallelism", r.get("adc_count_proxy", "0"))
        memory_bw = r.get("memory_bandwidth_gbps", r.get("memory_bandwidth_budget_gbps", "0"))
        thermal = r.get("thermal_calibration_multiplier", "1.0")
        if (
            int(float(r["pdpu_banks"])) == 2
            and int(float(r["tiles"])) == 2
            and int(float(adc)) == 16
            and float(memory_bw) == 128.0
            and float(thermal) == 1.0
        ):
            rows.append(r)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for model, color in zip(MODELS, [COLORS["blue"], COLORS["orange"], COLORS["green"]]):
        vals = sorted([r for r in rows if r["variant"] == model], key=lambda r: _safe_float(r["wavelengths"]))
        ax.plot(
            [_safe_float(r["wavelengths"]) for r in vals],
            [_safe_float(r["latency_estimate_ns"]) for r in vals],
            marker="o",
            linewidth=1.8,
            color=color,
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("Wavelength channels")
    ax.set_ylabel("Latency estimate (ns)")
    ax.set_title("Evidence-strengthening scalability proxy")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_scalability_sweep.png"))
    return (
        png,
        [source] + _existing([REPO_ROOT / "tables" / "p1_readiness_summary.json"]),
        "local/modelled scalability physical proxy",
        "Matplotlib-rendered evidence-strengthening scalability physical proxy. It is not physical layout closure or fabricated deployment evidence.",
    )


def _architecture_ablation() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "hpat_architecture_ablation.csv"
    rows = read_csv(source)
    wanted = [
        "baseline",
        "no_broadcast",
        "reprogrammed_weights",
        "wdm_channels_8",
        "wdm_channels_32",
        "pdpu_banks_1",
        "pdpu_banks_4",
        "precision_4bit",
        "precision_12bit",
        "optimistic_optical_only",
        "calibration_interval_100",
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = list(range(len(wanted)))
    height = 0.23
    for idx, model in enumerate(MODELS):
        values = []
        for ablation in wanted:
            match = [r for r in rows if r["variant"] == model and r["ablation"] == ablation]
            values.append(100.0 + _safe_float(match[0]["delta_energy_vs_baseline_percent"]) if match else 100.0)
        ypos = [v + (idx - 1) * height for v in y]
        ax.barh(ypos, values, height=height, label=MODEL_LABELS[model], color=[COLORS["blue"], COLORS["orange"], COLORS["green"]][idx])
    ax.set_yticks(y, wanted)
    ax.set_xlabel("Energy vs baseline (%)")
    ax.set_title("Evidence-strengthening modelled architecture ablation")
    ax.axvline(100, color="#222222", linewidth=0.8)
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_architecture_ablation.png"))
    derived_rows = [
        {
            "ablation": ablation,
            "model": model,
            "energy_vs_baseline_percent": f"{100.0 + next((_safe_float(r['delta_energy_vs_baseline_percent']) for r in rows if r['variant'] == model and r['ablation'] == ablation), 0.0):.6f}",
        }
        for ablation in wanted
        for model in MODELS
    ]
    derived = _write_source(
        "mpl_fig_architecture_ablation.csv",
        derived_rows,
        ["ablation", "model", "energy_vs_baseline_percent"],
    )
    return (
        png,
        [source, derived] + _existing([REPO_ROOT / "tables" / "p1_readiness_summary.json"]),
        "local/modelled P1 architecture ablation",
        "Matplotlib-rendered evidence-strengthening HPAT architecture ablation. This is modelled/proxy evidence, not measured hardware evidence.",
    )


def _energy_uncertainty() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    component_source = _resolved_energy_component_source()
    component_rows = read_csv(component_source)
    rows, uncertainty_sources = _resolved_uncertainty_rows(component_rows)
    cases = ["p05", "nominal", "p50", "p95"]
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    x = list(range(len(models)))
    width = 0.18
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for idx, case in enumerate(cases):
        values = []
        for model in models:
            match = [r for r in rows if r["model"] == model and r["case"] == case]
            values.append(_safe_float(match[0]["energy_mj"]) if match else 0.0)
        ax.bar([v + (idx - 1.5) * width for v in x], values, width=width, label=case, color=colors[idx])
    ax.set_xticks(x, [MODEL_LABELS.get(model, model) for model in models])
    ax.set_ylabel("Energy (mJ)")
    ax.set_title("Evidence-strengthening modelled energy uncertainty")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    png = _save(fig, _path("mpl_fig_energy_uncertainty_sweep.png"))
    return (
        png,
        [component_source, *uncertainty_sources]
        + _existing(
            [
                REPO_ROOT / "tables" / "energy_unit_cost_source_ledger.csv",
                REPO_ROOT / "tables" / "p1_readiness_summary.json",
            ]
        ),
        "modelled component uncertainty sensitivity",
        "Matplotlib-rendered evidence-strengthening component uncertainty envelope around the normalized HPAT energy model. This is not calibrated silicon energy.",
    )


def _p2_layout_area() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "layout_area_feasibility_proxy.csv"
    rows = [
        r
        for r in read_csv(source)
        if int(float(r["wavelengths"])) == 16 and int(float(r["pdpu_banks"])) == 2 and int(float(r["tiles"])) == 2
    ]
    labels = [r["variant"].replace("MobileViT-", "") for r in rows]
    components = [
        ("MRR", "mrr_area_proxy_mm2", COLORS["green"]),
        ("Converters", "converter_area_proxy_mm2", COLORS["orange"]),
        ("Interconnect", "interconnect_area_proxy_mm2", COLORS["gray"]),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bottoms = [0.0 for _ in rows]
    for label, key, color in components:
        values = [_safe_float(r[key]) for r in rows]
        ax.bar(labels, values, bottom=bottoms, label=label, color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_ylabel("Area proxy (mm2)")
    ax.set_title("Evidence-strengthening layout/area feasibility proxy")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_layout_area_feasibility_proxy.png"))
    return (
        png,
        [source, *_existing([REPO_ROOT / "tables" / "p2_readiness_summary.json"])],
        "local/modelled P2 layout-area proxy",
        "Matplotlib-rendered evidence-strengthening layout/area proxy for the fixed 16-wavelength, 2-bank, 2-tile configuration. Not physical-design closure.",
    )


def _p2_thermal_stress() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "thermal_tuning_stress.csv"
    rows = [
        r
        for r in read_csv(source)
        if int(float(r["retune_interval_inferences"])) == 1000
        and abs(float(r["thermal_calibration_multiplier"]) - 1.0) < 1e-9
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    for variant, color in zip(variants, [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]):
        vals = sorted([r for r in rows if r["variant"] == variant], key=lambda r: _safe_float(r["thermal_drift_c"]))
        ax.plot(
            [_safe_float(r["thermal_drift_c"]) for r in vals],
            [_safe_float(r["energy_overhead_mj_proxy"]) for r in vals],
            marker="o",
            label=variant.replace("MobileViT-", ""),
            color=color,
        )
    ax.set_xlabel("Thermal drift (C)")
    ax.set_ylabel("Energy overhead proxy (mJ)")
    ax.set_title("Evidence-strengthening thermal tuning stress proxy")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_thermal_tuning_stress.png"))
    return (
        png,
        [source, *_existing([REPO_ROOT / "tables" / "p2_readiness_summary.json"])],
        "local/modelled P2 thermal tuning stress proxy",
        "Matplotlib-rendered evidence-strengthening thermal tuning stress proxy. Not packaged-device thermal validation.",
    )


def _p2_additional_family() -> tuple[pathlib.Path, list[pathlib.Path], str, str]:
    plt = _mpl()
    source = REPO_ROOT / "tables" / "additional_model_family_operator_summary.csv"
    rows = [r for r in read_csv(source) if int(float(r.get("row_count") or 0)) > 0]
    labels = [r["variant"].replace("EfficientFormer-", "EF-") for r in rows]
    x = list(range(len(rows)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    series = [
        ("PDPU-candidate", "pdpu_candidate_mac_share_percent", COLORS["green"], -width),
        ("Electronic remainder", "electronic_remainder_mac_share_percent", COLORS["gray"], 0.0),
        ("Hybrid support", "hybrid_support_mac_share_percent", COLORS["teal"], width),
    ]
    for label, key, color, offset in series:
        ax.bar([v + offset for v in x], [_safe_float(r[key]) for r in rows], width=width, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAC share (%)")
    ax.set_title("Evidence-strengthening additional-family mapping check")
    ax.legend(frameon=False)
    png = _save(fig, _path("mpl_fig_additional_model_family_mapping.png"))
    return (
        png,
        [source, *_existing([REPO_ROOT / "tables" / "p2_readiness_summary.json"])],
        "local/modelled P2 additional-family mapping proxy",
        "Matplotlib-rendered evidence-strengthening additional-family hook trace summary. Supports bounded mapping plausibility only.",
    )


FIGURE_BUILDERS = [
    _fig5_operator_amdahl_combo,
    _fig6_energy_uncertainty_combo,
    _fig7_mixed_evidence_context,
    _supp_energy_break_even_sensitivity,
    _energy_breakdown,
    _operator_coverage,
    _speedup_bound,
    _qkv_traffic,
    _nonideality,
    _fixed_subset_dose_response,
    _supp_fixed_subset_stress_modes,
    _fixed_subset_safe_region,
    _fixed_subset_worst_class_drop,
    _fixed_subset_margin_drift,
    _boundary_ablation,
    _scalability,
    _architecture_ablation,
    _energy_uncertainty,
    _p2_layout_area,
    _p2_thermal_stress,
    _p2_additional_family,
]


def run(output_dir: pathlib.Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(OUT_DIR)
    manifest = base_manifest("render_matplotlib_evidence_figures", "matplotlib-rendered no-silicon evidence figures")
    rendered: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for build in FIGURE_BUILDERS:
        try:
            png, sources, evidence_label, caption = build()
            sidecar = figure_sidecar(
                output_path=png,
                repo_root=REPO_ROOT,
                source_paths=sources,
                evidence_label=evidence_label,
                caption=caption,
                notes=[
                    "Rendered with matplotlib as a raster PNG data figure.",
                    "No SVG/PDF/TikZ/vector architecture output is generated by this script.",
                ],
            )
            rendered.append({"figure": relative(png), "sidecar": relative(sidecar)})
        except Exception as exc:
            skipped.append({"figure": build.__name__.removeprefix("_"), "reason": str(exc)})

    manifest_path = output_dir / "render_matplotlib_evidence_figures_manifest.json"
    manifest.update(
        {
            "status": "ok" if not skipped else "partial",
            "figures_dir": relative(OUT_DIR),
            "rendered": rendered,
            "skipped": skipped,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; matplotlib figure rendering is short and deterministic.",
        }
    )
    write_json(manifest_path, manifest)
    return {"manifest": manifest_path, "rendered": rendered, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir))
    print(json.dumps({k: relative(v) if isinstance(v, pathlib.Path) else v for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
