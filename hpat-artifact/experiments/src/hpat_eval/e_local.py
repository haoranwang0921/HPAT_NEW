"""E-local 证据管线：在本地工作站上生成的"建模级"证据与就绪度检查。

背景：论文证据分 P0（模拟/建模）/P1（真实硬件实测）/P2（物理代理/仿真）。
E-local 是 P0 档在本地工作站上落地的一套子管线，核心产出是"能耗
盈亏平衡敏感性分析"：对能耗模型里的每个假设（激光源倍率、转换器
倍率、存储总线倍率、热/校准倍率、数字剩余倍率、位宽、权重驻留策略）
做网格扫描，看不同假设组合下 HPAT 的能耗模型相对"名义值"（全系数 1）
和相对"参考能耗"（文献/外部端侧 GPU 数据）分别落在哪个区间，
据此判断结论在多大假设范围内成立。

同时本文件还维护"E-local 就绪度表"：检查各个实验产物（表格 CSV、
清单 JSON）是否已生成，并给出每条证据的"可声明性"（claim_eligible）
与证据等级（evidence_tier），防止把建模证据当成实测。

易混淆点：
- nominal（名义）= 所有乘数取 1.0 的基准能耗；
- reference（参考）= 外部/文献给出的端侧 GPU 能耗，只作上下文对照，
  不构成"HPAT 实测基线"。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from .energy_model import COMPONENTS


# 声明边界：E-local 证据都是本地工作站上"轨迹驱动或建模"的，
# 外部端侧/移动端数据仅作上下文。不是流片验证、不是作者实测的
# HPAT 端侧部署、也不是实测的 HPAT 加速比。
E_LOCAL_CLAIM_BOUNDARY = (
    "E-local evidence is trace-driven or modelled on the local workstation, with "
    "external edge/mobile rows used only as context. It is not fabricated-silicon "
    "validation, not author-measured HPAT edge/mobile deployment, and not measured HPAT speedup."
)

# 各类产出的证据标签（用于 CSV 表里的 evidence_label 列）
TRACE_DRIVEN_EVIDENCE_LABEL = (
    "local trace-driven HPAT activity model with explicit unit costs; not silicon-calibrated energy"
)
BREAK_EVEN_EVIDENCE_LABEL = (
    "modelled E-local energy break-even sensitivity; not measured HPAT deployment energy"
)
EXTERNAL_CONTEXT_EVIDENCE_LABEL = (
    "external public edge/mobile context; not an author-measured HPAT baseline"
)


# ---- 各输出表的列名清单（顺序即 CSV 表头顺序）----
# 能耗盈亏平衡表：每个场景一行
ENERGY_BREAK_EVEN_FIELDS = [
    "scenario_id",
    "model_variant",
    "optical_source_multiplier",
    "converter_multiplier",
    "memory_bus_multiplier",
    "thermal_calibration_multiplier",
    "digital_remainder_multiplier",
    "calibration_overhead_multiplier",
    "precision_bits",
    "weight_mode",
    "nominal_energy_mj",
    "scenario_energy_mj",
    "delta_vs_nominal_percent",
    "reference_energy_mj",
    "reference_label",
    "reference_evidence_tier",
    "energy_vs_reference_percent",
    "below_reference",
    "assumption_region_status",
    "dominant_component_group",
    "evidence_label",
    "claim_boundary",
]

# 能耗敏感度排名表：每个因子一行
ENERGY_SENSITIVITY_RANK_FIELDS = [
    "model_variant",
    "factor",
    "low_setting",
    "high_setting",
    "low_energy_mj",
    "high_energy_mj",
    "swing_percent_of_nominal",
    "rank_within_model",
    "evidence_label",
    "claim_boundary",
]

# 假设台账表：列出所有可调假设的含义与取值范围
ENERGY_ASSUMPTION_LEDGER_FIELDS = [
    "factor",
    "nominal_setting",
    "low_setting",
    "high_setting",
    "applies_to_components",
    "interpretation",
    "evidence_label",
    "claim_boundary",
]

# E-local 就绪度表
E_LOCAL_READINESS_FIELDS = [
    "lane",
    "status",
    "claim_eligible",
    "evidence_tier",
    "artifact",
    "meaning",
    "claim_boundary",
]

# 外部端侧上下文表
EXTERNAL_EDGE_CONTEXT_FIELDS = [
    "source_id",
    "source",
    "evidence_tier",
    "model_task",
    "device_platform",
    "runtime_framework",
    "precision",
    "batch_scenario",
    "reported_latency_throughput",
    "power_energy",
    "measurement_scope",
    "hpat_use",
    "source_url_or_doi",
    "verified_date",
    "claim_boundary",
]

# 外部端侧上下文来源表
EXTERNAL_EDGE_CONTEXT_SOURCE_FIELDS = [
    "source_id",
    "title",
    "url_or_doi",
    "source_type",
    "verified_date",
    "refresh_status",
    "claim_boundary",
    "notes",
]

# 外部端侧上下文"已验证"表：记录每条外部数据的可用范围与不可暗示的结论
EXTERNAL_EDGE_CONTEXT_VERIFIED_FIELDS = [
    "source_id",
    "source_url",
    "verified_date",
    "source_type",
    "metric_scope",
    "device_platform",
    "task_scope",
    "safe_use",
    "must_not_imply",
    "external_data_label",
    "claim_boundary",
]

# 端侧/移动端基线状态表
EDGE_MOBILE_BASELINE_STATUS_FIELDS = [
    "lane",
    "status",
    "evidence_tier",
    "claim_eligible_for_speedup",
    "artifact",
    "safe_use",
    "must_not_imply",
    "external_data_label",
    "claim_boundary",
]


# 能耗分项 → 假设族 的映射（与 evidence_strength 的族划分一致）
COMPONENT_FAMILIES = {
    "optical_source": {"Optical source and passive loss budget"},
    "converters": {"DAC and input modulation", "O/E readout"},
    "memory_bus": {"SRAM/eDRAM local buffers", "Electrical bus/data movement"},
    "thermal_calibration": {
        "MRR programming/hold tuning",
        "Thermal tuning/tracking",
        "Amortized calibration",
    },
    "digital_remainder": {"Digital accumulation/control", "LayerNorm/softmax/nonlinear electronics"},
}


def _f(row: dict[str, Any], key: str) -> float:
    """从字典安全取值并转 float；缺失/空值按 0 处理。

    :param row: 一行数据。
    :param key: 字段名。
    :return: 数值（缺失时 0.0）。
    """
    value = row.get(key, 0)
    if value in ("", None):
        return 0.0
    return float(value)


def _component_family(component: str) -> str:
    """查一个能耗分项属于哪个"假设族"。

    :param component: 能耗分项名（见 energy_model.COMPONENTS）。
    :return: 族名；查不到时归入 digital_remainder。
    """
    for family, names in COMPONENT_FAMILIES.items():
        if component in names:
            return family
    return "digital_remainder"


def _group_energy_by_model(component_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """把组件级能耗行按模型变体聚合成 {变体: {分项: 能耗 mJ}}。

    :param component_rows: 组件级能耗行（energy_model 的输出）。
    :return: 按模型分组的能耗字典。
    """
    grouped: dict[str, dict[str, float]] = {}
    for row in component_rows:
        model = row["model_variant"]
        bucket = grouped.setdefault(model, {component: 0.0 for component in COMPONENTS})
        bucket[row["component_group"]] = bucket.get(row["component_group"], 0.0) + _f(row, "energy_mj")
    return grouped


def _reference_energy_by_model(fig6_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """从"图 6 参考数据"里提取每个模型的 GPU 能耗参考值。

    参考能耗用于回答"HPAT 能耗模型 vs 现有 GPU 能耗"的相对关系，
    但只是文献上下文，不代表 HPAT 实测。

    :param fig6_rows: 图 6 的数据行（含 metric / platform_label / value）。
    :return: {变体: {"energy_mj","label","evidence_tier"}}。
    """
    refs: dict[str, dict[str, str]] = {}
    for row in fig6_rows:
        if row.get("metric") != "energy":
            continue
        label = row.get("platform_label", "")
        if "GPU" not in label:  # 只取 GPU 平台的能耗作参考
            continue
        refs[row["model_variant"]] = {
            "energy_mj": str(row.get("value", "")),
            "label": label,
            "evidence_tier": row.get("evidence_tier", ""),
        }
    return refs


def _scenario_energy(
    component_energy: dict[str, float],
    *,
    optical_source_multiplier: float,
    converter_multiplier: float,
    memory_bus_multiplier: float,
    thermal_calibration_multiplier: float,
    digital_remainder_multiplier: float,
    calibration_overhead_multiplier: float,
    precision_bits: int,
    weight_mode: str,
) -> tuple[float, str]:
    """按一组假设系数计算某个"场景"的总能耗，并指出能耗最大的分项。

    系数含义：optical_source_multiplier=激光源倍率；converter_multiplier=
    转换器倍率；memory_bus_multiplier=存储总线倍率；thermal_calibration_
    multiplier=热/校准倍率；digital_remainder_multiplier=数字剩余倍率；
    calibration_overhead_multiplier=校准开销额外倍率。

    精度影响：precison_factor = 位宽/8，位宽越高，转换器/存储总线/数字
    部分能耗按比例放大（更高精度需要更贵的数据通路）。
    权重驻留策略影响：resident（常驻，省内存但热校高）、streamed（流式）、
    reprogrammed（每层重写权重，热校开销大）。

    :param component_energy: {分项: 能耗} 字典。
    :return: (场景总能耗, 能耗最大的分项名)。
    """
    precision_factor = max(float(precision_bits), 1.0) / 8.0  # 位宽相对 8 位的系数
    # 权重驻留策略对 存储总线/热校准 两族的修正系数
    weight_mode_factors = {
        "resident": {"memory_bus": 0.85, "thermal_calibration": 0.85},   # 常驻：省搬运，热校准低
        "streamed": {"memory_bus": 1.25, "thermal_calibration": 1.0},    # 流式：搬运多
        "reprogrammed": {"memory_bus": 1.0, "thermal_calibration": 1.6}, # 重写：热校准高
    }
    mode = weight_mode_factors.get(weight_mode, weight_mode_factors["resident"])
    # 各族的最终倍率 = 用户系数 × 精度影响 × 权重策略影响
    family_multipliers = {
        "optical_source": optical_source_multiplier,
        "converters": converter_multiplier * precision_factor,
        "memory_bus": memory_bus_multiplier * mode["memory_bus"] * precision_factor,
        "thermal_calibration": thermal_calibration_multiplier * mode["thermal_calibration"],
        "digital_remainder": digital_remainder_multiplier * precision_factor,
    }
    total = 0.0
    dominant_component = ""   # 能耗最大的分项
    dominant_value = -1.0
    for component, value in component_energy.items():
        family = _component_family(component)
        multiplier = family_multipliers[family]
        if component == "Amortized calibration":
            multiplier *= calibration_overhead_multiplier  # 校准分项额外乘校准倍率
        scenario_value = value * multiplier
        total += scenario_value
        if scenario_value > dominant_value:
            dominant_value = scenario_value
            dominant_component = component
    return total, dominant_component


def energy_assumption_ledger_rows() -> list[dict[str, Any]]:
    """生成"假设台账"：列出全部可调假设及其名义/低/高取值与含义。

    作用是让读者看到每个乘数的合理范围，避免"悄悄用了极端假设"。

    :return: 台账表（每行一个假设因子）。
    """
    rows = [
        (
            "optical_source_multiplier",
            "1.0",
            "0.5",
            "4.0",
            "Optical source and passive loss budget",
            "Models laser/source/passive-loss pressure around the nominal trace-driven energy account.",
        ),
        (
            "converter_multiplier",
            "1.0",
            "0.5",
            "4.0",
            "DAC and input modulation; O/E readout",
            "Models DAC/ADC/PD/TIA uncertainty and precision-coupled converter pressure.",
        ),
        (
            "memory_bus_multiplier",
            "1.0",
            "0.5",
            "2.0",
            "SRAM/eDRAM local buffers; Electrical bus/data movement",
            "Models memory hierarchy and bus/fabric energy pressure.",
        ),
        (
            "thermal_calibration_multiplier",
            "1.0",
            "0.5",
            "3.0",
            "MRR tuning; thermal tracking; amortized calibration",
            "Models thermal tuning and calibration stress before physical-design validation.",
        ),
        (
            "digital_remainder_multiplier",
            "1.0",
            "0.75",
            "1.5",
            "Digital accumulation/control; nonlinear electronics",
            "Models electronic remainder and nonlinear/normalization overhead.",
        ),
        (
            "weight_mode",
            "resident",
            "resident",
            "reprogrammed",
            "Memory/bus and MRR tuning families",
            "Models resident, streamed, and reprogrammed weight-use policies.",
        ),
    ]
    return [
        {
            "factor": factor,
            "nominal_setting": nominal,
            "low_setting": low,
            "high_setting": high,
            "applies_to_components": components,
            "interpretation": interpretation,
            "evidence_label": BREAK_EVEN_EVIDENCE_LABEL,
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        }
        for factor, nominal, low, high, components, interpretation in rows
    ]


def energy_break_even_rows(
    component_rows: list[dict[str, Any]], fig6_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """核心：能耗盈亏平衡网格扫描，输出每个假设场景一行。

    对 6 个乘数 × 3 档位宽 × 3 种权重策略做全组合扫描（共
    4×4×3×3×3×3×3×3 = 上万场景），每个场景算一次总能耗，并
    标记它与"名义能耗"的相对区间（at_or_below / within_2x / above_2x），
    以及相对"GPU 参考能耗"的百分比和是否低于参考值。

    :param component_rows: 组件级能耗行。
    :param fig6_rows: 图 6 参考数据行（提供 GPU 能耗参考）。
    :return: 盈亏平衡表（每场景一行）。
    """
    grouped = _group_energy_by_model(component_rows)
    references = _reference_energy_by_model(fig6_rows)
    rows: list[dict[str, Any]] = []
    scenario_id = 0
    for model, components in grouped.items():
        nominal = sum(components.values()) or 1.0  # 名义能耗 = 全乘数 1
        ref = references.get(model, {})
        reference_energy = float(ref["energy_mj"]) if ref.get("energy_mj") not in ("", None) else 0.0
        # 八重循环穷举所有假设组合
        for optical in [0.5, 1.0, 2.0, 4.0]:
            for converter in [0.5, 1.0, 2.0, 4.0]:
                for memory in [0.5, 1.0, 2.0]:
                    for thermal in [0.5, 1.0, 3.0]:
                        for digital in [0.75, 1.0, 1.5]:
                            for calibration in [0.25, 1.0, 4.0]:
                                for precision in [4, 8, 12]:
                                    for weight_mode in ["resident", "streamed", "reprogrammed"]:
                                        energy, dominant = _scenario_energy(
                                            components,
                                            optical_source_multiplier=optical,
                                            converter_multiplier=converter,
                                            memory_bus_multiplier=memory,
                                            thermal_calibration_multiplier=thermal,
                                            digital_remainder_multiplier=digital,
                                            calibration_overhead_multiplier=calibration,
                                            precision_bits=precision,
                                            weight_mode=weight_mode,
                                        )
                                        # 相对名义能耗的区间标签
                                        if energy <= nominal:
                                            status = "at_or_below_nominal_model"
                                        elif energy <= 2.0 * nominal:
                                            status = "within_2x_nominal_model"
                                        else:
                                            status = "above_2x_nominal_model"
                                        rows.append(
                                            {
                                                "scenario_id": scenario_id,
                                                "model_variant": model,
                                                "optical_source_multiplier": f"{optical:.4f}",
                                                "converter_multiplier": f"{converter:.4f}",
                                                "memory_bus_multiplier": f"{memory:.4f}",
                                                "thermal_calibration_multiplier": f"{thermal:.4f}",
                                                "digital_remainder_multiplier": f"{digital:.4f}",
                                                "calibration_overhead_multiplier": f"{calibration:.4f}",
                                                "precision_bits": precision,
                                                "weight_mode": weight_mode,
                                                "nominal_energy_mj": f"{nominal:.6f}",
                                                "scenario_energy_mj": f"{energy:.6f}",
                                                "delta_vs_nominal_percent": f"{100.0 * (energy - nominal) / nominal:.4f}",
                                                "reference_energy_mj": f"{reference_energy:.6f}" if reference_energy else "",
                                                "reference_label": ref.get("label", ""),
                                                "reference_evidence_tier": ref.get("evidence_tier", ""),
                                                "energy_vs_reference_percent": (
                                                    f"{100.0 * energy / reference_energy:.4f}" if reference_energy else ""
                                                ),
                                                "below_reference": (  # 是否低于 GPU 参考能耗
                                                    str(bool(reference_energy and energy <= reference_energy)).lower()
                                                    if reference_energy
                                                    else ""
                                                ),
                                                "assumption_region_status": status,
                                                "dominant_component_group": dominant,
                                                "evidence_label": BREAK_EVEN_EVIDENCE_LABEL,
                                                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
                                            }
                                        )
                                        scenario_id += 1
    return rows


def energy_sensitivity_rank_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """能耗敏感度排名：看哪个假设因子对总能耗的影响最大。

    方法：对每个因子，把它从"低档"调到"高档"（其余因子保持默认），
    计算总能耗的摆动幅度（swing = |高档-低档| / 名义 × 100%），
    然后在模型内按摆动幅度从大到小排名。这样能回答"哪个假设最敏感，
    值得优先精确化/实测"。

    :param component_rows: 组件级能耗行。
    :return: 敏感度排名表（每模型 × 每因子一行）。
    """
    grouped = _group_energy_by_model(component_rows)
    # 每个因子：(因子名, 低档覆盖, 高档覆盖)
    factors = [
        ("optical_source_multiplier", {"optical_source_multiplier": 0.5}, {"optical_source_multiplier": 4.0}),
        ("converter_multiplier", {"converter_multiplier": 0.5}, {"converter_multiplier": 4.0}),
        ("memory_bus_multiplier", {"memory_bus_multiplier": 0.5}, {"memory_bus_multiplier": 2.0}),
        (
            "thermal_calibration_multiplier",
            {"thermal_calibration_multiplier": 0.5},
            {"thermal_calibration_multiplier": 3.0},
        ),
        ("digital_remainder_multiplier", {"digital_remainder_multiplier": 0.75}, {"digital_remainder_multiplier": 1.5}),
        ("calibration_overhead_multiplier", {"calibration_overhead_multiplier": 0.25}, {"calibration_overhead_multiplier": 4.0}),
        ("precision_bits", {"precision_bits": 4}, {"precision_bits": 12}),
        ("weight_mode", {"weight_mode": "resident"}, {"weight_mode": "reprogrammed"}),
    ]
    rows: list[dict[str, Any]] = []
    # 非扫描因子的默认值（扫描哪个因子时其余保持这些默认）
    defaults: dict[str, Any] = {
        "optical_source_multiplier": 1.0,
        "converter_multiplier": 1.0,
        "memory_bus_multiplier": 1.0,
        "thermal_calibration_multiplier": 1.0,
        "digital_remainder_multiplier": 1.0,
        "calibration_overhead_multiplier": 1.0,
        "precision_bits": 8,
        "weight_mode": "resident",
    }
    for model, components in grouped.items():
        nominal = sum(components.values()) or 1.0
        staged = []
        for factor, low_override, high_override in factors:
            low_kwargs = dict(defaults)
            high_kwargs = dict(defaults)
            low_kwargs.update(low_override)   # 只改当前因子到低档
            high_kwargs.update(high_override) # 只改当前因子到高档
            low_energy, _ = _scenario_energy(components, **low_kwargs)
            high_energy, _ = _scenario_energy(components, **high_kwargs)
            swing = 100.0 * abs(high_energy - low_energy) / nominal  # 摆动幅度（% 名义值）
            staged.append(
                {
                    "model_variant": model,
                    "factor": factor,
                    "low_setting": str(next(iter(low_override.values()))),
                    "high_setting": str(next(iter(high_override.values()))),
                    "low_energy_mj": f"{low_energy:.6f}",
                    "high_energy_mj": f"{high_energy:.6f}",
                    "swing_percent_of_nominal": f"{swing:.4f}",
                    "rank_within_model": 0,  # 排名稍后按摆动幅度重排
                    "evidence_label": BREAK_EVEN_EVIDENCE_LABEL,
                    "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
                }
            )
        # 模型内按摆动幅度从大到小排序，再填排名
        staged.sort(key=lambda row: float(row["swing_percent_of_nominal"]), reverse=True)
        for rank, row in enumerate(staged, 1):
            row["rank_within_model"] = rank
            rows.append(row)
    return rows


def e_local_readiness_rows(repo_root: pathlib.Path, output_dir: pathlib.Path) -> list[dict[str, Any]]:
    """检查 E-local 各实验产物的就绪状态，输出就绪度表。

    对每个"证据通道"（lane）检查对应表格/清单文件是否生成：
    - 文件存在且非空 → ready_with_limitations（并给出含义/可声明性/证据等级）；
    - 个别通道有特殊逻辑（如 fixed_subset_robustness 无子集时降级、
      author_measured_edge_mobile_baseline 永远 blocked，因为作者没做端侧实测）；
    - 文件不存在 → not_run。

    :param repo_root: 仓库根目录。
    :param output_dir: 输出目录（用于找 manifest）。
    :return: 就绪度表（每通道一行）。
    """
    candidates = {
        "trace_driven_activity": repo_root / "tables" / "hpat_energy_by_component_trace_driven.csv",
        "energy_break_even": repo_root / "tables" / "energy_break_even_sweep.csv",
        "fixed_subset_robustness": repo_root / "tables" / "nonideality_accuracy_fixed_subset.csv",
        "fixed_subset_safe_region": repo_root / "tables" / "nonideality_accuracy_safe_region.csv",
        "fixed_subset_classwise_statistics": repo_root / "tables" / "nonideality_accuracy_classwise.csv",
        "boundary_ablation_diagnostic": repo_root / "tables" / "nonideality_boundary_ablation_summary.csv",
        "operator_mapping_closure": repo_root / "tables" / "operator_mapping_closure.csv",
        "operator_profiler_closure": repo_root / "tables" / "operator_fx_profiler_closure.csv",
        "author_measured_edge_mobile_baseline": repo_root / "tables" / "edge_mobile_baseline_status.csv",
        "external_edge_context": repo_root / "tables" / "external_edge_context_verified.csv",
        "external_edge_online_verification": repo_root / "tables" / "external_edge_context_online_verification.csv",
        "submission_freeze_package": repo_root / "experiments" / "results" / "e_local_submission_freeze_latest" / "e_local_freeze_manifest.json",
    }
    rows: list[dict[str, Any]] = []
    for lane, artifact in candidates.items():
        exists = artifact.exists() and artifact.stat().st_size > 0  # 存在且非空
        if lane == "fixed_subset_robustness" and not exists:
            # 固定标注子集缺失：只有 manifest 则 blocked，否则 not_run
            manifest = output_dir / "e_local_fixed_subset_manifest.json"
            status = "blocked" if manifest.exists() else "not_run"
            meaning = "No fixed labeled validation subset was supplied; robustness remains proxy-only."
            claim_eligible = "false"
            tier = "not claimable"
        elif lane == "author_measured_edge_mobile_baseline" and exists:
            # 作者实测端侧基线：文件存在也必须 blocked（没有真实日志就是没有）
            status = "blocked"
            meaning = "No author-measured edge/mobile device logs are available; external public context cannot replace this lane."
            claim_eligible = "false"
            tier = "not claimable"
        elif exists:
            status = "ready_with_limitations"
            meaning = {  # 每个通道的人类可读含义
                "trace_driven_activity": "Bottom-up local trace-driven activity and energy model is available.",
                "energy_break_even": "Modelled break-even and sensitivity sweep is available.",
                "fixed_subset_robustness": "Imagenette/external-public fixed-subset non-ideality accuracy evidence is available when paired with the source ledger.",
                "fixed_subset_safe_region": "Dose-response safe-region derivation is available for the fixed subset.",
                "fixed_subset_classwise_statistics": "Class-wise, bootstrap, worst-class, and margin-drift fixed-subset statistics are available.",
                "boundary_ablation_diagnostic": "HPAT-mapping versus all-linear diagnostic boundary ablation is available; all-linear rows are diagnostic only.",
                "operator_mapping_closure": "Linear-hook coverage and electronic/analytical remainder boundary ledger is available.",
                "operator_profiler_closure": "Synthetic profiler closure ledger is available for function-level boundary explanation only.",
                "external_edge_context": "External edge/mobile context table is available for motivation only; not HPAT speedup evidence.",
                "external_edge_online_verification": "External edge/mobile context URLs have online verification metadata.",
                "submission_freeze_package": "Submission evidence package manifest/checksum freeze is available.",
            }[lane]
            # 上下文类/诊断类通道不允许支撑结论（false），其余允许（true）
            claim_eligible = "false" if lane in {
                "external_edge_context",
                "external_edge_online_verification",
                "boundary_ablation_diagnostic",
                "submission_freeze_package",
            } else "true"
            tier = {  # 每个通道的证据等级标签
                "trace_driven_activity": "local trace-driven/modelled",
                "energy_break_even": "local/modelled sensitivity",
                "fixed_subset_robustness": "fixed-subset local accuracy",
                "fixed_subset_safe_region": "fixed-subset local sensitivity",
                "fixed_subset_classwise_statistics": "fixed-subset local post-processing statistics",
                "boundary_ablation_diagnostic": "fixed-subset diagnostic sensitivity",
                "operator_mapping_closure": "local hook/proxy boundary ledger",
                "operator_profiler_closure": "local synthetic profiler boundary ledger",
                "external_edge_context": "external-public context",
                "external_edge_online_verification": "external-public context verification",
                "submission_freeze_package": "artifact freeze/manifest",
            }[lane]
        else:
            status = "not_run"
            meaning = "E-local artifact has not been generated."
            claim_eligible = "false"
            tier = "not claimable"
        rows.append(
            {
                "lane": lane,
                "status": status,
                "claim_eligible": claim_eligible,
                "evidence_tier": tier,
                "artifact": str(artifact.relative_to(repo_root)),
                "meaning": meaning,
                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            }
        )
    return rows


def e_local_readiness_markdown(rows: list[dict[str, Any]]) -> str:
    """把就绪度表渲染成 Markdown 表格文本（供论文附录直接粘贴）。

    :param rows: e_local_readiness_rows 的输出。
    :return: Markdown 字符串。
    """
    lines = [
        "# E-local Strong Submission Readiness",
        "",
        "Status: generated readiness summary for local ASP-DAC evidence strengthening.",
        "",
        "| Lane | Status | Claim eligible | Evidence tier | Meaning |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lane']} | `{row['status']}` | {row['claim_eligible']} | {row['evidence_tier']} | {row['meaning']} |"
        )
    lines.extend(
        [
            "",
            "Claim boundary: E-local artifacts remain architecture-level, trace-driven, modelled, fixed-subset, or literature-context evidence. They do not create silicon validation, measured HPAT edge deployment, measured HPAT speedup, or full ImageNet robustness.",
            "",
        ]
    )
    return "\n".join(lines)


def manifest_status(path: pathlib.Path) -> str:
    """读取一个 manifest 的 status 字段，用于快速判断。

    :param path: manifest 文件路径。
    :return: "missing"（不存在）/ "invalid_json"（损坏）/ status 值。
    """
    if not path.exists():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "invalid_json"
