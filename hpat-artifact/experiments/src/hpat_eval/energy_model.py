"""能耗建模：把"活动轨迹"（activity trace）换算成光子芯片的分项能耗。

背景：活动轨迹记录了每个模型算子执行时用了多少次 DAC/ADC/PD/TIA
采样、激活了多少个 MRR 微环、搬了多少字节等"活动量"。本文件给每个
活动量乘以对应的"单位成本"（单位能耗，如每采样 1.5 pJ），累加得到
10 个能耗分项（激光源、DAC 调制、MRR 调谐、O/E 读出、片上存储、
总线搬运、数字控制、非线性电子、校准等），最终输出按"组件"汇总的
能耗表和按"层"汇总的能耗表。

易混淆点：
- pJ（皮焦耳）是原始单位，最终统一换算成 mJ（毫焦耳，1e9 pJ = 1e-3 J）。
- "envelope 归一化"：论文有个能耗包络（energy envelope）作为校准目标，
  若开启 normalize_to_envelope，会把算出的总能耗等比缩放对齐到包络值，
  保持各分项的"占比"不变，只改绝对量。
"""

from __future__ import annotations

import math
from typing import Any


# 能耗的 10 个分项名称（顺序即输出表里 component_group 的取值）
COMPONENTS = [
    "Optical source and passive loss budget",       # 激光光源与无源损耗预算
    "DAC and input modulation",                     # DAC 与输入调制
    "MRR programming/hold tuning",                  # MRR 微环编程/保持调谐
    "Thermal tuning/tracking",                      # 热调谐/热跟踪
    "O/E readout",                                  # 光电读出（PD/TIA/ADC 等）
    "SRAM/eDRAM local buffers",                     # 片上 SRAM/eDRAM 缓冲
    "Electrical bus/data movement",                 # 电总线/数据搬运
    "Digital accumulation/control",                 # 数字累加/控制
    "LayerNorm/softmax/nonlinear electronics",      # 归一化/softmax/非线性电子
    "Amortized calibration",                        # 摊还的校准开销
]


# 各项活动量的默认"单位成本"（单位能耗）表。
# 单位说明（见键名）：每纳秒多少 pJ（pj_per_ns）、每个采样多少 pJ（sample_pj）、
# 每个微环每纳秒多少 pJ（pj_per_ring_ns）、每字节多少 pJ（pj_per_byte）、
# 每事件多少 pJ（pj_per_event）、每次运算多少 pJ（per_op）。
DEFAULT_UNIT_COSTS = {
    "optical_source_pj_per_ns": 1500.0,     # 激光源：每纳秒 1500 pJ
    "dac_sample_pj": 1.5,                   # 一次 DAC 采样 1.5 pJ
    "input_mod_sample_pj": 0.8,             # 一次输入调制 0.8 pJ
    "mrr_program_pj": 4.0,                  # 编程一个 MRR 微环 4 pJ
    "mrr_hold_pj_per_ring_ns": 0.00003,     # 每个激活微环保持状态：每纳秒 0.00003 pJ
    "thermal_pj_per_ring_ns": 0.00007,      # 热跟踪：每微环每纳秒 0.00007 pJ
    "pd_sample_pj": 0.35,                   # 一次 PD（光电探测器）采样 0.35 pJ
    "tia_sample_pj": 0.65,                  # 一次 TIA（跨阻放大器）采样 0.65 pJ
    "sample_hold_pj": 0.20,                 # 一次采样保持 0.20 pJ
    "adc_sample_pj": 3.2,                   # 一次 ADC 采样 3.2 pJ
    "memory_read_pj_per_byte": 0.05,        # 从片上存储读 1 字节 0.05 pJ
    "memory_write_pj_per_byte": 0.07,       # 写入片上存储 1 字节 0.07 pJ
    "bus_pj_per_byte": 0.10,                # 总线搬运 1 字节 0.10 pJ
    "control_pj_per_event": 8.0,            # 一次控制事件 8 pJ
    "nonlinear_pj_per_op": 0.25,            # 一次非线性运算 0.25 pJ
    "calibration_pj_per_event": 15000.0,    # 一次完整校准 15000 pJ（成本高，需摊还）
}

# 单位成本表必须包含的键；以及每个成本条目应有的元信息字段（用于审计溯源）
UNIT_COST_REQUIRED_KEYS = list(DEFAULT_UNIT_COSTS.keys())
UNIT_COST_METADATA_FIELDS = ["value", "unit", "source", "technology_assumption", "precision", "scope_note"]


def _f(row: dict[str, Any], key: str) -> float:
    """从字典安全取值并转 float；缺失/空值按 0 处理。

    :param row: 一行活动数据。
    :param key: 字段名。
    :return: 数值（缺失时 0.0）。
    """
    value = row.get(key, 0)
    if value in ("", None):
        return 0.0
    return float(value)


def _flat_unit_cost_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """把"外部单位成本载荷"拍平成 {键: 数值} 的简单字典。

    外部载荷可能有两种形态：{"costs": {...}} 或直接是平铺字典；
    其中每个成本项又可能是 {"value": 1.5, ...} 或纯数字。
    本函数统一取出纯数值，供后续覆盖默认成本。

    :param payload: 外部传入的成本载荷。
    :return: 键到数值的字典（只保留必填键）。
    """
    if not payload:
        return {}
    if isinstance(payload.get("costs"), dict):
        return {
            key: (value.get("value") if isinstance(value, dict) else value)
            for key, value in payload["costs"].items()
            if key in UNIT_COST_REQUIRED_KEYS
        }
    return {key: value for key, value in payload.items() if key in UNIT_COST_REQUIRED_KEYS}


def _rich_cost_records(config: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """收集"富格式"成本记录（每个成本项带完整元信息字典）。

    优先级：config 里的 energy_unit_cost_records 先放入，外部载荷里的
    成本项再覆盖（后者更具体）。只有 value 字段存在的条目才会在
    后续真正生效。

    :param config: 实验配置。
    :param payload: 外部成本载荷。
    :return: {成本键: 完整元信息字典}。
    """
    records: dict[str, dict[str, Any]] = {}
    config_records = config.get("energy_unit_cost_records", {})
    if isinstance(config_records, dict):
        records.update({key: dict(value) for key, value in config_records.items() if isinstance(value, dict)})
    if isinstance(payload, dict):
        raw = payload.get("costs", payload)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key in UNIT_COST_REQUIRED_KEYS and isinstance(value, dict):
                    records[key] = dict(value)
    return records


def resolve_unit_costs(
    config: dict[str, Any], unit_cost_payload: dict[str, Any] | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """解析最终使用的单位成本表，并返回成本完整性审计信息。

    覆盖优先级（高到低）：外部载荷 > 配置文件 > 内置默认值。
    同时审计三项完整性：数值是否齐全、来源说明是否齐全、
    元信息字段是否齐全——这些审计结果会写入清单，方便审稿人核对
    每个能耗数字是从哪来的。

    :param config: 实验配置。
    :param unit_cost_payload: 外部单位成本载荷（可选）。
    :return: (最终成本字典, 审计元信息字典)。
    """
    costs = dict(DEFAULT_UNIT_COSTS)          # 起点：内置默认值
    config_costs = dict(config.get("energy_unit_costs", {}))
    payload_costs = _flat_unit_cost_payload(unit_cost_payload)
    rich_records = _rich_cost_records(config, unit_cost_payload)
    # explicit_keys：记录"哪些键被用户显式指定过"（无论是否带元信息）
    explicit_keys = set(config_costs) | set(payload_costs)
    explicit_keys |= {key for key, value in rich_records.items() if "value" in value}
    # 依次覆盖：配置文件 → 外部载荷 → 富记录（后者优先级最高）
    costs.update({k: float(v) for k, v in config_costs.items() if k in UNIT_COST_REQUIRED_KEYS})
    costs.update({k: float(v) for k, v in payload_costs.items() if k in UNIT_COST_REQUIRED_KEYS})
    costs.update(
        {
            key: float(record["value"])
            for key, record in rich_records.items()
            if key in UNIT_COST_REQUIRED_KEYS and "value" in record
        }
    )
    missing = [key for key in UNIT_COST_REQUIRED_KEYS if key not in explicit_keys]

    # 来源审计：统计哪些成本项没有"出处"（source 字段为空）
    config_sources = config.get("energy_unit_cost_sources", {})
    payload_sources = unit_cost_payload.get("sources", {}) if isinstance(unit_cost_payload, dict) else {}
    source_keys = set(config_sources) | set(payload_sources)
    source_keys |= {
        key
        for key, record in rich_records.items()
        if key in UNIT_COST_REQUIRED_KEYS and str(record.get("source", "")).strip()
    }
    missing_sources = [key for key in UNIT_COST_REQUIRED_KEYS if key not in source_keys]

    # 元信息审计：逐项检查富记录的 6 个元字段是否填全
    missing_metadata: dict[str, list[str]] = {}
    for key in UNIT_COST_REQUIRED_KEYS:
        record = rich_records.get(key, {})
        missing_fields = [field for field in UNIT_COST_METADATA_FIELDS if not str(record.get(field, "")).strip()]
        if missing_fields:
            missing_metadata[key] = missing_fields

    # 汇总审计结果与工艺假设/精度信息
    meta = {
        "required_keys": UNIT_COST_REQUIRED_KEYS,
        "required_metadata_fields": UNIT_COST_METADATA_FIELDS,
        "explicit_keys": sorted(explicit_keys),
        "missing_explicit_keys": missing,
        "values_complete": not missing,
        "sources_complete": not missing_sources,
        "missing_source_keys": missing_sources,
        "metadata_complete": not missing_metadata,
        "missing_metadata": missing_metadata,
        "technology_assumption": (  # 工艺假设（如 45nm / 130nm 光子工艺）
            unit_cost_payload.get("technology_assumption")
            if isinstance(unit_cost_payload, dict)
            else config.get("energy_unit_cost_technology_assumption", "")
        )
        or config.get("energy_unit_cost_technology_assumption", ""),
        "precision": (  # 精度声明（如 "engineering estimate"）
            unit_cost_payload.get("precision")
            if isinstance(unit_cost_payload, dict)
            else config.get("energy_unit_cost_precision", "")
        )
        or config.get("energy_unit_cost_precision", ""),
    }
    return costs, meta


def component_energy_rows(
    activity_rows: list[dict[str, Any]],
    config: dict[str, Any],
    normalize_to_envelope: bool = True,
    source_label: str = "model/config-derived activity proxy",
    unit_costs: dict[str, float] | None = None,
    activity_source: str = "proxy_hook_trace",
    unit_cost_status: str = "config_explicit_values",
    evidence_tier: str = "local/modelled",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """核心入口：由活动轨迹行计算能耗，返回"组件级"与"层级"两张表。

    计算步骤：
    1) 逐行把活动量 × 单位成本，得到 10 个分项的能耗（单位 pJ）；
    2) 同一模型变体的所有层累加到同一个"组件桶"里；
    3) （可选）用能耗包络做归一化：把总能耗等比缩放到包络目标值；
    4) 组件级表输出每个分项占总能耗的百分比；层级表输出每层的
       光学路径 / 存储控制电子 两块原始能耗。

    :param activity_rows: 归一化后的活动轨迹行（见 schemas.normalize_hpat_activity_rows）。
    :param config: 实验配置（读 energy_envelope_mj 等）。
    :param normalize_to_envelope: 是否缩放对齐到能耗包络。
    :param source_label: 证据标签前缀。
    :param unit_costs: 单位成本表；None 时自动解析。
    :param activity_source: 活动数据来源标签（用于溯源）。
    :param unit_cost_status: 单位成本状态标签。
    :param evidence_tier: 证据等级标签。
    :return: (组件级能耗行列表, 层级能耗行列表)。
    """
    costs = unit_costs or resolve_unit_costs(config)[0]  # 取最终单位成本
    by_variant: dict[str, dict[str, float]] = {}  # 变体名 -> {分项: 累计 pJ}
    per_layer_rows: list[dict[str, Any]] = []     # 层级输出表
    for row in activity_rows:
        variant = str(row["model_variant"])
        bucket = by_variant.setdefault(variant, {component: 0.0 for component in COMPONENTS})
        # 取出该行全部活动量（_f 保证缺失时按 0 处理）
        optical_ns = _f(row, "active_optical_time_ns")    # 光路活动时长（ns）
        thermal_ns = _f(row, "thermal_tracking_time_ns")  # 热跟踪时长（ns）
        dac = _f(row, "n_dac_samples")      # DAC 采样次数
        adc = _f(row, "n_adc_samples")      # ADC 采样次数
        pd = _f(row, "n_pd_samples")        # PD 采样次数
        tia = _f(row, "n_tia_samples")      # TIA 采样次数
        mrr_active = _f(row, "n_mrr_active")   # 激活 MRR 数
        mrr_program = _f(row, "n_mrr_program") # 编程 MRR 数
        read_bytes = _f(row, "buffer_read_bytes")   # 读出字节
        write_bytes = _f(row, "buffer_write_bytes") # 写入字节
        bus_bytes = _f(row, "bus_bytes")            # 总线字节
        nonlinear = _f(row, "nonlinear_ops")        # 非线性算子数
        calibration = _f(row, "calibration_events") # 校准事件数
        # 控制事件 = 校准事件 + DAC/ADC 采样每 4096 次计一次控制（近似）；
        # 若完全没有任何采样，则只算校准事件
        control_events = calibration + math.ceil((dac + adc) / 4096.0) if (dac + adc) else calibration
        # 本层 10 个分项的能耗（单位 pJ），每项 = 活动量 × 单位成本
        layer_components = {
            "Optical source and passive loss budget": optical_ns * costs["optical_source_pj_per_ns"],
            "DAC and input modulation": dac * (costs["dac_sample_pj"] + costs["input_mod_sample_pj"]),
            "MRR programming/hold tuning": (
                mrr_program * costs["mrr_program_pj"] + mrr_active * optical_ns * costs["mrr_hold_pj_per_ring_ns"]
            ),
            "Thermal tuning/tracking": mrr_active * max(thermal_ns, optical_ns) * costs["thermal_pj_per_ring_ns"],
            "O/E readout": adc * costs["adc_sample_pj"]
            + pd * costs["pd_sample_pj"]
            + tia * costs["tia_sample_pj"]
            + adc * costs["sample_hold_pj"],
            "SRAM/eDRAM local buffers": read_bytes * costs["memory_read_pj_per_byte"]
            + write_bytes * costs["memory_write_pj_per_byte"],
            "Electrical bus/data movement": bus_bytes * costs["bus_pj_per_byte"],
            "Digital accumulation/control": control_events * costs["control_pj_per_event"],
            "LayerNorm/softmax/nonlinear electronics": nonlinear * costs["nonlinear_pj_per_op"],
            "Amortized calibration": calibration * costs["calibration_pj_per_event"],
        }
        # 累加进该变体的组件桶
        for component, value_pj in layer_components.items():
            bucket[component] += value_pj
        # 层级输出：把本层能耗粗分成"光学路径"与"存储/控制/电子"两块（×1e-9 从 pJ 转 mJ）
        per_layer_rows.append(
            {
                "model_variant": variant,
                "layer_id": row["layer_id"],
                "op_type": row["op_type"],
                "optical_path_mj_raw": f"{(layer_components['Optical source and passive loss budget'] + layer_components['DAC and input modulation'] + layer_components['MRR programming/hold tuning'] + layer_components['O/E readout']) * 1e-9:.9f}",
                "memory_control_electronic_mj_raw": f"{(layer_components['SRAM/eDRAM local buffers'] + layer_components['Electrical bus/data movement'] + layer_components['Digital accumulation/control'] + layer_components['LayerNorm/softmax/nonlinear electronics'] + layer_components['Amortized calibration']) * 1e-9:.9f}",
                "evidence_label": source_label,
                "activity_source": activity_source,
                "unit_cost_status": unit_cost_status,
                "evidence_tier": evidence_tier,
            }
        )

    envelope = config.get("energy_envelope_mj", {})  # 能耗包络：{变体: 目标 mJ}
    component_rows: list[dict[str, Any]] = []
    for variant, component_pj in by_variant.items():
        raw_total_mj = sum(component_pj.values()) * 1e-9  # 原始总能耗（pJ→mJ）
        # 归一化目标：包络里指定了就对齐到包络，否则保持原始值（factor=1）
        target = float(envelope.get(variant, raw_total_mj)) if normalize_to_envelope else None
        # 缩放因子：目标/原始；目标或原始任一非法时退回 1.0（不缩放）
        factor = (
            target / raw_total_mj
            if target is not None and raw_total_mj > 0 and target > 0
            else 1.0
        )
        final_total_mj = raw_total_mj * factor
        for component in COMPONENTS:
            raw_mj = component_pj[component] * 1e-9
            energy_mj = raw_mj * factor  # 各分项等比例缩放，占比保持不变
            component_rows.append(
                {
                    "model_variant": variant,
                    "component_group": component,
                    "energy_mj": f"{energy_mj:.6f}",
                    "share_percent": f"{100.0 * energy_mj / final_total_mj if final_total_mj else 0.0:.4f}",
                    "raw_energy_mj": f"{raw_mj:.9f}",
                    "normalization_factor": f"{factor:.6f}",
                    "normalization_target_mj": f"{target:.6f}" if target is not None else "",
                    "evidence_label": (
                        source_label
                        + "; normalized to draft HPAT energy envelope"
                        if normalize_to_envelope
                        else source_label
                    ),
                    "activity_source": activity_source,
                    "unit_cost_status": unit_cost_status,
                    "evidence_tier": evidence_tier,
                }
            )
    return component_rows, per_layer_rows


# 组件级能耗表的列名清单（顺序即 CSV 表头顺序）
ENERGY_COMPONENT_FIELDS = [
    "model_variant",
    "component_group",
    "energy_mj",
    "share_percent",
    "raw_energy_mj",
    "normalization_factor",
    "normalization_target_mj",
    "evidence_label",
    "activity_source",
    "unit_cost_status",
    "evidence_tier",
]


# 层级能耗表的列名清单
PER_LAYER_ENERGY_FIELDS = [
    "model_variant",
    "layer_id",
    "op_type",
    "optical_path_mj_raw",
    "memory_control_electronic_mj_raw",
    "evidence_label",
    "activity_source",
    "unit_cost_status",
    "evidence_tier",
]
