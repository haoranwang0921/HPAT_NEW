"""P1 证据管线：用本地仿真/建模代理支撑"架构级"讨论（非硅片实测）。

背景：论文证据三档 P0/P1/P2 中，P1 的定位是"比纯配置推导更进一步，
但还没有真实硬件实测"。本文件产出的是"本地/建模的 P1 代理"——
用 torch/timm hook 抓的真实算子轨迹 + 解析物理模型（波长、PDPU 组、
Tile 数、MRR 面积、路径损耗、激光功率等），估出流量、延迟、能耗的
代理值。所有输出都带 P1_CLAIM_BOUNDARY 声明边界：只支撑"有边界的
架构级开销讨论"，不支撑实测加速比、硅片鲁棒性、端侧部署或最终
物理设计能耗声明。

主要功能：
- unit_cost_rows：把带完整元信息（低/高区间、来源）的单位成本载荷
  展开成表格行（比 energy_model 的更严格：value/low/high 必须齐全）；
- qkv_calibrated_rows：用 hook 轨迹"校准"QKV 流量（优先用真实轨迹
  推算字节量，缺轨迹时退回配置代理）；
- architecture_ablation_rows：架构消融——逐项关闭/改变设计假设
  （广播、权重驻留、波长数、位宽、校准间隔等），看延迟/能耗怎么变；
- scalability_physical_proxy_rows：物理代理规模扩展——加入 MRR 面积、
  转换器面积、路径损耗、激光功率、热密度等物理量，判断资源配置
  是否越界。

易混淆点：P1 ≠ P2。P2（p2.py）进一步用"物理代理/仿真"（如真实
DNN 推理 + 误差注入后的精度统计）给出更强证据；P1 主要是解析建模。
"""

from __future__ import annotations

import json
import pathlib
import hashlib
from typing import Any

from .energy_model import UNIT_COST_REQUIRED_KEYS
from .mobilevit_loader import primary_token_count, variants_from_config


# P1 证据标签与声明边界（写进每张表，提醒读者这仍是建模/代理证据）
P1_EVIDENCE_LABEL = "local/modelled P1 proxy; not silicon, not edge deployment, not physical layout closure"
P1_CLAIM_BOUNDARY = (
    "Supports bounded architecture-level overhead discussion only. "
    "Does not support measured HPAT speedup, fabricated-silicon robustness, "
    "edge deployment, or final calibrated physical-design energy claims."
)


# ---- 各输出表的列名清单 ----
# QKV 校准流量表：优先用轨迹数据，逐项给出各环节的字节量
QKV_CALIBRATED_FIELDS = [
    "variant",
    "timm_model",
    "input_resolution",
    "token_source",
    "n_tokens",
    "embedding_dim",
    "bit_width",
    "weight_mode",
    "bus_width_bits",
    "memory_hierarchy",
    "reuse_interval",
    "activation_input_bytes",
    "qkv_projection_output_bytes",
    "attention_score_bytes",
    "value_product_bytes",
    "electronic_remainder_bytes",
    "weight_stream_bytes",
    "programming_bytes",
    "bus_bytes",
    "total_bytes",
    "total_kib",
    "programming_share_percent",
    "traceability_source",
    "activity_source_status",
    "trace_source",
    "trace_row_count",
    "source_layer_names",
    "source_rows_sha256",
    "evidence_label",
    "claim_boundary",
]

# 单位成本表（带低/高区间与完整元信息）
UNIT_COST_FIELDS = [
    "cost_key",
    "value",
    "low",
    "high",
    "unit",
    "source",
    "technology_assumption",
    "precision",
    "scope_note",
    "evidence_status",
    "source_id",
    "source_url_or_doi",
    "source_role",
    "numeric_basis",
    "low_multiplier",
    "high_multiplier",
    "claim_boundary",
]

# 单位成本来源台账表
UNIT_COST_SOURCE_LEDGER_FIELDS = [
    "source_id",
    "title",
    "source_url_or_doi",
    "source_role",
    "applies_to_cost_keys",
    "numeric_basis",
    "claim_boundary",
]

# 架构消融表：每个消融场景一行
ARCHITECTURE_ABLATION_FIELDS = [
    "variant",
    "ablation",
    "broadcast_enabled",
    "weight_mode",
    "wavelengths",
    "pdpu_banks",
    "tiles",
    "bit_width",
    "electronic_remainder_included",
    "calibration_interval_inferences",
    "qkv_total_kib",
    "mrr_count_proxy",
    "latency_proxy_ns",
    "energy_proxy_mj",
    "delta_latency_vs_baseline_percent",
    "delta_energy_vs_baseline_percent",
    "bottleneck",
    "evidence_label",
    "claim_boundary",
]

# 物理代理规模扩展表：每个硬件配置组合一行
SCALABILITY_PHYSICAL_PROXY_FIELDS = [
    "variant",
    "n_tokens",
    "embedding_dim",
    "wavelengths",
    "pdpu_banks",
    "tiles",
    "wavelength_budget",
    "wavelength_budget_status",
    "mrr_count_proxy",
    "adc_count_proxy",
    "dac_count_proxy",
    "memory_bandwidth_demand_gbps",
    "memory_bandwidth_budget_gbps",
    "thermal_calibration_multiplier",
    "thermal_tuning_proxy",
    "mrr_area_proxy_mm2",
    "converter_area_proxy_mm2",
    "path_loss_db_proxy",
    "laser_power_proxy_mw",
    "thermal_density_proxy",
    "resource_limit_status",
    "latency_estimate_ns",
    "energy_estimate_mj",
    "bottleneck",
    "proxy_status",
    "evidence_label",
    "claim_boundary",
]

# 单位成本键 → 假设族 的映射（用于不确定性网格归并）
UNIT_COST_FAMILIES = {
    "optical": ["optical_source_pj_per_ns"],
    "converters": [
        "dac_sample_pj",
        "input_mod_sample_pj",
        "pd_sample_pj",
        "tia_sample_pj",
        "sample_hold_pj",
        "adc_sample_pj",
    ],
    "memory_bus": ["memory_read_pj_per_byte", "memory_write_pj_per_byte", "bus_pj_per_byte"],
    "thermal_calibration": [
        "mrr_program_pj",
        "mrr_hold_pj_per_ring_ns",
        "thermal_pj_per_ring_ns",
        "calibration_pj_per_event",
    ],
    "digital_remainder": ["control_pj_per_event", "nonlinear_pj_per_op"],
}


def load_json_compatible_yaml(path: pathlib.Path) -> dict[str, Any]:
    """读取一个"JSON 兼容"的配置文件（本仓库约定用 JSON 而非 YAML）。

    :param path: 配置文件路径。
    :return: 解析出的字典。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unit_cost_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """把"P1 严格版单位成本载荷"展开成表格行。

    与 energy_model 的宽松版本不同，这里强制要求每个成本项都是对象，
    且 value/low/high/unit/source 等 13 个元字段全部非空、value≤low≤high
    且三者都为正数。这样每一行能耗数字都能一路溯源到文献/数据来源。

    :param payload: 单位成本载荷（含 "costs" 对象）。
    :return: 单位成本表（每成本键一行）。
    :raises ValueError: 缺键、元信息不全或数值非法时抛出。
    """
    costs = payload.get("costs", {})
    if not isinstance(costs, dict):
        raise ValueError("unit cost payload must contain a 'costs' object")

    rows: list[dict[str, Any]] = []
    missing = [key for key in UNIT_COST_REQUIRED_KEYS if key not in costs]
    if missing:
        raise ValueError("missing required unit-cost keys: " + ", ".join(missing))

    required_meta = [  # 每个成本项必须带的 13 个元信息字段
        "value",
        "low",
        "high",
        "unit",
        "source",
        "technology_assumption",
        "precision",
        "scope_note",
        "evidence_status",
        "source_id",
        "source_url_or_doi",
        "source_role",
        "numeric_basis",
    ]
    for key in UNIT_COST_REQUIRED_KEYS:
        record = costs[key]
        if not isinstance(record, dict):
            raise ValueError(f"{key}: unit-cost record must be an object")
        missing_meta = [field for field in required_meta if record.get(field) in ("", None)]
        if missing_meta:
            raise ValueError(f"{key}: missing metadata fields: {', '.join(missing_meta)}")
        value = float(record["value"])  # 名义值
        low = float(record["low"])      # 低估值
        high = float(record["high"])    # 高估值
        if value <= 0 or low <= 0 or high <= 0:
            raise ValueError(f"{key}: value, low, and high must be positive")
        if low > high:
            raise ValueError(f"{key}: low must be <= high")
        rows.append(
            {
                "cost_key": key,
                "value": f"{value:.10g}",
                "low": f"{low:.10g}",
                "high": f"{high:.10g}",
                "unit": str(record["unit"]),
                "source": str(record["source"]),
                "technology_assumption": str(record["technology_assumption"]),
                "precision": str(record["precision"]),
                "scope_note": str(record["scope_note"]),
                "evidence_status": str(record["evidence_status"]),
                "source_id": str(record["source_id"]),
                "source_url_or_doi": str(record["source_url_or_doi"]),
                "source_role": str(record["source_role"]),
                "numeric_basis": str(record["numeric_basis"]),
                "low_multiplier": f"{low / value:.6f}",   # 低估值相对名义的倍数
                "high_multiplier": f"{high / value:.6f}", # 高估值相对名义的倍数
                "claim_boundary": P1_CLAIM_BOUNDARY,
            }
        )
    return rows


def unit_cost_source_ledger_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从单位成本载荷的 source_ledger 生成"来源台账"表。

    每条来源记录它覆盖了哪些成本键、取数依据是什么，形成可审计的
    引用清单。

    :param payload: 单位成本载荷（含 "source_ledger" 列表）。
    :return: 来源台账表。
    :raises ValueError: source_ledger 不是列表时抛出。
    """
    sources = payload.get("source_ledger", [])
    if not isinstance(sources, list):
        raise ValueError("unit cost payload source_ledger must be a list")
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source_ledger entries must be objects")
        applies = source.get("applies_to_cost_keys", [])
        # 多个键用分号拼成一个字符串，便于存 CSV
        if isinstance(applies, list):
            applies_text = ";".join(str(item) for item in applies)
        else:
            applies_text = str(applies)
        rows.append(
            {
                "source_id": str(source.get("source_id", "")),
                "title": str(source.get("title", "")),
                "source_url_or_doi": str(source.get("source_url_or_doi", "")),
                "source_role": str(source.get("source_role", "")),
                "applies_to_cost_keys": applies_text,
                "numeric_basis": str(source.get("numeric_basis", "")),
                "claim_boundary": P1_CLAIM_BOUNDARY,
            }
        )
    return rows


def uncertainty_grid_from_unit_cost_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[float]]:
    """从单位成本行的低/高区间，推导每个假设族的"不确定系数网格"。

    对每个族：收集族内所有成本项的 low/value 与 high/value 比值，
    取 最低比值、1.0、最高比值 三个档作为该族的扫描系数；
    若族内没有可用数据，退回 config 里 energy_uncertainty 的默认档。

    :param rows: unit_cost_rows 的输出。
    :param config: 实验配置。
    :return: {族名: [低, 名义1.0, 高]} 的网格。
    """
    by_key = {row["cost_key"]: row for row in rows}
    fallback = config.get("energy_uncertainty", {})
    grid: dict[str, list[float]] = {}
    for family, keys in UNIT_COST_FAMILIES.items():
        low_ratios = [float(by_key[key]["low"]) / float(by_key[key]["value"]) for key in keys if key in by_key]
        high_ratios = [float(by_key[key]["high"]) / float(by_key[key]["value"]) for key in keys if key in by_key]
        if not low_ratios or not high_ratios:
            grid[family] = [float(v) for v in fallback.get(family, [1.0])]  # 无数据退回默认
            continue
        low = min(low_ratios)   # 族内最低比值（最乐观）
        high = max(high_ratios) # 族内最高比值（最悲观）
        values = sorted({round(low, 6), 1.0, round(high, 6)})  # 三档：低/名义/高
        grid[family] = values
    return grid


def _qkv_bits(
    *,
    n_tokens: int,
    embedding_dim: int,
    bit_width: int,
    mode: dict[str, Any],
    bus_multiplier: float,
    broadcast_enabled: bool = True,
    electronic_remainder_multiplier: float = 1.0,
) -> dict[str, int]:
    """按解析公式计算 QKV 各环节的比特数（配置代理路径）。

    各环节：
    - activation_bits：输入激活比特；broadcast_enabled 时 Q/K/V 共用
      同一份输入（×1），否则每个投影各带一份输入（×3）；
    - qkv_output_bits：Q/K/V 三路投影输出；
    - attention_score_bits：注意力分数矩阵（n×n）；
    - value_product_bits：Attention@V 输出（n×d）；
    - electronic_remainder_bits：电子端算子搬运；
    - weight_bits：三组权重矩阵（3·d·d）；
    - programming_bits：权重编程（仅 reprogrammed 模式）；
    - weight_stream_bits：权重流式搬入（仅 streamed 模式）；
    - bus_bits：总线搬运 = 除编程外各环节之和 × bus_multiplier。

    :param n_tokens: token 数。
    :param embedding_dim: 嵌入维度 d。
    :param bit_width: 位宽。
    :param mode: 权重模式字典（name/programming_events/stream_weights）。
    :param bus_multiplier: 总线放大系数。
    :param broadcast_enabled: 是否开启输入广播复用。
    :param electronic_remainder_multiplier: 电子端算子搬运系数。
    :return: 各环节比特数的字典。
    """
    activation_multiplier = 1 if broadcast_enabled else 3  # 广播复用开关
    activation_bits = activation_multiplier * n_tokens * embedding_dim * bit_width
    qkv_output_bits = 3 * n_tokens * embedding_dim * bit_width
    attention_score_bits = n_tokens * n_tokens * bit_width
    value_product_bits = n_tokens * embedding_dim * bit_width
    electronic_remainder_bits = int(electronic_remainder_multiplier * 2 * n_tokens * embedding_dim * bit_width)
    weight_bits = 3 * embedding_dim * embedding_dim * bit_width
    programming_bits = int(mode.get("programming_events", 0)) * weight_bits
    weight_stream_bits = weight_bits if mode.get("stream_weights", False) else 0
    bus_bits = int(
        (
            activation_bits
            + qkv_output_bits
            + attention_score_bits
            + value_product_bits
            + electronic_remainder_bits
            + weight_stream_bits
        )
        * bus_multiplier
    )
    total_bits = (
        activation_bits
        + qkv_output_bits
        + attention_score_bits
        + value_product_bits
        + electronic_remainder_bits
        + weight_stream_bits
        + programming_bits
        + bus_bits
    )
    return {
        "activation_bits": activation_bits,
        "qkv_output_bits": qkv_output_bits,
        "attention_score_bits": attention_score_bits,
        "value_product_bits": value_product_bits,
        "electronic_remainder_bits": electronic_remainder_bits,
        "weight_stream_bits": weight_stream_bits,
        "programming_bits": programming_bits,
        "bus_bits": bus_bits,
        "total_bits": total_bits,
    }


def _precision_bits(row: dict[str, Any]) -> int:
    """从算子行的 precision 字段换算位宽（fp16→16、int8→8、其余→32）。

    :param row: 算子行。
    :return: 位宽（整数）。
    """
    precision = str(row.get("precision") or "fp32").lower()
    if precision in {"fp16", "float16", "bf16"}:
        return 16
    if precision in {"int8", "uint8"}:
        return 8
    return 32


def _scaled_bytes(row: dict[str, Any], key: str, bit_width: int) -> int:
    """把算子行里某字段的字节数按位宽等比缩放到目标位宽。

    例：fp32（4 字节/元素）下 100 字节，换算到 8 位（1 字节/元素）
    得 100×8/32 = 25 字节。

    :param row: 算子行。
    :param key: 字节字段名。
    :param bit_width: 目标位宽。
    :return: 缩放后的字节数。
    """
    value = float(row.get(key) or 0.0)
    source_bits = max(_precision_bits(row), 1)
    return int(round(value * bit_width / source_bits))


def _source_hash(rows: list[dict[str, Any]]) -> str:
    """对一组源行做 SHA-256，用于溯源校验（轨迹数据有没有被改过）。

    :param rows: 源行列表。
    :return: 十六进制哈希。
    """
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trace_activity_by_variant(operator_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把 torch hook 抓的算子行按模型变体分组，并按算子组分类。

    只保留 trace_source == "torch_hooks" 的真实轨迹行；每个变体桶里
    记录 rows（全部）、qkv（QKV 投影行）、attention_score、attention_value、
    electronic（电子端行），以及 tokens/embedding_dim/分辨率等汇总信息。

    :param operator_rows: 算子行列表。
    :return: {变体名: 分组桶字典}。
    """
    by_variant: dict[str, dict[str, Any]] = {}
    for row in operator_rows:
        if row.get("trace_source") != "torch_hooks":
            continue  # 跳过配置代理行，只用真实 hook 轨迹
        model = str(row.get("model", ""))
        if not model:
            continue
        bucket = by_variant.setdefault(
            model,
            {
                "rows": [],
                "qkv": [],
                "attention_score": [],
                "attention_value": [],
                "electronic": [],
                "tokens": 0,
                "embedding_dim": 0,
                "input_resolution": "",
                "timm_model": "",
            },
        )
        bucket["rows"].append(row)
        # 同一变体不同层可能记录不同 token/维度，取最大值作为代表
        bucket["tokens"] = max(int(float(row.get("tokens") or 0)), int(bucket["tokens"]))
        bucket["embedding_dim"] = max(int(float(row.get("embedding_dim") or 0)), int(bucket["embedding_dim"]))
        bucket["timm_model"] = row.get("model_name") or bucket["timm_model"]
        shape = str(row.get("input_shape") or "")
        if shape and not bucket["input_resolution"]:
            bucket["input_resolution"] = shape.split("x")[-1]  # 分辨率 = 形状最后一维
        # 按算子组归类到对应子列表
        op_group = row.get("op_group")
        if op_group == "qkv_projection":
            bucket["qkv"].append(row)
        elif op_group == "attention_score":
            bucket["attention_score"].append(row)
        elif op_group == "attention_value":
            bucket["attention_value"].append(row)
        elif row.get("execution_domain") == "electronic remainder":
            bucket["electronic"].append(row)
    return by_variant


def qkv_calibrated_rows(
    config: dict[str, Any],
    operator_rows: list[dict[str, Any]] | None = None,
    trace_mode: str = "auto",
) -> list[dict[str, Any]]:
    """核心：生成"轨迹校准版"QKV 流量表（P1 的主要证据之一）。

    两条路径：
    - 轨迹路径（use_trace=True）：有 torch hook 轨迹时，用真实算子行的
      字节量逐项汇总，字节量按目标位宽等比缩放，并记录来源层名列表与
      SHA-256 哈希，保证可溯源；
    - 配置代理路径：没有轨迹时退回 _qkv_bits 解析公式。

    trace_mode 取值："auto"（有轨迹用轨迹，否则退回配置）、"trace"
    （强制要求轨迹，没有就标注"trace-required-missing"）。

    :param config: 实验配置。
    :param operator_rows: 算子行列表（可为空）。
    :param trace_mode: 轨迹使用策略。
    :return: QKV 流量表（每 (变体, token, 位宽, 权重模式) 一行）。
    """
    qcfg = config.get("qkv", {})
    modes = qcfg.get("weight_modes") or [
        {"name": "resident", "programming_events": 0, "stream_weights": False, "reuse_interval": "layer"},
        {"name": "reprogrammed", "programming_events": 1, "stream_weights": False, "reuse_interval": "layer"},
    ]
    bus_multiplier = float(qcfg.get("bus_multiplier", 1.0))
    bus_width_bits = int(qcfg.get("bus_width_bits", 128))
    memory_hierarchy = str(qcfg.get("memory_hierarchy", "local buffer plus electrical bus proxy"))
    electronic_remainder_multiplier = float(qcfg.get("electronic_remainder_multiplier", 1.0))
    trace_by_variant = _trace_activity_by_variant(operator_rows or [])
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])
        token_sweep = variant.get("token_count_sweep") or [primary_token_count(variant, config)]
        trace_bucket = trace_by_variant.get(variant["variant"], {})
        use_trace = trace_mode in {"auto", "trace"} and bool(trace_bucket.get("rows"))
        if use_trace:
            # 有轨迹：token 数直接采用轨迹里的实际值
            token_sweep = [int(trace_bucket.get("tokens") or primary_token_count(variant, config))]
        if trace_mode == "trace" and not use_trace:
            token_sweep = [primary_token_count(variant, config)]  # 强制 trace 但没轨迹：退回配置 token
        for n_tokens in token_sweep:
            for bit_width in config.get("bit_width_sweep", [8, 10, 12]):
                for mode in modes:
                    if use_trace:
                        # ---- 轨迹路径：逐项累加真实轨迹的字节量（×8 转比特）----
                        source_rows = list(trace_bucket.get("rows", []))
                        qkv_rows = list(trace_bucket.get("qkv", []))
                        attention_rows = list(trace_bucket.get("attention_score", []))
                        value_rows = list(trace_bucket.get("attention_value", []))
                        electronic_rows = list(trace_bucket.get("electronic", []))
                        activation_bits = sum(_scaled_bytes(row, "input_bytes", int(bit_width)) for row in qkv_rows) * 8
                        qkv_output_bits = sum(_scaled_bytes(row, "output_bytes", int(bit_width)) for row in qkv_rows) * 8
                        attention_score_bits = sum(_scaled_bytes(row, "output_bytes", int(bit_width)) for row in attention_rows) * 8
                        value_product_bits = sum(_scaled_bytes(row, "output_bytes", int(bit_width)) for row in value_rows) * 8
                        electronic_remainder_bits = int(
                            electronic_remainder_multiplier
                            * sum(
                                _scaled_bytes(row, "input_bytes", int(bit_width))
                                + _scaled_bytes(row, "output_bytes", int(bit_width))
                                for row in electronic_rows
                            )
                            * 8
                        )
                        weight_bits = 3 * d * d * int(bit_width)  # 权重规模固定公式
                        programming_bits = int(mode.get("programming_events", 0)) * weight_bits
                        weight_stream_bits = weight_bits if mode.get("stream_weights", False) else 0
                        bus_bits = sum(  # 总线流量 = 全部源行的输入/输出/权重字节 ×8
                            _scaled_bytes(row, "input_bytes", int(bit_width))
                            + _scaled_bytes(row, "output_bytes", int(bit_width))
                            + _scaled_bytes(row, "weight_bytes", int(bit_width))
                            for row in source_rows
                        ) * 8
                        bits = {
                            "activation_bits": activation_bits,
                            "qkv_output_bits": qkv_output_bits,
                            "attention_score_bits": attention_score_bits,
                            "value_product_bits": value_product_bits,
                            "electronic_remainder_bits": electronic_remainder_bits,
                            "weight_stream_bits": weight_stream_bits,
                            "programming_bits": programming_bits,
                            "bus_bits": bus_bits,
                            "total_bits": activation_bits
                            + qkv_output_bits
                            + attention_score_bits
                            + value_product_bits
                            + electronic_remainder_bits
                            + weight_stream_bits
                            + programming_bits
                            + bus_bits,
                        }
                        row_tokens = int(trace_bucket.get("tokens") or n_tokens)
                        row_d = int(trace_bucket.get("embedding_dim") or d)
                        trace_status = "trace-backed local"
                        trace_source = "torch_hooks"
                        trace_row_count = len(source_rows)
                        source_layer_names = ";".join(  # 参与的层名（最多 24 个）
                            sorted({str(row.get("layer_name", "")) for row in source_rows if row.get("layer_name")})[:24]
                        )
                        source_rows_sha256 = _source_hash(source_rows)
                        traceability_source = "torch/timm operator activity trace; simulator export still required for physical-design promotion"
                    else:
                        # ---- 配置代理路径：解析公式 ----
                        bits = _qkv_bits(
                            n_tokens=int(n_tokens),
                            embedding_dim=d,
                            bit_width=int(bit_width),
                            mode=mode,
                            bus_multiplier=bus_multiplier,
                            electronic_remainder_multiplier=electronic_remainder_multiplier,
                        )
                        row_tokens = int(n_tokens)
                        row_d = d
                        trace_status = (
                            "trace-required-missing; config proxy emitted"  # 要求轨迹但缺失
                            if trace_mode == "trace"
                            else "config-derived proxy"
                        )
                        trace_source = "config_proxy"
                        trace_row_count = 0
                        source_layer_names = ""
                        source_rows_sha256 = ""
                        traceability_source = "experiments/config/hpat_experiment_config.json mobilevit_variants + qkv"
                    total_bytes = bits["total_bits"] / 8.0
                    rows.append(
                        {
                            "variant": variant["variant"],
                            "timm_model": trace_bucket.get("timm_model") or variant.get("timm_model", ""),
                            "input_resolution": trace_bucket.get("input_resolution") or int(variant.get("input_resolution", 0)),
                            "token_source": "torch/timm operator trace" if use_trace else "configured token_count_sweep / qkv primary token count",
                            "n_tokens": row_tokens,
                            "embedding_dim": row_d,
                            "bit_width": int(bit_width),
                            "weight_mode": mode["name"],
                            "bus_width_bits": bus_width_bits,
                            "memory_hierarchy": memory_hierarchy,
                            "reuse_interval": mode.get("reuse_interval", ""),
                            "activation_input_bytes": int(bits["activation_bits"] / 8),
                            "qkv_projection_output_bytes": int(bits["qkv_output_bits"] / 8),
                            "attention_score_bytes": int(bits["attention_score_bits"] / 8),
                            "value_product_bytes": int(bits["value_product_bits"] / 8),
                            "electronic_remainder_bytes": int(bits["electronic_remainder_bits"] / 8),
                            "weight_stream_bytes": int(bits["weight_stream_bits"] / 8),
                            "programming_bytes": int(bits["programming_bits"] / 8),
                            "bus_bytes": int(bits["bus_bits"] / 8),
                            "total_bytes": int(total_bytes),
                            "total_kib": f"{total_bytes / 1024.0:.4f}",
                            "programming_share_percent": f"{100.0 * bits['programming_bits'] / bits['total_bits'] if bits['total_bits'] else 0.0:.4f}",
                            "traceability_source": traceability_source,
                            "activity_source_status": trace_status,
                            "trace_source": trace_source,
                            "trace_row_count": trace_row_count,
                            "source_layer_names": source_layer_names,
                            "source_rows_sha256": source_rows_sha256,
                            "evidence_label": "trace-backed local P1 Q/K/V traffic; not silicon, not edge deployment, not physical layout closure"
                            if use_trace
                            else P1_EVIDENCE_LABEL,
                            "claim_boundary": P1_CLAIM_BOUNDARY,
                        }
                    )
    return rows


def _scenario_metrics(
    config: dict[str, Any],
    variant: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """按一个"架构场景"计算代理指标（流量/延迟/能耗/瓶颈）。

    延迟分解（纳秒）：光计算 + 转换器 + 内存 + 热校准 + 周期校准；
    能耗：基准能耗 × 各硬件规模/位宽/权重策略的加性修正系数。

    :param config: 实验配置。
    :param variant: 模型变体规格。
    :param scenario: 架构场景字典（见 architecture_ablation_rows 的 cases）。
    :return: {"qkv_total_kib","mrr_count_proxy","latency_proxy_ns",
              "energy_proxy_mj","bottleneck"} 的指标字典。
    """
    d = int(variant["embedding_dim"])
    n = primary_token_count(variant, config)
    bit_width = int(scenario["bit_width"])
    wavelengths = int(scenario["wavelengths"])
    pdpu_banks = int(scenario["pdpu_banks"])
    tiles = int(scenario["tiles"])
    calibration_interval = max(int(scenario["calibration_interval_inferences"]), 1)
    mode = scenario["mode"]
    electronic_multiplier = 1.0 if scenario["electronic_remainder_included"] else 0.0
    qcfg = config.get("qkv", {})
    qkv = _qkv_bits(
        n_tokens=n,
        embedding_dim=d,
        bit_width=bit_width,
        mode=mode,
        bus_multiplier=float(qcfg.get("bus_multiplier", 1.0)),
        broadcast_enabled=bool(scenario["broadcast_enabled"]),
        electronic_remainder_multiplier=electronic_multiplier,
    )

    base_ops = 3 * n * d * d + 2 * n * n * d + 2 * n * d * (2 * d)  # 一次前向总操作数
    optical_parallel = max(wavelengths * pdpu_banks * tiles, 1)
    compute_ns = base_ops / (optical_parallel * float(config.get("activity_proxy", {}).get("ops_per_ns", 250000)))
    converter_ns = (4 * n * d) / max(int(scenario.get("adc_parallelism", 16)), 1) * 0.25
    memory_bandwidth_gbps = float(scenario.get("memory_bandwidth_gbps", 128.0))
    memory_ns = qkv["total_bits"] / memory_bandwidth_gbps
    thermal_ns = 40.0 * float(scenario.get("thermal_multiplier", 1.0)) * max(1.0, pdpu_banks / 2.0)
    calibration_ns = 60000.0 / calibration_interval  # 每次校准摊还到单次推理的时间
    if not scenario["broadcast_enabled"]:
        compute_ns *= 1.08  # 无广播复用 → 光计算慢 8%（要多算两次投影）
    latency_ns = compute_ns + converter_ns + memory_ns + thermal_ns + calibration_ns
    # 瓶颈 = 五项里最大的一项
    bottleneck = max(
        [
            ("optical_compute", compute_ns),
            ("adc_dac_parallelism", converter_ns),
            ("memory_bandwidth", memory_ns),
            ("thermal_calibration", thermal_ns + calibration_ns),
        ],
        key=lambda item: item[1],
    )[0]

    base_energy = float(config.get("energy_envelope_mj", {}).get(variant["variant"], 1.0))
    energy_multiplier = 1.0
    energy_multiplier += 0.012 * wavelengths + 0.055 * pdpu_banks + 0.035 * tiles  # 硬件规模越大越耗电
    energy_multiplier += 0.02 * max(bit_width - 8, 0) - 0.015 * max(8 - bit_width, 0)  # 位宽偏离 8 位的影响
    energy_multiplier += (1000.0 / calibration_interval) * 0.03  # 校准越频繁能耗越高
    if mode.get("name") == "reprogrammed":
        energy_multiplier += 0.18  # 每层重写权重 → +18%
    elif mode.get("name") == "streamed":
        energy_multiplier += 0.08  # 流式搬权重 → +8%
    if not scenario["broadcast_enabled"]:
        energy_multiplier += 0.12
    if not scenario["electronic_remainder_included"]:
        energy_multiplier -= 0.22  # 乐观假设（忽略电子端）→ -22%
    energy_mj = max(base_energy * energy_multiplier, base_energy * 0.25)  # 下限钳制，防过低

    return {
        "qkv_total_kib": qkv["total_bits"] / 8.0 / 1024.0,
        "mrr_count_proxy": wavelengths * d * pdpu_banks * tiles,  # MRR 数量代理指标
        "latency_proxy_ns": latency_ns,
        "energy_proxy_mj": energy_mj,
        "bottleneck": bottleneck,
    }


def architecture_ablation_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """架构消融分析：在基准设计上逐个改变一项假设，看指标如何变化。

    基准（baseline）：16 波长 × 2 PDPU 组 × 2 Tile、8 位精度、
    广播开启、权重常驻、每 1000 次推理校准一次、包含电子端。

    消融项：关广播、改权重策略（流式/重写）、改波长数（8/32）、
    改 PDPU 组数（1/4）、改位宽（4/12）、乐观忽略电子端、改校准间隔。

    输出每个消融相对基准的延迟/能耗变化百分比。

    :param config: 实验配置。
    :return: 消融表（每 (变体, 消融项) 一行）。
    """
    # 三种权重驻留模式
    resident = {"name": "resident", "programming_events": 0, "stream_weights": False}
    reprogrammed = {"name": "reprogrammed", "programming_events": 1, "stream_weights": False}
    streamed = {"name": "streamed", "programming_events": 0, "stream_weights": True}
    baseline = {
        "ablation": "baseline",
        "broadcast_enabled": True,
        "mode": resident,
        "wavelengths": 16,
        "pdpu_banks": 2,
        "tiles": 2,
        "bit_width": 8,
        "electronic_remainder_included": True,
        "calibration_interval_inferences": 1000,
        "adc_parallelism": 16,
        "memory_bandwidth_gbps": 128.0,
        "thermal_multiplier": 1.0,
    }
    # 每个消融项 = 基准 + 一项改动（其余保持基准值）
    cases = [
        baseline,
        {**baseline, "ablation": "no_broadcast", "broadcast_enabled": False},
        {**baseline, "ablation": "streamed_weights", "mode": streamed},
        {**baseline, "ablation": "reprogrammed_weights", "mode": reprogrammed},
        {**baseline, "ablation": "wdm_channels_8", "wavelengths": 8},
        {**baseline, "ablation": "wdm_channels_32", "wavelengths": 32},
        {**baseline, "ablation": "pdpu_banks_1", "pdpu_banks": 1},
        {**baseline, "ablation": "pdpu_banks_4", "pdpu_banks": 4},
        {**baseline, "ablation": "precision_4bit", "bit_width": 4},
        {**baseline, "ablation": "precision_12bit", "bit_width": 12},
        {**baseline, "ablation": "optimistic_optical_only", "electronic_remainder_included": False},
        {**baseline, "ablation": "calibration_interval_100", "calibration_interval_inferences": 100},
        {**baseline, "ablation": "calibration_interval_10000", "calibration_interval_inferences": 10000},
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        base_metrics = _scenario_metrics(config, variant, baseline)  # 先算基准指标
        for case in cases:
            metrics = _scenario_metrics(config, variant, case)
            rows.append(
                {
                    "variant": variant["variant"],
                    "ablation": case["ablation"],
                    "broadcast_enabled": str(bool(case["broadcast_enabled"])).lower(),
                    "weight_mode": case["mode"]["name"],
                    "wavelengths": int(case["wavelengths"]),
                    "pdpu_banks": int(case["pdpu_banks"]),
                    "tiles": int(case["tiles"]),
                    "bit_width": int(case["bit_width"]),
                    "electronic_remainder_included": str(bool(case["electronic_remainder_included"])).lower(),
                    "calibration_interval_inferences": int(case["calibration_interval_inferences"]),
                    "qkv_total_kib": f"{metrics['qkv_total_kib']:.6f}",
                    "mrr_count_proxy": int(metrics["mrr_count_proxy"]),
                    "latency_proxy_ns": f"{metrics['latency_proxy_ns']:.6f}",
                    "energy_proxy_mj": f"{metrics['energy_proxy_mj']:.6f}",
                    "delta_latency_vs_baseline_percent": f"{100.0 * (metrics['latency_proxy_ns'] - base_metrics['latency_proxy_ns']) / base_metrics['latency_proxy_ns']:.4f}",
                    "delta_energy_vs_baseline_percent": f"{100.0 * (metrics['energy_proxy_mj'] - base_metrics['energy_proxy_mj']) / base_metrics['energy_proxy_mj']:.4f}",
                    "bottleneck": metrics["bottleneck"],
                    "evidence_label": P1_EVIDENCE_LABEL,
                    "claim_boundary": P1_CLAIM_BOUNDARY,
                }
            )
    return rows


def scalability_physical_proxy_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """物理代理规模扩展：把硬件规模参数扫一遍，并加入物理约束检查。

    相比 scalability.py（纯延迟/能耗公式），这里额外加入物理量：
    - MRR 面积与转换器面积（μm² → mm²）；
    - 路径损耗（dB）= 基准损耗 + 每 Tile × 每波长损耗；
    - 激光功率代理 = 损耗 × 每 dB 功率 × 组数；
    - 热密度 = 微环数×温度系数 / 总面积；
    - 资源越界检查：波长超过预算、或热密度超限 → proxy_limit_exceeded。

    :param config: 实验配置（读 scalability.physical_proxy 段）。
    :return: 物理代理规模扩展表（每硬件配置一行）。
    """
    scfg = config.get("scalability", {})
    pcfg = scfg.get("physical_proxy", {})
    wavelengths = scfg.get("wavelengths", [8, 16, 32])
    pdpu_banks = scfg.get("pdpu_banks", [1, 2, 4])
    tiles = scfg.get("tiles", [1, 2])
    adc_parallelism = scfg.get("adc_parallelism", [8, 16, 32])
    memory_bandwidth = scfg.get("memory_bandwidth_gbps", [64, 128])
    thermal_multipliers = scfg.get("thermal_calibration_multipliers", [1.0, 1.5])
    wavelength_budget = int(scfg.get("wavelength_budget", max(wavelengths)))  # 波长预算上限
    mrr_area_um2 = float(pcfg.get("mrr_area_um2", 100.0))           # 单 MRR 面积（μm²）
    converter_area_um2 = float(pcfg.get("converter_area_um2", 4000.0))  # 单转换器面积（μm²）
    base_path_loss_db = float(pcfg.get("base_path_loss_db", 2.0))       # 基准路径损耗
    loss_per_tile_db = float(pcfg.get("loss_per_tile_db", 0.25))       # 每 Tile 损耗增量
    loss_per_wavelength_db = float(pcfg.get("loss_per_wavelength_db", 0.01))  # 每波长损耗增量
    laser_mw_per_db = float(pcfg.get("laser_mw_per_db", 0.8))          # 每 dB 损耗需补的激光功率
    thermal_density_limit = float(pcfg.get("thermal_density_limit", 200000.0))  # 热密度上限
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])
        n = primary_token_count(variant, config)
        base_energy = float(config.get("energy_envelope_mj", {}).get(variant["variant"], 1.0))
        base_ops = 3 * n * d * d + 2 * n * n * d + 2 * n * d * (2 * d)
        traffic_bits = (4 * n * d + n * n + 3 * d * d) * int(config.get("qkv", {}).get("default_bit_width", 8))
        for w in wavelengths:
            for bank in pdpu_banks:
                for tile in tiles:
                    for adc in adc_parallelism:
                        for bw in memory_bandwidth:
                            for thermal in thermal_multipliers:
                                optical_parallel = max(int(w) * int(bank) * int(tile), 1)
                                compute_ns = base_ops / (optical_parallel * 250000.0)
                                converter_ns = (4 * n * d) / max(int(adc), 1) * 0.25
                                memory_ns = traffic_bits / float(bw)
                                mrr_count = int(w) * d * int(bank) * int(tile)  # 微环总数代理
                                thermal_ns = 40.0 * float(thermal) * max(1.0, int(bank) / 2.0)
                                latency_ns = compute_ns + converter_ns + memory_ns + thermal_ns
                                demand_gbps = traffic_bits / max(latency_ns, 1e-9)  # 内存带宽需求
                                bottleneck = max(
                                    [
                                        ("optical_compute", compute_ns),
                                        ("adc_dac_parallelism", converter_ns),
                                        ("memory_bandwidth", memory_ns),
                                        ("thermal_calibration", thermal_ns),
                                    ],
                                    key=lambda item: item[1],
                                )[0]
                                energy_multiplier = 1.0 + 0.015 * int(w) + 0.06 * int(bank) + 0.04 * int(tile)
                                energy_multiplier += 0.08 * (float(thermal) - 1.0)
                                converter_count = int(adc) + int(w) * int(bank) * int(tile)  # 转换器总数代理
                                mrr_area_mm2 = mrr_count * mrr_area_um2 / 1_000_000.0   # μm²→mm²
                                converter_area_mm2 = converter_count * converter_area_um2 / 1_000_000.0
                                path_loss_db = base_path_loss_db + loss_per_tile_db * int(tile) + loss_per_wavelength_db * int(w)
                                laser_power_proxy_mw = path_loss_db * laser_mw_per_db * max(1, int(bank))
                                thermal_density = (mrr_count * float(thermal)) / max(mrr_area_mm2 + converter_area_mm2, 1e-9)
                                # 资源越界判断：波长超预算 或 热密度超限
                                resource_status = (
                                    "proxy_limit_exceeded"
                                    if int(w) > wavelength_budget or thermal_density > thermal_density_limit
                                    else "within_declared_proxy_limits"
                                )
                                rows.append(
                                    {
                                        "variant": variant["variant"],
                                        "n_tokens": n,
                                        "embedding_dim": d,
                                        "wavelengths": int(w),
                                        "pdpu_banks": int(bank),
                                        "tiles": int(tile),
                                        "wavelength_budget": wavelength_budget,
                                        "wavelength_budget_status": "within_proxy_budget"
                                        if int(w) <= wavelength_budget
                                        else "exceeds_proxy_budget",
                                        "mrr_count_proxy": mrr_count,
                                        "adc_count_proxy": int(adc),
                                        "dac_count_proxy": int(w) * int(bank) * int(tile),
                                        "memory_bandwidth_demand_gbps": f"{demand_gbps:.6f}",
                                        "memory_bandwidth_budget_gbps": f"{float(bw):.6f}",
                                        "thermal_calibration_multiplier": f"{float(thermal):.6f}",
                                        "thermal_tuning_proxy": f"{mrr_count * float(thermal):.6f}",
                                        "mrr_area_proxy_mm2": f"{mrr_area_mm2:.6f}",
                                        "converter_area_proxy_mm2": f"{converter_area_mm2:.6f}",
                                        "path_loss_db_proxy": f"{path_loss_db:.6f}",
                                        "laser_power_proxy_mw": f"{laser_power_proxy_mw:.6f}",
                                        "thermal_density_proxy": f"{thermal_density:.6f}",
                                        "resource_limit_status": resource_status,
                                        "latency_estimate_ns": f"{latency_ns:.6f}",
                                        "energy_estimate_mj": f"{base_energy * energy_multiplier:.6f}",
                                        "bottleneck": bottleneck,
                                        "proxy_status": "physical_proxy_not_layout_closure",  # 明确：物理代理≠版图收敛
                                        "evidence_label": P1_EVIDENCE_LABEL,
                                        "claim_boundary": P1_CLAIM_BOUNDARY,
                                    }
                                )
    return rows
