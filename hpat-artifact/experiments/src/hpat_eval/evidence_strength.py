"""证据强度（evidence strength）评估：把"能映射多少计算上光子端"
换算成"理论上限加速比"。

核心思想：HPAT 是光子+电子混合架构，只有能被映射进 PDPU（光点积单元）
的那部分计算（矩阵乘法类算子）才享受光加速；剩下的电子端算子
（归一化、softmax、控制等）速度不变。这就引出了阿姆达尔定律
（Amdahl's law）：整体加速比上界 = 1 / (电子端占比 + 光子端占比/光加速比)。
即使光加速比为无穷大，整体加速比也最多是 1/电子端占比。

本文件据此生成：
- 算子域汇总表（每个模型有多少 MAC 能上 PDPU、多少留在电子端）；
- 加速比上界表（按不同光加速比假设扫描）；
- 三档映射场景表（保守/合理/最大，见 MAPPING_SCENARIOS）；
- 能耗不确定性分析（把能耗分项按 5 个族各乘一个不确定系数，扫网格）。

易混淆点：这里算的是"设计上界"（bound），不是实测加速比——
所有输出都带着 claim_boundary / evidence_label 声明，提醒读者别把
模拟或上界当成硅片实测。
"""

from __future__ import annotations

from typing import Any


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


def operator_domain_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按模型汇总算子清单，输出"算子执行域"分布表。

    每个算子行按 execution_domain 分成三类：PDPU-candidate linear/MVM
    （能上光子端）、electronic remainder（电子端）、其余归为 hybrid
    support（混合支撑）。同时汇总 MAC 数、输入/输出/权重字节量、
    非线性算子数，并按模型排序输出。

    :param rows: 算子清单行（见 injection_boundary 相关数据结构）。
    :return: 每模型一行的汇总表。
    """
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        # 每个模型一个累计桶，首次见到时初始化全部统计字段
        bucket = by_model.setdefault(
            model,
            {
                "model": model,
                "backend": row.get("backend", ""),          # 后端标识（如 torch/timm）
                "trace_source": row.get("trace_source", ""), # 轨迹来源（hook 抓的）
                "total_rows": 0,          # 算子行总数
                "total_macs": 0.0,        # MAC（乘加）总次数
                "pdpu_candidate_macs": 0.0,     # 光子端候选 MAC
                "electronic_remainder_macs": 0.0,  # 电子端 MAC
                "hybrid_support_macs": 0.0,       # 混合支撑 MAC
                "total_input_bytes": 0.0,   # 输入字节总量
                "total_output_bytes": 0.0,  # 输出字节总量
                "total_weight_bytes": 0.0,  # 权重字节总量
                "nonlinear_ops": 0.0,       # 非线性算子次数
            },
        )
        macs = _f(row, "estimated_macs")  # 该算子估计的 MAC 数
        bucket["total_rows"] += 1
        bucket["total_macs"] += macs
        bucket["total_input_bytes"] += _f(row, "input_bytes")
        bucket["total_output_bytes"] += _f(row, "output_bytes")
        bucket["total_weight_bytes"] += _f(row, "weight_bytes")
        bucket["nonlinear_ops"] += _f(row, "nonlinear_ops")
        # 按执行域把 MAC 分到对应桶
        domain = row.get("execution_domain", "")
        if domain == "PDPU-candidate linear/MVM":
            bucket["pdpu_candidate_macs"] += macs
        elif domain == "electronic remainder":
            bucket["electronic_remainder_macs"] += macs
        else:
            bucket["hybrid_support_macs"] += macs

    out: list[dict[str, Any]] = []
    order = {"MobileViT-XXS": 0, "MobileViT-XS": 1, "MobileViT-S": 2}  # 论文固定展示顺序
    for model, bucket in sorted(by_model.items(), key=lambda item: order.get(item[0], 99)):
        total = bucket["total_macs"] or 1.0  # 防除零
        electronic = bucket["electronic_remainder_macs"] + bucket["hybrid_support_macs"]  # 电子端=剩余+混合
        out.append(
            {
                "model": model,
                "backend": bucket["backend"],
                "trace_source": bucket["trace_source"],
                "total_rows": bucket["total_rows"],
                "total_macs": f"{bucket['total_macs']:.0f}",
                "pdpu_candidate_macs": f"{bucket['pdpu_candidate_macs']:.0f}",
                "electronic_remainder_macs": f"{electronic:.0f}",
                "pdpu_candidate_mac_share_percent": f"{100.0 * bucket['pdpu_candidate_macs'] / total:.4f}",
                "electronic_remainder_mac_share_percent": f"{100.0 * electronic / total:.4f}",
                "total_input_mib": f"{bucket['total_input_bytes'] / 1024.0 / 1024.0:.6f}",  # 字节→MiB
                "total_output_mib": f"{bucket['total_output_bytes'] / 1024.0 / 1024.0:.6f}",
                "total_weight_mib": f"{bucket['total_weight_bytes'] / 1024.0 / 1024.0:.6f}",
                "nonlinear_ops": f"{bucket['nonlinear_ops']:.0f}",
                "evidence_label": "local torch/timm hook trace summary; not edge evidence",  # 明确：本地轨迹，非端侧实测
            }
        )
    return out


def speedup_bound_rows(summary_rows: list[dict[str, Any]], optical_speedups: list[float]) -> list[dict[str, Any]]:
    """基于算子域汇总表，用阿姆达尔定律计算端到端加速比上界。

    公式：整体加速比上界 = 1 / (电子端占比 + 光子端占比 / 光加速比)。
    其中"电子端占比"包括 electronic remainder + hybrid support 两部分。

    :param summary_rows: operator_domain_summary_rows 的输出。
    :param optical_speedups: 要扫描的光加速比假设列表（如 [10, 100]）。
    :return: 每个 (模型, 光加速比) 一行的上界表。
    """
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        f_elec = float(row["electronic_remainder_mac_share_percent"]) / 100.0  # 电子端 MAC 占比
        for speedup in optical_speedups:
            # 阿姆达尔公式；光加速比 <=0 视为无加速（上界=1）
            bound = 1.0 / (f_elec + (1.0 - f_elec) / float(speedup)) if speedup > 0 else 1.0
            rows.append(
                {
                    "model": row["model"],
                    "electronic_fraction": f"{f_elec:.8f}",
                    "pdpu_candidate_fraction": f"{1.0 - f_elec:.8f}",
                    "optical_speedup_assumption": f"{float(speedup):.6f}",
                    "e2e_speedup_upper_bound": f"{bound:.6f}",
                    "evidence_label": (
                        "Conservative traced-Linear baseline Amdahl bound; "
                        "not the paper-facing reasonable-high boundary and not measured HPAT speedup"
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 三档映射场景：从保守到激进，映射边界逐级放宽
# ---------------------------------------------------------------------------
# - linear_only：只映射抓到的 Linear 层（最保守，纯轨迹基线）；
# - plus_pointwise_attention：追加 1×1 卷积与注意力矩阵乘（论文对外口径）；
# - maximal_all_mac：把空间/深度卷积也硬拉进映射（工程上界，不面向论文）。
MAPPING_SCENARIOS = [
    {
        "scenario": "linear_only",
        "label": "Traced Linear only",
        "groups": {
            "pointwise_projection",
            "qkv_projection",
            "attention_output_projection",
            "ffn_linear",
            "classifier_linear",
        },
        "evidence_tier": "local module-hook trace",
        "claim_status": "conservative trace-only baseline",
    },
    {
        "scenario": "plus_pointwise_attention",
        "label": "Reasonable-high: + 1x1 conv + attention matmuls",
        "groups": {
            "pointwise_projection",
            "qkv_projection",
            "attention_output_projection",
            "ffn_linear",
            "classifier_linear",
            "pointwise_conv",
            "attention_score",
            "attention_value",
        },
        "evidence_tier": "local module trace plus attention-shape-derived function-op estimate",
        "claim_status": "paper-facing reasonable-high boundary",
    },
    {
        "scenario": "maximal_all_mac",
        "label": "Maximal direct-streaming all-MAC mapping",
        "groups": {
            "pointwise_projection",
            "qkv_projection",
            "attention_output_projection",
            "ffn_linear",
            "classifier_linear",
            "pointwise_conv",
            "attention_score",
            "attention_value",
            "spatial_conv",
            "depthwise_conv",
        },
        "evidence_tier": "local shape/MAC trace with analytical direct sliding-window convolution lowering",
        "claim_status": "engineering-only analytical ceiling; not paper-facing",
    },
]


def operator_mapping_scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize increasingly broad mapping boundaries without promoting them to measurements."""
    # （英文原注释）汇总逐级放宽的映射边界，但绝不把它们包装成实测。
    # 通俗说：对每个模型、每个场景，统计"属于该场景算子组的 MAC 总数"
    # 占全体 MAC 的比例——比例越高说明能享受光加速的部分越大。
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    order = {"MobileViT-XXS": 0, "MobileViT-XS": 1, "MobileViT-S": 2}
    out: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items(), key=lambda item: order.get(item[0], 99)):
        total = sum(_f(row, "estimated_macs") for row in model_rows) or 1.0
        for scenario in MAPPING_SCENARIOS:
            mapped = sum(
                _f(row, "estimated_macs")
                for row in model_rows
                if row.get("op_group") in scenario["groups"]  # 只累计属于本场景算子组的 MAC
            )
            remainder = max(total - mapped, 0.0)  # 电子端剩余（防负数）
            out.append(
                {
                    "model": model,
                    "scenario": scenario["scenario"],
                    "scenario_label": scenario["label"],
                    "total_macs": f"{total:.0f}",
                    "mapped_macs": f"{mapped:.0f}",
                    "electronic_remainder_macs": f"{remainder:.0f}",
                    "mapped_mac_share_percent": f"{100.0 * mapped / total:.4f}",
                    "electronic_remainder_share_percent": f"{100.0 * remainder / total:.4f}",
                    "evidence_tier": scenario["evidence_tier"],
                    "claim_status": scenario["claim_status"],
                    "claim_boundary": (
                        "Mapping opportunity and Amdahl design bound only; not measured HPAT runtime, "
                        "energy, edge deployment, or fabricated-silicon evidence."
                    ),
                }
            )
    return out


def scenario_speedup_bound_rows(
    scenario_rows: list[dict[str, Any]], optical_speedups: list[float]
) -> list[dict[str, Any]]:
    """对每个映射场景，用阿姆达尔定律计算加速比上界。

    与 speedup_bound_rows 的区别：这里用的是"映射占比"（mapped_fraction）
    而非固定分区，可对比同一模型在不同映射假设下的上界差异。

    :param scenario_rows: operator_mapping_scenario_rows 的输出。
    :param optical_speedups: 要扫描的光加速比列表。
    :return: 每个 (模型, 场景, 光加速比) 一行的上界表。
    """
    out: list[dict[str, Any]] = []
    for row in scenario_rows:
        f_map = float(row["mapped_mac_share_percent"]) / 100.0  # 映射进光子端的 MAC 占比
        for speedup in optical_speedups:
            bound = 1.0 / ((1.0 - f_map) + f_map / speedup) if speedup > 0 else 1.0
            out.append(
                {
                    "model": row["model"],
                    "scenario": row["scenario"],
                    "scenario_label": row["scenario_label"],
                    "mapped_fraction": f"{f_map:.8f}",
                    "optical_speedup_assumption": f"{float(speedup):.6f}",
                    "e2e_speedup_upper_bound": f"{bound:.6f}",
                    "evidence_tier": row["evidence_tier"],
                    "claim_status": row["claim_status"],
                    "claim_boundary": row["claim_boundary"],
                }
            )
    return out


# 能耗分项归并成 5 个"族"，供不确定性扫描用（每族一个乘数）
COMPONENT_GROUPS = {
    "optical": {"Optical source and passive loss budget"},          # 光学族
    "converters": {"DAC and input modulation", "O/E readout"},      # 转换器族
    "memory_bus": {"SRAM/eDRAM local buffers", "Electrical bus/data movement"},  # 存储总线族
    "thermal_calibration": {"MRR programming/hold tuning", "Thermal tuning/tracking", "Amortized calibration"},  # 热/校准族
    "digital_remainder": {"Digital accumulation/control", "LayerNorm/softmax/nonlinear electronics"},  # 数字剩余族
}


def energy_uncertainty_rows(component_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """能耗不确定性网格：给 5 个能耗族各乘一个不确定系数，看总能耗波动。

    动机：单位成本本身有不确定性（工程估算），因此把每个族在某个范围内
    乘上不同系数，穷举组合得到一批"情景"，统计总能耗相对名义值（乘数全 1）
    的偏差百分比，用于给论文能耗结论画误差带。

    :param component_rows: 组件级能耗行（energy_model.component_energy_rows 的输出）。
    :param config: 实验配置（读 energy_uncertainty 段的系数网格）。
    :return: 每个情景一行的不确定性表。
    """
    grid = config.get("energy_uncertainty", {})
    optical = grid.get("optical", [0.75, 1.0, 1.5])          # 光学族系数
    converters = grid.get("converters", [0.75, 1.0, 2.0])    # 转换器族系数
    memory_bus = grid.get("memory_bus", [0.5, 1.0, 2.0])     # 存储总线族系数
    thermal_calibration = grid.get("thermal_calibration", [0.5, 1.0, 3.0])  # 热/校准族系数
    digital_remainder = grid.get("digital_remainder", [0.75, 1.0, 1.5])     # 数字剩余族系数
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in component_rows:
        by_model.setdefault(row["model_variant"], []).append(row)
    out: list[dict[str, Any]] = []
    order = {"MobileViT-XXS": 0, "MobileViT-XS": 1, "MobileViT-S": 2}
    for model, rows in sorted(by_model.items(), key=lambda item: order.get(item[0], 99)):
        nominal = sum(_f(row, "energy_mj") for row in rows) or 1.0  # 名义总能耗（全系数 1）
        scenario_id = 0
        # 五重循环穷举全部系数组合
        for m_opt in optical:
            for m_conv in converters:
                for m_mem in memory_bus:
                    for m_therm in thermal_calibration:
                        for m_dig in digital_remainder:
                            multipliers = {
                                "optical": float(m_opt),
                                "converters": float(m_conv),
                                "memory_bus": float(m_mem),
                                "thermal_calibration": float(m_therm),
                                "digital_remainder": float(m_dig),
                            }
                            total = 0.0
                            for row in rows:
                                component = row["component_group"]
                                multiplier = 1.0
                                # 找到该分项所属的族，取对应乘数
                                for family, names in COMPONENT_GROUPS.items():
                                    if component in names:
                                        multiplier = multipliers[family]
                                        break
                                total += _f(row, "energy_mj") * multiplier
                            out.append(
                                {
                                    "scenario_id": scenario_id,
                                    "model": model,
                                    "optical_multiplier": f"{multipliers['optical']:.4f}",
                                    "converter_multiplier": f"{multipliers['converters']:.4f}",
                                    "memory_bus_multiplier": f"{multipliers['memory_bus']:.4f}",
                                    "thermal_calibration_multiplier": f"{multipliers['thermal_calibration']:.4f}",
                                    "digital_remainder_multiplier": f"{multipliers['digital_remainder']:.4f}",
                                    "nominal_energy_mj": f"{nominal:.6f}",
                                    "total_energy_mj": f"{total:.6f}",
                                    "delta_vs_nominal_percent": f"{100.0 * (total - nominal) / nominal:.4f}",
                                    "evidence_label": "modelled component uncertainty sensitivity; not calibrated silicon energy",
                                }
                            )
                            scenario_id += 1
    return out


def energy_uncertainty_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把不确定性网格结果浓缩成每模型的 p05/名义/p50/p95 四档能耗。

    :param rows: energy_uncertainty_rows 的输出。
    :return: 每模型 4 行的汇总表（p05=悲观下限、p50=中位、p95=乐观上限）。
    """
    by_model: dict[str, list[float]] = {}
    nominal: dict[str, float] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(float(row["total_energy_mj"]))
        nominal[row["model"]] = float(row["nominal_energy_mj"])
    out: list[dict[str, Any]] = []
    order = {"MobileViT-XXS": 0, "MobileViT-XS": 1, "MobileViT-S": 2}
    for model, values in sorted(by_model.items(), key=lambda item: order.get(item[0], 99)):
        values = sorted(values)
        n = len(values)
        p05 = values[int(0.05 * (n - 1))]  # 5% 分位
        p50 = values[int(0.50 * (n - 1))]  # 50% 分位（中位数）
        p95 = values[int(0.95 * (n - 1))]  # 95% 分位
        for case, value in [("p05", p05), ("nominal", nominal[model]), ("p50", p50), ("p95", p95)]:
            out.append(
                {
                    "model": model,
                    "case": case,
                    "energy_mj": f"{value:.6f}",
                    "evidence_label": "modelled component uncertainty sensitivity summary",
                }
            )
    return out


# 算子域汇总表的列名清单
OPERATOR_DOMAIN_SUMMARY_FIELDS = [
    "model",
    "backend",
    "trace_source",
    "total_rows",
    "total_macs",
    "pdpu_candidate_macs",
    "electronic_remainder_macs",
    "pdpu_candidate_mac_share_percent",
    "electronic_remainder_mac_share_percent",
    "total_input_mib",
    "total_output_mib",
    "total_weight_mib",
    "nonlinear_ops",
    "evidence_label",
]


# 加速比上界表的列名清单
SPEEDUP_BOUND_FIELDS = [
    "model",
    "electronic_fraction",
    "pdpu_candidate_fraction",
    "optical_speedup_assumption",
    "e2e_speedup_upper_bound",
    "evidence_label",
]


# 映射场景汇总表的列名清单
OPERATOR_MAPPING_SCENARIO_FIELDS = [
    "model",
    "scenario",
    "scenario_label",
    "total_macs",
    "mapped_macs",
    "electronic_remainder_macs",
    "mapped_mac_share_percent",
    "electronic_remainder_share_percent",
    "evidence_tier",
    "claim_status",
    "claim_boundary",
]


# 场景级加速比上界表的列名清单
SCENARIO_SPEEDUP_BOUND_FIELDS = [
    "model",
    "scenario",
    "scenario_label",
    "mapped_fraction",
    "optical_speedup_assumption",
    "e2e_speedup_upper_bound",
    "evidence_tier",
    "claim_status",
    "claim_boundary",
]


# 能耗不确定性表的列名清单
ENERGY_UNCERTAINTY_FIELDS = [
    "scenario_id",
    "model",
    "optical_multiplier",
    "converter_multiplier",
    "memory_bus_multiplier",
    "thermal_calibration_multiplier",
    "digital_remainder_multiplier",
    "nominal_energy_mj",
    "total_energy_mj",
    "delta_vs_nominal_percent",
    "evidence_label",
]


# 能耗不确定性汇总表的列名清单
ENERGY_UNCERTAINTY_SUMMARY_FIELDS = ["model", "case", "energy_mj", "evidence_label"]
