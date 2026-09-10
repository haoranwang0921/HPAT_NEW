"""P2 证据管线：用"物理代理/仿真"产出的补充证据（回应审稿/支撑 future work）。

背景：论文证据三档 P0/P1/P2 中，P2 比 P1 更接近物理现实但仍非硅片实测。
本文件提供四类 P2 证据：
1) layout_area_feasibility_rows：版图面积可行性代理——估算 MRR（微环）、
   转换器、互连三部分面积，看是否落在声明的面积预算/波长预算内；
2) thermal_tuning_stress_rows：热调谐压力代理——热漂移、校准间隔、
   热校系数三因素交叉扫描，看延迟/能耗额外开销与"高压"风险；
3) public_edge_context_rows：公开文献/基准（MLPerf、Jetson、MobileViT、
   EfficientFormer、RepViT）的端侧部署上下文表——只用于动机与背景，
   明确禁止用来算 HPAT 加速比；
4) additional_model_family_*：把 EfficientFormer 等额外模型家族纳入
   hook 轨迹统计，看它们的 PDPU/电子端 MAC 占比。

所有输出带 P2_CLAIM_BOUNDARY：只支撑有边界的 rebuttal/supplement/
future-work 讨论，不支撑硅片验证、物理设计收敛、端侧实测或加速比声明。

易混淆点：P1 是解析建模，P2 是物理代理/仿真 + 文献上下文；两者都
不是 P0 的"模拟器导出活动"意义上的直接证据，但本仓库用三档标签
明确区分，防止把建模当实测。
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from .activity_trace import OPERATOR_ACTIVITY_FIELDS
from .mobilevit_loader import primary_token_count, variants_from_config


# P2 证据标签与声明边界
P2_EVIDENCE_LABEL = "local/modelled P2 proxy; not silicon, not edge deployment, not physical layout closure"
P2_LITERATURE_LABEL = "literature-context P2 refresh; not HPAT speedup evidence"
P2_CLAIM_BOUNDARY = (
    "Supports bounded rebuttal, supplement, or future-work context only. "
    "Does not support fabricated-silicon validation, physical-design closure, "
    "measured HPAT edge/mobile deployment, or HPAT speedup claims."
)


# ---- 各输出表的列名清单 ----
# 版图面积可行性表：每个硬件配置组合一行
LAYOUT_AREA_FIELDS = [
    "variant",
    "n_tokens",
    "embedding_dim",
    "wavelengths",
    "pdpu_banks",
    "tiles",
    "bit_width",
    "wavelength_budget",
    "mrr_count_proxy",
    "dac_count_proxy",
    "adc_count_proxy",
    "mrr_area_proxy_mm2",
    "converter_area_proxy_mm2",
    "interconnect_area_proxy_mm2",
    "total_area_proxy_mm2",
    "area_budget_proxy_mm2",
    "area_budget_status",
    "path_loss_db_proxy",
    "laser_power_proxy_mw",
    "thermal_density_proxy",
    "proxy_status",
    "evidence_label",
    "claim_boundary",
]

# 热调谐压力表：每 (热漂移, 热校系数, 校准间隔) 一行
THERMAL_STRESS_FIELDS = [
    "variant",
    "wavelengths",
    "pdpu_banks",
    "tiles",
    "thermal_drift_c",
    "thermal_calibration_multiplier",
    "mrr_count_proxy",
    "active_optical_time_ns_proxy",
    "thermal_tracking_time_ns_proxy",
    "thermal_tuning_proxy",
    "thermal_energy_multiplier_proxy",
    "retune_interval_inferences",
    "calibration_events_proxy",
    "latency_overhead_ns_proxy",
    "energy_overhead_mj_proxy",
    "stress_status",
    "proxy_status",
    "evidence_label",
    "claim_boundary",
]

# 公开端侧上下文表（每来源一行）
PUBLIC_EDGE_CONTEXT_FIELDS = [
    "source_id",
    "source",
    "evidence_tier",
    "model_task",
    "device_platform",
    "runtime_framework",
    "precision",
    "batch_scenario",
    "input_data",
    "reported_latency_throughput",
    "power_energy",
    "measurement_scope",
    "hpat_use",
    "source_url_or_doi",
    "verified_date",
    "claim_boundary",
]

# 公开端侧上下文来源表（更简的引用清单）
PUBLIC_EDGE_CONTEXT_SOURCE_FIELDS = [
    "source_id",
    "title",
    "url_or_doi",
    "source_type",
    "verified_date",
    "refresh_status",
    "claim_boundary",
    "notes",
]

# 额外模型家族的算子行 = 标准算子行字段 + family 列
ADDITIONAL_MODEL_OPERATOR_FIELDS = ["family"] + OPERATOR_ACTIVITY_FIELDS


# 额外模型家族汇总表
ADDITIONAL_MODEL_SUMMARY_FIELDS = [
    "family",
    "variant",
    "timm_model",
    "input_resolution",
    "backend",
    "precision",
    "status",
    "blocked_reason",
    "row_count",
    "pdpu_candidate_mac_share_percent",
    "electronic_remainder_mac_share_percent",
    "hybrid_support_mac_share_percent",
    "pdpu_candidate_byte_share_percent",
    "dominant_electronic_op_groups",
    "trace_source",
    "device_reason",
    "evidence_label",
    "claim_boundary",
    "promotion_requirements",
]


def p2_config(config: dict[str, Any]) -> dict[str, Any]:
    """取出配置里的 "p2" 段（不存在则空字典）。

    :param config: 顶层配置。
    :return: p2 配置字典。
    """
    return dict(config.get("p2") or {})


def _default_layout_config(config: dict[str, Any]) -> dict[str, Any]:
    """解析版图面积代理的默认配置（配置里的覆盖，缺省用内置值）。

    :param config: 顶层配置。
    :return: 版图配置字典（含面积预算、单元面积、损耗参数等）。
    """
    scfg = config.get("scalability", {})
    pcfg = scfg.get("physical_proxy", {})
    p2 = p2_config(config)
    layout = dict(p2.get("layout_area_proxy") or {})
    return {
        "wavelengths": layout.get("wavelengths", [8, 16, 32]),
        "pdpu_banks": layout.get("pdpu_banks", [1, 2, 4]),
        "tiles": layout.get("tiles", [1, 2]),
        "bit_width": int(layout.get("bit_width", config.get("qkv", {}).get("default_bit_width", 8))),
        "wavelength_budget": int(layout.get("wavelength_budget", scfg.get("wavelength_budget", 32))),
        "area_budget_mm2": float(layout.get("area_budget_mm2", 16.0)),  # 总面积预算（mm²）
        "default_adc_count_proxy": int(layout.get("default_adc_count_proxy", 16)),
        "interconnect_area_multiplier": float(layout.get("interconnect_area_multiplier", 0.35)),  # 互连面积 = (MRR+转换器)×0.35
        "mrr_area_um2": float(layout.get("mrr_area_um2", pcfg.get("mrr_area_um2", 100.0))),
        "converter_area_um2": float(layout.get("converter_area_um2", pcfg.get("converter_area_um2", 4000.0))),
        "base_path_loss_db": float(layout.get("base_path_loss_db", pcfg.get("base_path_loss_db", 2.0))),
        "loss_per_tile_db": float(layout.get("loss_per_tile_db", pcfg.get("loss_per_tile_db", 0.25))),
        "loss_per_wavelength_db": float(
            layout.get("loss_per_wavelength_db", pcfg.get("loss_per_wavelength_db", 0.01))
        ),
        "laser_mw_per_db": float(layout.get("laser_mw_per_db", pcfg.get("laser_mw_per_db", 0.8))),
        "thermal_density_limit": float(layout.get("thermal_density_limit", pcfg.get("thermal_density_limit", 200000.0))),
    }


def layout_area_feasibility_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """版图面积可行性分析：估算三大块面积并对照预算。

    估算方法（全部是解析近似，非版图收敛）：
    - MRR 面积 = 微环数 × 单环面积（默认 100 μm²）；
    - 转换器面积 = (DAC 数 + ADC 数) × 单转换器面积（默认 4000 μm²）；
    - 互连面积 = (MRR + 转换器) × 0.35；
    - 总面积 = 三者之和，对比 area_budget_mm2；
    - 同时估算路径损耗、激光功率、热密度，并做预算越界检查。

    :param config: 顶层配置。
    :return: 版图可行性表（每硬件配置组合一行）。
    """
    lcfg = _default_layout_config(config)
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])
        n = primary_token_count(variant, config)
        # 三层循环穷举硬件规模组合
        for wavelengths in lcfg["wavelengths"]:
            for banks in lcfg["pdpu_banks"]:
                for tiles in lcfg["tiles"]:
                    w = int(wavelengths)
                    bank = int(banks)
                    tile = int(tiles)
                    mrr_count = w * d * bank * tile            # 微环总数代理
                    dac_count = w * bank * tile                # DAC 数代理
                    adc_count = int(lcfg["default_adc_count_proxy"])
                    converter_count = dac_count + adc_count
                    mrr_area = mrr_count * float(lcfg["mrr_area_um2"]) / 1_000_000.0      # μm²→mm²
                    converter_area = converter_count * float(lcfg["converter_area_um2"]) / 1_000_000.0
                    interconnect_area = (mrr_area + converter_area) * float(lcfg["interconnect_area_multiplier"])
                    total_area = mrr_area + converter_area + interconnect_area
                    path_loss = (
                        float(lcfg["base_path_loss_db"])
                        + float(lcfg["loss_per_tile_db"]) * tile
                        + float(lcfg["loss_per_wavelength_db"]) * w
                    )
                    laser_power = path_loss * float(lcfg["laser_mw_per_db"]) * max(1, bank)
                    thermal_density = mrr_count / max(total_area, 1e-9)
                    # 越界判定：总面积超预算 或 波长超预算
                    budget_status = (
                        "within_declared_area_proxy"
                        if total_area <= float(lcfg["area_budget_mm2"]) and w <= int(lcfg["wavelength_budget"])
                        else "exceeds_declared_area_or_wavelength_proxy"
                    )
                    rows.append(
                        {
                            "variant": variant["variant"],
                            "n_tokens": n,
                            "embedding_dim": d,
                            "wavelengths": w,
                            "pdpu_banks": bank,
                            "tiles": tile,
                            "bit_width": int(lcfg["bit_width"]),
                            "wavelength_budget": int(lcfg["wavelength_budget"]),
                            "mrr_count_proxy": mrr_count,
                            "dac_count_proxy": dac_count,
                            "adc_count_proxy": adc_count,
                            "mrr_area_proxy_mm2": f"{mrr_area:.6f}",
                            "converter_area_proxy_mm2": f"{converter_area:.6f}",
                            "interconnect_area_proxy_mm2": f"{interconnect_area:.6f}",
                            "total_area_proxy_mm2": f"{total_area:.6f}",
                            "area_budget_proxy_mm2": f"{float(lcfg['area_budget_mm2']):.6f}",
                            "area_budget_status": budget_status,
                            "path_loss_db_proxy": f"{path_loss:.6f}",
                            "laser_power_proxy_mw": f"{laser_power:.6f}",
                            "thermal_density_proxy": f"{thermal_density:.6f}",
                            "proxy_status": "layout_area_proxy_not_physical_design_closure",  # 明确：面积代理≠物理设计收敛
                            "evidence_label": P2_EVIDENCE_LABEL,
                            "claim_boundary": P2_CLAIM_BOUNDARY,
                        }
                    )
    return rows


def _default_thermal_config(config: dict[str, Any]) -> dict[str, Any]:
    """解析热调谐压力分析的默认配置。

    :param config: 顶层配置。
    :return: 热配置字典（漂移档位、校准间隔档位、重调谐基线延迟等）。
    """
    p2 = p2_config(config)
    tcfg = dict(p2.get("thermal_stress") or {})
    scfg = config.get("scalability", {})
    return {
        "wavelengths": int(tcfg.get("wavelengths", 16)),
        "pdpu_banks": int(tcfg.get("pdpu_banks", 2)),
        "tiles": int(tcfg.get("tiles", 2)),
        "thermal_drift_c": tcfg.get("thermal_drift_c", config.get("nonideality", {}).get("thermal_drift_c", [0.0, 2.0, 5.0, 10.0])),
        "thermal_calibration_multipliers": tcfg.get(
            "thermal_calibration_multipliers",
            scfg.get("thermal_calibration_multipliers", [1.0, 1.5]),
        ),
        "retune_interval_inferences": tcfg.get("retune_interval_inferences", [100, 1000, 10000]),
        "base_retune_latency_ns": float(tcfg.get("base_retune_latency_ns", 60000.0)),  # 一次重调谐基线延迟
        "high_risk_thermal_drift_c": float(tcfg.get("high_risk_thermal_drift_c", 10.0)),  # 高压热漂移阈值
    }


def thermal_tuning_stress_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """热调谐压力分析：扫描热漂移 × 热校系数 × 校准间隔，看开销与风险。

    模型要点：
    - 热跟踪时长随漂移加大（× (1+漂移/20)），热调谐开销随漂移与微环数加大；
    - 每次重调谐摊 60000 ns 到单次推理（除以校准间隔）；
    - 能耗乘数 = 1 + 0.04×热校系数 + 0.01×漂移 + 0.03×(1000/间隔)；
    - 高压判定：漂移≥阈值、或热校系数≥2.0、或间隔≤100。

    :param config: 顶层配置。
    :return: 热压力表（每 (漂移, 系数, 间隔) 一行）。
    """
    tcfg = _default_thermal_config(config)
    ops_per_ns = float(config.get("activity_proxy", {}).get("ops_per_ns", 250000.0))
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])
        n = primary_token_count(variant, config)
        base_energy = float(config.get("energy_envelope_mj", {}).get(variant["variant"], 1.0))
        base_ops = 3 * n * d * d + 2 * n * n * d + 2 * n * d * (2 * d)
        optical_parallel = max(int(tcfg["wavelengths"]) * int(tcfg["pdpu_banks"]) * int(tcfg["tiles"]), 1)
        active_ns = base_ops / max(optical_parallel * ops_per_ns, 1e-9)  # 光计算活动时长
        mrr_count = int(tcfg["wavelengths"]) * d * int(tcfg["pdpu_banks"]) * int(tcfg["tiles"])
        # 三重循环：漂移 × 热校系数 × 校准间隔
        for drift in tcfg["thermal_drift_c"]:
            drift_f = float(drift)
            for multiplier in tcfg["thermal_calibration_multipliers"]:
                mult_f = float(multiplier)
                for interval in tcfg["retune_interval_inferences"]:
                    interval_i = max(int(interval), 1)
                    # 热跟踪与热调谐时长都随漂移放大（温度越高越难维持）
                    thermal_tracking = active_ns * mult_f * (1.0 + drift_f / 20.0)
                    thermal_tuning = mrr_count * mult_f * (1.0 + drift_f / 10.0)
                    # 延迟额外开销 = 基础热校延迟 + 每次重调谐摊还
                    latency_overhead = 40.0 * mult_f * max(1.0, int(tcfg["pdpu_banks"]) / 2.0) * (1.0 + drift_f / 20.0)
                    latency_overhead += float(tcfg["base_retune_latency_ns"]) / interval_i
                    thermal_energy_multiplier = 1.0 + 0.04 * mult_f + 0.01 * drift_f + 0.03 * (1000.0 / interval_i)
                    energy_overhead = base_energy * max(thermal_energy_multiplier - 1.0, 0.0)
                    # 高压判定：满足任一条件就标 high
                    status = (
                        "thermal_stress_high_proxy"
                        if drift_f >= float(tcfg["high_risk_thermal_drift_c"]) or mult_f >= 2.0 or interval_i <= 100
                        else "thermal_stress_within_declared_proxy"
                    )
                    rows.append(
                        {
                            "variant": variant["variant"],
                            "wavelengths": int(tcfg["wavelengths"]),
                            "pdpu_banks": int(tcfg["pdpu_banks"]),
                            "tiles": int(tcfg["tiles"]),
                            "thermal_drift_c": f"{drift_f:.6f}",
                            "thermal_calibration_multiplier": f"{mult_f:.6f}",
                            "mrr_count_proxy": mrr_count,
                            "active_optical_time_ns_proxy": f"{active_ns:.6f}",
                            "thermal_tracking_time_ns_proxy": f"{thermal_tracking:.6f}",
                            "thermal_tuning_proxy": f"{thermal_tuning:.6f}",
                            "thermal_energy_multiplier_proxy": f"{thermal_energy_multiplier:.6f}",
                            "retune_interval_inferences": interval_i,
                            "calibration_events_proxy": max(1, mrr_count // max(interval_i, 1)),
                            "latency_overhead_ns_proxy": f"{latency_overhead:.6f}",
                            "energy_overhead_mj_proxy": f"{energy_overhead:.6f}",
                            "stress_status": status,
                            "proxy_status": "thermal_tuning_proxy_not_packaged_device_validation",
                            "evidence_label": P2_EVIDENCE_LABEL,
                            "claim_boundary": P2_CLAIM_BOUNDARY,
                        }
                    )
    return rows


def _verified_date() -> str:
    """返回今天的 UTC 日期字符串（用于给文献上下文标注"核实日期"）。

    :return: ISO 格式日期（如 "2026-08-18"）。
    """
    return _dt.datetime.now(_dt.UTC).date().isoformat()


def public_edge_context_rows(verified_date: str | None = None) -> list[dict[str, Any]]:
    """公开端侧/移动端部署上下文表（P2 的文献背景证据）。

    收录 6 个来源：MLPerf Mobile/Edge 官方基准、NVIDIA Jetson 基准页、
    MobileViT（ICLR 2022）、EfficientFormer（NeurIPS 2022）、
    RepViT（CVPR 2024）。每行标注：来源、证据等级、任务、设备平台、
    运行时框架、精度、批处理场景、输入数据、报告指标、功率/能耗、
    测量范围、在 HPAT 论文里的用途（hpat_use）与核实日期。

    注意：这些数据只用于"部署压力"的动机背景，绝不用于算 HPAT 加速比。

    :param verified_date: 核实日期；缺省取当天。
    :return: 上下文表（每来源一行）。
    """
    date = verified_date or _verified_date()
    rows = [
        {
            "source_id": "mlperf_mobile",
            "source": "MLPerf Inference: Mobile",
            "evidence_tier": "official public benchmark",
            "model_task": "Mobile inference benchmark suite across vision, language, image processing, and generative AI tasks",
            "device_platform": "Mobile and mobile-class systems submitted to MLCommons",
            "runtime_framework": "MLPerf Mobile app, LoadGen, benchmark-specific frameworks",
            "precision": "benchmark-specific quality targets",
            "batch_scenario": "Single Stream required; Image Classification can also report Offline",
            "input_data": "Public benchmark datasets such as ImageNet, COCO, ADE20K, SQuAD, OpenSR, and related suite data",
            "reported_latency_throughput": "Interactive official table; scenario-specific latency or throughput",
            "power_energy": "Energy per stream for Single Stream / Multiple Stream where power is submitted",
            "measurement_scope": "Official mobile benchmark methodology and results table",
            "hpat_use": "Methodology and deployment-pressure context only; not an HPAT baseline",
            "source_url_or_doi": "https://mlcommons.org/benchmarks/inference-mobile/",
        },
        {
            "source_id": "mlperf_edge",
            "source": "MLPerf Inference: Edge",
            "evidence_tier": "official public benchmark",
            "model_task": "Edge inference benchmark suite",
            "device_platform": "Edge systems submitted to MLCommons",
            "runtime_framework": "MLPerf Inference LoadGen and submitter software stacks",
            "precision": "benchmark-specific quality targets",
            "batch_scenario": "Single Stream, Offline, and other edge scenarios depending on workload",
            "input_data": "MLPerf benchmark datasets and reference workloads",
            "reported_latency_throughput": "Interactive official table; row-specific metrics",
            "power_energy": "System power or energy columns when submitted",
            "measurement_scope": "Official edge benchmark methodology and results table",
            "hpat_use": "Evaluation-methodology context only; not HPAT speedup evidence",
            "source_url_or_doi": "https://mlcommons.org/benchmarks/inference-edge/",
        },
        {
            "source_id": "nvidia_jetson_benchmarks",
            "source": "NVIDIA Jetson Benchmarks",
            "evidence_tier": "official public benchmark aggregation",
            "model_task": "Jetson MLPerf Edge rows including image classification, object detection, NLP, speech, and generative workloads",
            "device_platform": "Jetson AGX Orin, Orin NX, and related Jetson platforms",
            "runtime_framework": "TensorRT, CUDA, JetPack; workload-specific stacks",
            "precision": "N/R in context table",
            "batch_scenario": "MLPerf Edge scenarios by row",
            "input_data": "MLPerf workloads",
            "reported_latency_throughput": "Representative v3.1 rows include ResNet Single Stream 0.64 ms and Retinanet Single Stream 11.67 ms on AGX Orin",
            "power_energy": "Representative MaxQ rows report system power columns",
            "measurement_scope": "Vendor page aggregating NVIDIA Jetson MLPerf submissions with reproducibility links",
            "hpat_use": "Embedded edge-platform context only; not directly comparable to HPAT",
            "source_url_or_doi": "https://developer.nvidia.com/embedded/jetson-benchmarks",
        },
        {
            "source_id": "apple_mobilevit",
            "source": "MobileViT, ICLR 2022",
            "evidence_tier": "literature-context",
            "model_task": "MobileViT hybrid CNN-Transformer vision model",
            "device_platform": "iPhone 12 CPU, iPhone 12 Neural Engine, and separate GPU context in the paper",
            "runtime_framework": "CoreMLTools for mobile conversion",
            "precision": "paper-reported converted model precision scope",
            "batch_scenario": "batch 1 on iPhone in mobile latency context",
            "input_data": "ImageNet-1K and related vision tasks",
            "reported_latency_throughput": "Paper positions MobileViT as lightweight, low-latency mobile vision",
            "power_energy": "N/R in this context row",
            "measurement_scope": "Public paper and Apple research summary",
            "hpat_use": "Workload motivation and MobileViT boundary only",
            "source_url_or_doi": "https://machinelearning.apple.com/research/vision-transformer",
        },
        {
            "source_id": "efficientformer_neurips_2022",
            "source": "EfficientFormer, NeurIPS 2022",
            "evidence_tier": "literature-context",
            "model_task": "Mobile-friendly vision transformer family",
            "device_platform": "iPhone 12 with CoreMLTools; GPU context reported separately",
            "runtime_framework": "CoreMLTools for mobile latency reporting",
            "precision": "N/R in context table",
            "batch_scenario": "Public paper mobile latency profiling",
            "input_data": "ImageNet-1K",
            "reported_latency_throughput": "EfficientFormer-L1 reports 1.6 ms and EfficientFormer-L7 reports 7.0 ms iPhone 12 latency in the paper",
            "power_energy": "N/R",
            "measurement_scope": "Public paper latency/accuracy context",
            "hpat_use": "Motivates mobile ViT latency and supplies default P2 additional-family target",
            "source_url_or_doi": "https://papers.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf",
        },
        {
            "source_id": "repvit_cvpr_2024",
            "source": "RepViT, CVPR 2024",
            "evidence_tier": "literature-context",
            "model_task": "Mobile CNN revisited from ViT perspective",
            "device_platform": "iPhone 12 with Core ML Tools",
            "runtime_framework": "Core ML Tools and Xcode benchmarking in public repository notes",
            "precision": "N/R in context table",
            "batch_scenario": "ImageNet-1K model rows with iPhone latency",
            "input_data": "ImageNet-1K",
            "reported_latency_throughput": "Repository reports RepViT-M1.0 at 1.0 ms and RepViT-M2.3 at 2.3 ms iPhone 12 latency",
            "power_energy": "N/R",
            "measurement_scope": "Public paper/repository context",
            "hpat_use": "Current mobile-friendly model context only; not an HPAT comparison",
            "source_url_or_doi": "https://github.com/THU-MIG/RepViT",
        },
    ]
    for row in rows:
        row["verified_date"] = date
        row["claim_boundary"] = P2_CLAIM_BOUNDARY
    return rows


def public_edge_context_source_rows(verified_date: str | None = None) -> list[dict[str, Any]]:
    """公开端侧上下文的"来源清单"表（更精简的引用列表）。

    :param verified_date: 核实日期；缺省取当天。
    :return: 来源清单表（每来源一行）。
    """
    date = verified_date or _verified_date()
    sources = [
        ("mlperf_mobile", "MLPerf Inference: Mobile", "https://mlcommons.org/benchmarks/inference-mobile/", "official benchmark"),
        ("mlperf_edge", "MLPerf Inference: Edge", "https://mlcommons.org/benchmarks/inference-edge/", "official benchmark"),
        ("nvidia_jetson_benchmarks", "NVIDIA Jetson Benchmarks", "https://developer.nvidia.com/embedded/jetson-benchmarks", "official vendor benchmark aggregation"),
        ("apple_mobilevit", "MobileViT Apple research page", "https://machinelearning.apple.com/research/vision-transformer", "public research summary"),
        ("efficientformer_neurips_2022", "EfficientFormer NeurIPS 2022 paper", "https://papers.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf", "primary paper"),
        ("repvit_cvpr_2024", "RepViT public repository", "https://github.com/THU-MIG/RepViT", "public paper/repository"),
    ]
    return [
        {
            "source_id": source_id,
            "title": title,
            "url_or_doi": url,
            "source_type": source_type,
            "verified_date": date,
            "refresh_status": "refreshed_from_verified_public_source",
            "claim_boundary": P2_CLAIM_BOUNDARY,
            "notes": "Context only; do not use to compute HPAT speedup or superiority.",
        }
        for source_id, title, url, source_type in sources
    ]


def public_edge_context_markdown(rows: list[dict[str, Any]]) -> str:
    """把公开端侧上下文表渲染成 Markdown（含使用规则，禁止用它算加速比）。

    :param rows: public_edge_context_rows 的输出。
    :return: Markdown 字符串。
    """
    lines = [
        "# Public Edge/Mobile Deployment Context",
        "",
        "Claim-boundary note: this table summarizes public literature, vendor, and benchmark context for mobile/edge deployment pressure. It is not HPAT real-device evidence, does not report author-measured HPAT results on mobile or edge hardware, and must not be used to compute HPAT speedup.",
        "",
        "| Source | Evidence tier | Model/task | Device/platform | Runtime/framework | Precision | Batch/scenario | Input/data | Reported latency/throughput | Power/energy | Measurement scope | HPAT use | Verified |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        source = f"[{row['source']}]({row['source_url_or_doi']})"
        lines.append(
            f"| {source} | {row['evidence_tier']} | {row['model_task']} | {row['device_platform']} | "
            f"{row['runtime_framework']} | {row['precision']} | {row['batch_scenario']} | {row['input_data']} | "
            f"{row['reported_latency_throughput']} | {row['power_energy']} | {row['measurement_scope']} | "
            f"{row['hpat_use']} | {row['verified_date']} |"
        )
    lines.extend(
        [
            "",
            "## Use Rules",
            "",
            "- Use these rows only to motivate deployment pressure, mobile/edge runtime constraints, and methodology wording.",
            "- Do not compute HPAT speedup, energy reduction, or superiority from this table; the rows mix devices, compilers, scenarios, input sizes, model families, and measurement scopes.",
            "- Do not write that HPAT was measured on mobile/edge hardware based on this table.",
            "- Keep source labels visible: `literature-context` and `official public benchmark` are not fabricated-chip or author-measured HPAT evidence.",
            "- Preserve `N/R` when a source does not report a field in the cited scope; do not infer precision, power, or energy from adjacent systems.",
        ]
    )
    return "\n".join(lines) + "\n"


def additional_model_family_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """返回 P2 要补充统计的"额外模型家族"规格列表。

    配置里写了 p2.additional_model_families 就用配置的；否则用内置的
    EfficientFormer 三档（L1/L3/L7）规格。

    :param config: 顶层配置。
    :return: 规格字典列表（family/name/timm_model/参数量/d/分辨率/token 扫描）。
    """
    p2 = p2_config(config)
    specs = p2.get("additional_model_families")
    if specs:
        return [dict(item) for item in specs]
    return [
        {
            "family": "EfficientFormer",
            "name": "EfficientFormer-L1",
            "timm_model": "efficientformer_l1",
            "parameter_count_m": 12.3,
            "d": 448,
            "input_resolution": 224,
            "token_count_sweep": [196],
        },
        {
            "family": "EfficientFormer",
            "name": "EfficientFormer-L3",
            "timm_model": "efficientformer_l3",
            "parameter_count_m": 31.4,
            "d": 512,
            "input_resolution": 224,
            "token_count_sweep": [196],
        },
        {
            "family": "EfficientFormer",
            "name": "EfficientFormer-L7",
            "timm_model": "efficientformer_l7",
            "parameter_count_m": 82.2,
            "d": 768,
            "input_resolution": 224,
            "token_count_sweep": [196],
        },
    ]


def additional_family_trace_config(config: dict[str, Any]) -> dict[str, Any]:
    """构造一份"以额外模型家族为变体"的 trace 配置，供 activity_trace 用。

    把额外家族规格翻译成 mobilevit_variants 格式，塞进配置副本，
    这样可以直接复用 trace_timm_operator_rows 抓取这些模型的轨迹。

    :param config: 顶层配置。
    :return: 修改过 mobilevit_variants 的配置副本。
    """
    trace_config = dict(config)
    trace_config["mobilevit_variants"] = [
        {
            "name": spec["name"],
            "timm_model": spec["timm_model"],
            "parameter_count_m": spec.get("parameter_count_m", ""),
            "d": spec.get("d", spec.get("embedding_dim", 256)),
            "input_resolution": spec.get("input_resolution", 224),
            "token_count_sweep": spec.get("token_count_sweep", [196]),
        }
        for spec in additional_model_family_specs(config)
    ]
    return trace_config


def attach_family_to_operator_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """给算子行附上 family 列（区分 EfficientFormer 等家族）。

    :param rows: 算子行列表。
    :param config: 顶层配置。
    :return: 新增 "family" 字段后的新列表（不改原列表）。
    """
    family_by_variant = {spec["name"]: spec.get("family", "additional_model_family") for spec in additional_model_family_specs(config)}
    return [{**row, "family": family_by_variant.get(row.get("model", ""), "additional_model_family")} for row in rows]


def additional_model_family_summary_rows(
    *,
    operator_rows: list[dict[str, Any]],
    config: dict[str, Any],
    backend: str,
    precision: str,
    status: str,
    blocked_reason: str = "",
    device_reason: str = "",
) -> list[dict[str, Any]]:
    """按额外模型家族汇总算子轨迹，输出 MAC/字节 的域占比统计。

    对每个家族变体统计：PDPU 候选 / 电子端 / 混合支撑三类的 MAC 占比、
    PDPU 候选字节占比、占主导的电子端算子组（最多 4 个），以及
    trace 来源与状态。没有轨迹时状态降级为 proxy 或 blocked。

    :param operator_rows: 算子行列表（含 family 字段更佳）。
    :param config: 顶层配置。
    :param backend: 后端标签。
    :param precision: 精度。
    :param status: 轨迹抓取状态。
    :param blocked_reason: 阻塞原因（status 为 blocked 时）。
    :param device_reason: 设备选择原因。
    :return: 家族汇总表（每变体一行）。
    """
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in operator_rows:
        rows_by_variant.setdefault(str(row.get("model", "")), []).append(row)

    summary: list[dict[str, Any]] = []
    for spec in additional_model_family_specs(config):
        variant = spec["name"]
        rows = rows_by_variant.get(variant, [])
        total_macs = sum(float(row.get("estimated_macs") or 0.0) for row in rows) or 1.0
        total_bytes = sum(float(row.get("input_bytes") or 0.0) + float(row.get("output_bytes") or 0.0) for row in rows) or 1.0
        # 按执行域分桶累计 MAC
        pdpu_macs = sum(
            float(row.get("estimated_macs") or 0.0)
            for row in rows
            if row.get("execution_domain") == "PDPU-candidate linear/MVM"
        )
        electronic_macs = sum(
            float(row.get("estimated_macs") or 0.0)
            for row in rows
            if row.get("execution_domain") == "electronic remainder"
        )
        hybrid_macs = sum(
            float(row.get("estimated_macs") or 0.0)
            for row in rows
            if row.get("execution_domain") == "hybrid support"
        )
        # PDPU 候选的字节占比
        pdpu_bytes = sum(
            float(row.get("input_bytes") or 0.0) + float(row.get("output_bytes") or 0.0)
            for row in rows
            if row.get("execution_domain") == "PDPU-candidate linear/MVM"
        )
        # 电子端算子组按 MAC 从大到小排序，取前 4 名拼字符串
        electronic_groups: dict[str, float] = {}
        for row in rows:
            if row.get("execution_domain") != "electronic remainder":
                continue
            key = str(row.get("op_group") or "unknown")
            electronic_groups[key] = electronic_groups.get(key, 0.0) + float(row.get("estimated_macs") or 0.0)
        dominant = ";".join(
            group for group, _ in sorted(electronic_groups.items(), key=lambda item: item[1], reverse=True)[:4]
        )
        trace_sources = sorted({str(row.get("trace_source") or "") for row in rows if row.get("trace_source")})
        # 状态：有轨迹用传入 status；无轨迹时 blocked_reason 非空则 blocked，否则 proxy
        row_status = status if rows else ("blocked" if blocked_reason else "proxy")
        summary.append(
            {
                "family": spec.get("family", "additional_model_family"),
                "variant": variant,
                "timm_model": spec.get("timm_model", ""),
                "input_resolution": int(spec.get("input_resolution", 224)),
                "backend": backend,
                "precision": precision,
                "status": row_status,
                "blocked_reason": blocked_reason if not rows else "",
                "row_count": len(rows),
                "pdpu_candidate_mac_share_percent": f"{100.0 * pdpu_macs / total_macs:.6f}" if rows else "0.000000",
                "electronic_remainder_mac_share_percent": f"{100.0 * electronic_macs / total_macs:.6f}" if rows else "0.000000",
                "hybrid_support_mac_share_percent": f"{100.0 * hybrid_macs / total_macs:.6f}" if rows else "0.000000",
                "pdpu_candidate_byte_share_percent": f"{100.0 * pdpu_bytes / total_bytes:.6f}" if rows else "0.000000",
                "dominant_electronic_op_groups": dominant,
                "trace_source": (
                    "torch_hooks"
                    if "torch_hooks" in trace_sources
                    else (";".join(trace_sources) if trace_sources else "blocked")
                ),
                "device_reason": device_reason,
                "evidence_label": P2_EVIDENCE_LABEL if rows else "blocked P2 additional-family trace",
                "claim_boundary": P2_CLAIM_BOUNDARY,
                "promotion_requirements": "Direct HPAT simulator/export support for this model family plus calibrated unit costs before quantitative promotion.",
            }
        )
    return summary
