"""论文实验图渲染脚本：把实验产出的 CSV/JSON 数据渲染成论文插图。

本文件属于 HPAT（光子张量处理器，见论文《HPAT：光子张量处理器》，ASP-DAC 2027 投稿）
可复现实验库。它负责把实验流水线生成的各类结果表（CSV/JSON）渲染成论文插图
（PNG），并为每张图生成 sidecar（配套元数据 JSON，记录数据来源与证据说明）。

数据从哪来：output_dir/tables/ 下的各张 CSV（如 fig6_evidence_repaired.csv、
hpat_energy_by_component.csv 等），由上游实验脚本产出。
输出到哪：output_dir/figures/ 下的 PNG 图 + 每张图的数据副本 CSV，
以及 output_dir/render_experiment_figures_manifest.json 渲染清单。
怎么运行：python render_experiment_figures.py --output-dir <实验输出目录>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.manifests import figure_sidecar
from hpat_eval.plotting import grouped_metric_panels, horizontal_grouped_bar, line_panel, stacked_bar


def _exists(path: pathlib.Path) -> bool:
    """判断文件是否存在且非空（文件存在但大小为零视为无效）。"""
    return path.exists() and path.stat().st_size > 0


def _first_existing(paths: list[pathlib.Path]) -> pathlib.Path | None:
    """按优先级返回第一个存在且非空的文件；全部无效则返回 None。

    用途：同一份实验数据可能因版本不同有新旧两种文件名
    （例如新格式 qkv_traffic_calibrated.csv 与旧格式 qkv_traffic_sensitivity.csv），
    这里依次查找，找到哪个就用哪个。
    """
    for path in paths:
        if _exists(path):
            return path
    return None


def _existing(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """过滤出列表中所有存在且非空的文件路径。

    用途：一张图可能同时依赖多个辅助文件（如就绪检查 JSON），
    这里把"真实存在"的文件挑出来一并记入 sidecar 的数据来源。
    """
    return [path for path in paths if _exists(path)]


def _copy_rows(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> pathlib.Path:
    """把行数据按列名写成 CSV，作为该插图的"数据来源副本"。

    用途：渲染 PNG 的同时把喂给绘图函数的行数据原样另存一份 CSV，
    方便审稿人/读者核对图中数字，返回该 CSV 路径。
    """
    write_csv(path, rows, fieldnames)
    return path


def _run_record(path: pathlib.Path, output_dir: pathlib.Path) -> str:
    """把绝对路径转成相对 output_dir 的正斜杠相对路径（记入 manifest 用）。

    用途：渲染清单里只记录相对路径而非绝对路径，
    这样整个实验输出目录可以被整体移动/打包而不失效。
    """
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def _energy_rows(tables_dir: pathlib.Path) -> tuple[list[dict[str, Any]], pathlib.Path | None]:
    """读取能耗分解数据，兼容新、旧两种 CSV 格式。

    优先读 hpat_energy_by_component.csv（新格式，已是标准行结构）；
    没有时退回旧的 energy_breakdown.csv（旧格式，一列一个模型），
    并把旧格式展开成 model_variant / component_group / energy_mj 的标准行。
    返回 (行数据, 源文件路径)；都找不到时返回 ([], None)。
    """
    component_path = tables_dir / "hpat_energy_by_component.csv"
    if _exists(component_path):
        rows = read_csv(component_path)
        return rows, component_path
    legacy = tables_dir / "energy_breakdown.csv"
    if not _exists(legacy):
        return [], None
    rows = []
    for row in read_csv(legacy):
        # 旧格式每个模型占一列（如 xxs_mj），这里逐列展开成独立数据行
        for model, key in [
            ("MobileViT-XXS", "xxs_mj"),
            ("MobileViT-XS", "xs_mj"),
            ("MobileViT-S", "s_mj"),
        ]:
            rows.append(
                {
                    "model_variant": model,
                    "component_group": row["component_group"],
                    "energy_mj": row[key],
                    "evidence_label": row.get("claim_status", "modelled allocation"),
                }
            )
    return rows, legacy


def _operator_domain_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """按 (模型, 执行域) 汇总估算的 MAC 数（乘累加运算量）。

    用途：为 "fig_operator_domain_breakdown" 图准备数据——同一模型、
    同一执行域（如 PDPU 候选的线性/MVM 运算 vs 电子剩余）的多行
    估算 MAC 累加成一行，用于堆叠柱状图。
    """
    buckets: dict[tuple[str, str], float] = {}
    for row in read_csv(path):
        key = (row["model"], row["execution_domain"])
        buckets[key] = buckets.get(key, 0.0) + float(row.get("estimated_macs") or 0.0)  # 按组累加估算 MAC
    return [
        {"model_variant": model, "execution_domain": domain, "estimated_macs": f"{value:.0f}"}
        for (model, domain), value in sorted(buckets.items())
    ]


def _qkv_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为 Q/K/V 流量敏感性图（fig_qkv_traffic_sensitivity）准备堆叠数据。

    只保留 8-bit 量化、196 个 token 这一组固定配置的行；再把每种流量组件
    （激活输入、Q/K/V 投影输出、注意力分数、值乘积、电子剩余、权重重流、
    编程、总线）的字节数统一换算成 KiB（1 KiB = 1024 字节），并兼容
    旧版本用 *_bits 命名的列。最终每行是 (case, 组件, 流量 KiB)。
    """
    rows = []
    for row in read_csv(path):
        if int(float(row["bit_width"])) != 8 or int(float(row["n_tokens"])) != 196:
            continue  # 只保留论文使用的固定配置（8-bit、196 tokens）
        group = f"{row['variant'].replace('MobileViT-', '')} {row['weight_mode']}"
        for component, key in [
            ("input", "activation_input_bytes"),
            ("Q/K/V output", "qkv_projection_output_bytes"),
            ("attention score", "attention_score_bytes"),
            ("value product", "value_product_bytes"),
            ("electronic remainder", "electronic_remainder_bytes"),
            ("weight stream", "weight_stream_bytes"),
            ("programming", "programming_bytes"),
            ("bus", "bus_bytes"),
        ]:
            legacy_bits_key = {
                "activation_input_bytes": "input_bits",
                "qkv_projection_output_bytes": "qkv_output_bits",
                "weight_stream_bytes": "weight_stream_bits",
                "programming_bytes": "programming_bits",
                "bus_bytes": "bus_bits",
            }.get(key)
            if key in row:
                traffic_kib = float(row[key]) / 1024.0  # 新格式直接是字节数，除以 1024 转成 KiB
            elif legacy_bits_key and legacy_bits_key in row:
                traffic_kib = float(row[legacy_bits_key]) / 8.0 / 1024.0  # 旧格式是"位"：先 /8 得字节，再 /1024 得 KiB
            else:
                traffic_kib = 0.0
            rows.append(
                {
                    "case": group,
                    "traffic_component": component,
                    "traffic_kib": f"{traffic_kib:.6f}",
                }
            )
    return rows


def _nonideality_summary_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """汇总非理想性（噪声、串扰、量化等）造成的相对误差，用于"解析近似"版图。

    用途：当缺少带标签精度数据时，退而绘制"相对误差"灵敏度图。
    先把效果名翻译成图例用的短名（如 gaussian_pd_tia_noise → PD/TIA noise），
    再按 (效果, 模型变体) 取最大平均相对误差作为代表值。
    """
    short = {
        "gaussian_pd_tia_noise": "PD/TIA noise",
        "wdm_adjacent_crosstalk": "WDM crosstalk",
        "uniform_converter_quantization": "converter quantization",
        "insertion_loss": "insertion loss",
        "wavelength_detuning": "wavelength detuning",
        "mrr_variation": "MRR variation",
        "thermal_drift": "thermal drift",
    }
    buckets: dict[tuple[str, str], float] = {}
    for row in read_csv(path):
        key = (short.get(row["effect"], row["effect"]), row["variant"].replace("MobileViT-", ""))
        buckets[key] = max(buckets.get(key, 0.0), float(row["mean_relative_error"]))
    return [
        {
            "effect": effect,
            "variant": variant,
            "max_mean_relative_error": f"{value:.8f}",
        }
        for (effect, variant), value in sorted(buckets.items())
    ]


def _nonideality_accuracy_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """汇总非理想性扰动导致的 Top-1 精度下降（百分比）。

    用途：当实验数据里有带标签子集的实测精度（top1_delta 非空）时，
    按 (效果, 模型变体) 取最大 |精度下降| 作为代表值，
    供 "fig_nonideality_sensitivity" 图的"数据集精度下降"版本使用。
    """
    buckets: dict[tuple[str, str], float] = {}
    for row in read_csv(path):
        if int(float(row.get("label_count") or 0)) <= 0 or row.get("top1_delta") in ("", None):
            continue  # 跳过没有标签子集或没有精度差值的行
        key = (row["effect"], row["variant"].replace("MobileViT-", ""))
        buckets[key] = max(buckets.get(key, 0.0), abs(float(row["top1_delta"])))  # 取最大下降幅度
    return [
        {"effect": effect, "variant": variant, "top1_accuracy_drop_percent": f"{value:.6f}"}
        for (effect, variant), value in sorted(buckets.items())
    ]


def _scalability_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为可扩展性趋势图（fig_scalability_sweep）筛选数据行。

    用途：只保留固定硬件配置（2 个 PDPU 组、2 个 tile、16 路 ADC、
    128 Gbps 内存带宽、热校准倍数 1.0）的实验行，其余保持波长数变化，
    以便画出"波长数 → 延迟"的扩展趋势折线。
    """
    rows = []
    for row in read_csv(path):
        adc = row.get("adc_parallelism", row.get("adc_count_proxy", "0"))  # 兼容新旧列名
        memory_bw = row.get("memory_bandwidth_gbps", row.get("memory_bandwidth_budget_gbps", "0"))
        thermal = row.get("thermal_calibration_multiplier", "1.0")
        if (
            int(float(row["pdpu_banks"])) == 2
            and int(float(row["tiles"])) == 2
            and int(float(adc)) == 16
            and float(memory_bw) == 128.0
            and float(thermal) == 1.0
        ):
            rows.append(
                {
                    "panel": "PDPU banks=2, tiles=2, adc=16, mem=128Gbps",
                    "variant": row["variant"].replace("MobileViT-", ""),
                    "wavelengths": row["wavelengths"],
                    "latency_estimate_ns": row["latency_estimate_ns"],
                }
            )
    return rows


def _architecture_ablation_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为架构消融图（fig_architecture_ablation）筛选并换算数据。

    用途：只保留 wanted 里列出的消融配置（去掉广播、权重重编程、8/32 路
    WDM 波分复用、1/4 个 PDPU 组、4/12-bit 精度、纯光学乐观假设、100 次
    校准间隔等），并把"相对基线的能耗变化百分比"换算成相对基线百分比值
    （例如 +5% 存成 105），供横向分组条形图使用。
    """
    wanted = {
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
    }
    rows = []
    for row in read_csv(path):
        if row["ablation"] not in wanted:
            continue
        rows.append(
            {
                "ablation": row["ablation"],
                "variant": row["variant"].replace("MobileViT-", ""),
                "energy_vs_baseline_percent": f"{100.0 + float(row['delta_energy_vs_baseline_percent']):.6f}",
            }
        )
    return rows


def _speedup_bound_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为端到端加速比上界图（fig_e2e_speedup_bound）准备数据。

    用途：基于 Amdahl 定律（整体加速比受"电子剩余"串行部分限制），
    把光学加速假设倍数转成 log2 值（对数横轴让曲线更好读），
    再配上升加速比上界，绘制折线图。
    """
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "panel": "Amdahl bound from traced electronic remainder",
                "model": row["model"].replace("MobileViT-", ""),
                "optical_speedup_assumption": row["optical_speedup_assumption"],
                "optical_speedup_log2": f"{math.log2(float(row['optical_speedup_assumption'])):.6f}",  # 加速假设取 log2，便于对数横轴展示
                "e2e_speedup_upper_bound": row["e2e_speedup_upper_bound"],
            }
        )
    return rows


def _energy_uncertainty_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为能量不确定性包络图（fig_energy_uncertainty_sweep）准备数据。

    用途：把能量不确定性汇总表原样转成统一的
    (model, case, metric, unit, energy_mj) 行结构，
    供分组指标面板图（按模型分组展示不同扰动情形下的能量范围）使用。
    """
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "model": row["model"],
                "case": row["case"],
                "metric": "energy uncertainty",
                "unit": "mJ",
                "energy_mj": row["energy_mj"],
            }
        )
    return rows


def _layout_area_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为布局/面积可行性图（fig_layout_area_feasibility_proxy）准备数据。

    用途：只保留固定配置（16 波长、2 个 PDPU 组、2 个 tile）的布局估算行，
    并把 MRR（微环谐振器）、转换器（光电/电光转换器）、互连三部分的
    面积代理值拆成多行，供堆叠柱状图展示各部分面积占比。
    """
    rows = []
    for row in read_csv(path):
        if (
            int(float(row["wavelengths"])) != 16
            or int(float(row["pdpu_banks"])) != 2
            or int(float(row["tiles"])) != 2
        ):
            continue
        case = row["variant"].replace("MobileViT-", "")
        for component, key in [
            ("MRR proxy", "mrr_area_proxy_mm2"),
            ("converter proxy", "converter_area_proxy_mm2"),
            ("interconnect proxy", "interconnect_area_proxy_mm2"),
        ]:
            rows.append({"case": case, "area_component": component, "area_mm2": row[key]})
    return rows


def _thermal_stress_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为热调谐压力图（fig_thermal_tuning_stress）筛选数据行。

    用途：只保留固定重调间隔（每 1000 次推理重调一次）且热校准倍数为 1.0
    的仿真行，这样横轴就只剩下热漂移温度（thermal_drift_c）在变化，
    便于观察温度漂移对能量开销代理值的影响。
    """
    rows = []
    for row in read_csv(path):
        if int(float(row["retune_interval_inferences"])) != 1000:
            continue  # 固定重调间隔
        if abs(float(row["thermal_calibration_multiplier"]) - 1.0) > 1e-9:
            continue  # 固定热校准倍数（浮点比较用容差 1e-9）
        rows.append(
            {
                "panel": "retune interval=1000, multiplier=1.0",
                "variant": row["variant"].replace("MobileViT-", ""),
                "thermal_drift_c": row["thermal_drift_c"],
                "energy_overhead_mj_proxy": row["energy_overhead_mj_proxy"],
            }
        )
    return rows


def _additional_family_figure_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """为额外模型家族的映射检查图（fig_additional_model_family_mapping）准备数据。

    用途：跳过没有有效追踪行（row_count<=0）的模型，把每个模型变体在
    三种 MAC 占比指标（PDPU 候选 / 电子剩余 / 混合支持）上的百分比
    拆成多行，供横向分组条形图展示。
    """
    rows = []
    for row in read_csv(path):
        if int(float(row.get("row_count") or 0)) <= 0:
            continue  # 没有追踪数据的模型不画
        for metric, key in [
            ("PDPU-candidate MAC share", "pdpu_candidate_mac_share_percent"),
            ("electronic remainder MAC share", "electronic_remainder_mac_share_percent"),
            ("hybrid support MAC share", "hybrid_support_mac_share_percent"),
        ]:
            rows.append({"variant": row["variant"], "metric": metric, "share_percent": row[key]})
    return rows


def run(output_dir: pathlib.Path) -> dict[str, Any]:
    """渲染全部论文实验图的主流程，返回渲染结果清单。

    参数:
        output_dir: 实验输出根目录；tables/ 是输入数据，figures/ 写图片。

    返回:
        包含三部分的字典：manifest 为渲染清单 JSON 路径；
        rendered 为成功渲染的图（含 sidecar）列表；skipped 为缺数据被跳过的图列表。

    流程：对每张论文图，先检查输入 CSV 是否存在 → 预处理成绘图行 →
    调用 hpat_eval.plotting 里的绘图函数画 PNG → 用 figure_sidecar 生成
    配套元数据（数据来源、证据标签、图注）→ 记入 rendered；
    输入缺失的图记入 skipped。
    """
    output_dir = output_dir.resolve()
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")  # 存放输入实验表的目录
    fig_dir = ensure_dir(output_dir / "figures")  # 输出 PNG 插图的目录
    manifest = base_manifest("render_experiment_figures", "experiment-result figure rendering")
    rendered: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    # 检查绘图依赖 PIL/Pillow（图像底层库）是否可用；不可用则整批跳过并写 blocked
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        manifest.update({"status": "blocked", "blocked_reason": f"PIL/Pillow unavailable: {exc}"})
        manifest_path = output_dir / "render_experiment_figures_manifest.json"
        write_json(manifest_path, manifest)
        return {"manifest": manifest_path, "rendered": rendered, "skipped": skipped}

    # 图 1：混合证据的 MobileViT 对比背景图（延迟/能耗/功耗，多指标分组面板）
    fig6_csv = tables_dir / "fig6_evidence_repaired.csv"
    if _exists(fig6_csv):
        rows = read_csv(fig6_csv)
        source = _copy_rows(fig_dir / "fig6_mixed_evidence_context.csv", rows, list(rows[0].keys()))
        png = grouped_metric_panels(
            fig_dir / "fig6_mixed_evidence_context.png",
            rows,
            title="Mixed-evidence MobileViT comparison context",
            group_key="model_variant",
            series_key="platform_label",
            metric_key="metric",
            value_key="value",
            unit_key="unit",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, fig6_csv],
            evidence_label="mixed: provisional / measured desktop reference / modelled",
            caption="Mixed-evidence latency, energy, and power context. HPAT rows are estimated/modelled, not measured edge or silicon results.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig6_mixed_evidence_context", "reason": "missing tables/fig6_evidence_repaired.csv"})

    # 图 2：系统级 HPAT 能耗分解堆叠柱状图（按组件分组、按模型堆叠）
    energy_rows, energy_source = _energy_rows(tables_dir)
    if energy_rows and energy_source:
        # 判断数据是否"已标定"：来自仿真导出且归一化因子全为 1.0，据此选择不同的图注文案
        calibrated = any(row.get("activity_source") == "simulator_export" for row in energy_rows) and all(
            float(row.get("normalization_factor") or 1.0) == 1.0 for row in energy_rows
        )
        energy_evidence_label = (
            "modelled from simulator activity counts and explicit unit costs"
            if calibrated
            else "modelled / normalized energy allocation"
        )
        energy_caption = (
            "System-level HPAT energy breakdown by component, modelled from simulator activity counts and explicit unit costs. This is not silicon-measured energy."
            if calibrated
            else "System-level HPAT energy breakdown by component. Values are modelled and normalized to the HPAT energy envelope unless simulator activity is provided."
        )
        source = _copy_rows(fig_dir / "fig_energy_breakdown_stacked.csv", energy_rows, list(energy_rows[0].keys()))
        png = stacked_bar(
            fig_dir / "fig_energy_breakdown_stacked.png",
            energy_rows,
            title="System-level HPAT energy breakdown",
            group_key="model_variant",
            stack_key="component_group",
            value_key="energy_mj",
            ylabel="Energy (mJ)",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, energy_source],
            evidence_label=energy_evidence_label,
            caption=energy_caption,
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_energy_breakdown_stacked", "reason": "missing energy CSV"})

    # 图 3：MobileViT 算子域分解堆叠图（PDPU 候选线性/MVM vs 电子剩余）
    op_csv = tables_dir / "mobilevit_operator_activity.csv"
    if _exists(op_csv):
        rows = _operator_domain_rows(op_csv)
        source = _copy_rows(fig_dir / "fig_operator_domain_breakdown.csv", rows, list(rows[0].keys()))
        png = stacked_bar(
            fig_dir / "fig_operator_domain_breakdown.png",
            rows,
            title="MobileViT operator-domain breakdown",
            group_key="model_variant",
            stack_key="execution_domain",
            value_key="estimated_macs",
            ylabel="Estimated MACs",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, op_csv],
            evidence_label="local trace or model/config-derived proxy",
            caption="Operator-domain breakdown showing PDPU-candidate linear/MVM work versus electronic remainder. Proxy rows must not be described as measured MobileViT runtime.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_operator_domain_breakdown", "reason": "missing mobilevit_operator_activity.csv"})

    # 图 4：Q/K/V 流量敏感性堆叠图（8-bit、196 tokens 固定配置）
    qkv_csv = _first_existing(
        [tables_dir / "qkv_traffic_calibrated.csv", tables_dir / "qkv_traffic_sensitivity.csv"]
    )
    if qkv_csv:
        rows = _qkv_figure_rows(qkv_csv)
        source = _copy_rows(fig_dir / "fig_qkv_traffic_sensitivity.csv", rows, list(rows[0].keys()))
        png = stacked_bar(
            fig_dir / "fig_qkv_traffic_sensitivity.png",
            rows,
            title="Q/K/V traffic sensitivity at 8-bit, N=196",
            group_key="case",
            stack_key="traffic_component",
            value_key="traffic_kib",
            ylabel="Traffic (KiB)",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, qkv_csv]
            + _existing(
                [
                    tables_dir / "mobilevit_operator_activity.csv",
                    tables_dir / "p1_readiness_summary.json",
                ]
            ),
            evidence_label="local/modelled P1 Q/K/V traffic calibration",
            caption="Q/K/V activation, attention, value-product, electronic-remainder, weight, programming, and bus traffic. This is modelled/proxy, not measured memory energy.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_qkv_traffic_sensitivity", "reason": "missing qkv_traffic_sensitivity.csv"})

    # 图 5：非理想性灵敏度横向分组条形图（优先画精度下降，退而画相对误差）
    non_csv = tables_dir / "nonideality_accuracy_sweep.csv"
    if _exists(non_csv):
        accuracy_rows = _nonideality_accuracy_rows(non_csv)
        rows = accuracy_rows or _nonideality_summary_rows(non_csv)
        source = _copy_rows(fig_dir / "fig_nonideality_sensitivity.csv", rows, list(rows[0].keys()))
        png = horizontal_grouped_bar(
            fig_dir / "fig_nonideality_sensitivity.png",
            rows,
            title="Dataset accuracy drop under selected non-idealities" if accuracy_rows else "Selected non-ideality and analytical sensitivity proxy",
            group_key="effect",
            series_key="variant",
            value_key="top1_accuracy_drop_percent" if accuracy_rows else "max_mean_relative_error",
            xlabel="Worst top-1 accuracy drop (%)" if accuracy_rows else "Worst mean relative error across configured sweep",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, non_csv],
            evidence_label="dataset-coupled MobileViT accuracy subset" if accuracy_rows else "modelled selected non-ideality / analytical sensitivity",
            caption=(
                "Top-1 accuracy drop on a fixed labeled MobileViT subset under selected non-ideality perturbations. Not silicon robustness."
                if accuracy_rows
                else "Sensitivity proxy across noise, crosstalk, precision, insertion loss, detuning, MRR variation, and thermal drift. Not silicon robustness."
            ),
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_nonideality_sensitivity", "reason": "missing nonideality_accuracy_sweep.csv"})

    # 图 6：可扩展性趋势折线图（固定硬件配置下，波长数 → 延迟）
    scale_csv = _first_existing(
        [tables_dir / "scalability_physical_proxy.csv", tables_dir / "scalability_sweep.csv"]
    )
    if scale_csv:
        rows = _scalability_figure_rows(scale_csv)
        source = _copy_rows(fig_dir / "fig_scalability_sweep.csv", rows, list(rows[0].keys()))
        png = line_panel(
            fig_dir / "fig_scalability_sweep.png",
            rows,
            title="Modelled scalability trend across WDM configurations",
            x_key="wavelengths",
            y_key="latency_estimate_ns",
            series_key="variant",
            panel_key="panel",
            ylabel="Latency (ns)",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, scale_csv, * _existing([tables_dir / "p1_readiness_summary.json"])],
            evidence_label="local/modelled scalability physical proxy",
            caption="Modelled latency trend across wavelength counts for a fixed PDPU/memory configuration with resource proxies. Not physical layout closure or fabricated deployment evidence.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_scalability_sweep", "reason": "missing scalability_sweep.csv"})

    # 图 7：架构消融对比（相对基线的能耗百分比）横向条形图
    ablation_csv = tables_dir / "hpat_architecture_ablation.csv"
    if _exists(ablation_csv):
        rows = _architecture_ablation_figure_rows(ablation_csv)
        source = _copy_rows(fig_dir / "fig_architecture_ablation.csv", rows, list(rows[0].keys()))
        png = horizontal_grouped_bar(
            fig_dir / "fig_architecture_ablation.png",
            rows,
            title="Modelled HPAT architecture ablation",
            group_key="ablation",
            series_key="variant",
            value_key="energy_vs_baseline_percent",
            xlabel="Energy vs baseline (%)",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, ablation_csv, * _existing([tables_dir / "p1_readiness_summary.json"])],
            evidence_label="local/modelled P1 architecture ablation",
            caption="Modelled HPAT architecture ablation relative to the resident-weight, broadcast-enabled baseline. This is not measured hardware evidence.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_architecture_ablation", "reason": "missing hpat_architecture_ablation.csv"})

    # 图 8：端到端加速比上界折线图（Amdahl 定律界，横轴为 log2 光学加速假设）
    speed_csv = tables_dir / "e2e_speedup_bound.csv"
    if _exists(speed_csv):
        rows = _speedup_bound_rows(speed_csv)
        source = _copy_rows(fig_dir / "fig_e2e_speedup_bound.csv", rows, list(rows[0].keys()))
        png = line_panel(
            fig_dir / "fig_e2e_speedup_bound.png",
            rows,
            title="End-to-end speedup bound from electronic remainder",
            x_key="optical_speedup_log2",
            y_key="e2e_speedup_upper_bound",
            series_key="model",
            panel_key="panel",
            ylabel="E2E speedup upper bound",
            xlabel="log2(optical speedup assumption)",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, speed_csv],
            evidence_label="Amdahl-style bound from local hook trace",
            caption="Upper bound on end-to-end speedup under optical speedup assumptions, limited by traced electronic remainder. Not measured HPAT speedup.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_e2e_speedup_bound", "reason": "missing e2e_speedup_bound.csv"})

    # 图 9：能量不确定性包络（组件乘子扰动下的能量范围）分组指标图
    uncertainty_csv = tables_dir / "energy_uncertainty_summary.csv"
    full_uncertainty_csv = tables_dir / "energy_uncertainty_sweep.csv"
    if _exists(uncertainty_csv):
        rows = _energy_uncertainty_figure_rows(uncertainty_csv)
        source = _copy_rows(fig_dir / "fig_energy_uncertainty_sweep.csv", rows, list(rows[0].keys()))
        png = grouped_metric_panels(
            fig_dir / "fig_energy_uncertainty_sweep.png",
            rows,
            title="Modelled HPAT energy uncertainty envelope",
            group_key="model",
            series_key="case",
            metric_key="metric",
            value_key="energy_mj",
            unit_key="unit",
        )
        sidecar = figure_sidecar(
            output_path=png,
            repo_root=output_dir,
            source_paths=[source, uncertainty_csv, full_uncertainty_csv]
            + _existing(
                [
                    tables_dir / "energy_unit_cost_source_ledger.csv",
                    tables_dir / "p1_readiness_summary.json",
                ]
            ),
            evidence_label="modelled component uncertainty sensitivity",
            caption="Component multiplier sensitivity around the normalized HPAT energy model. This is not calibrated silicon energy.",
        )
        rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
    else:
        skipped.append({"figure": "fig_energy_uncertainty_sweep", "reason": "missing energy_uncertainty_summary.csv"})

    # 图 10：P2 布局/面积可行性代理堆叠图（MRR、转换器、互连三部分面积）
    layout_csv = tables_dir / "layout_area_feasibility_proxy.csv"
    if _exists(layout_csv):
        rows = _layout_area_figure_rows(layout_csv)
        if rows:
            source = _copy_rows(fig_dir / "fig_layout_area_feasibility_proxy.csv", rows, list(rows[0].keys()))
            png = stacked_bar(
                fig_dir / "fig_layout_area_feasibility_proxy.png",
                rows,
                title="P2 layout/area feasibility proxy",
                group_key="case",
                stack_key="area_component",
                value_key="area_mm2",
                ylabel="Area proxy (mm2)",
            )
            sidecar = figure_sidecar(
                output_path=png,
                repo_root=output_dir,
                source_paths=[source, layout_csv, *_existing([tables_dir / "p2_readiness_summary.json"])],
                evidence_label="local/modelled P2 layout-area proxy",
                caption="Layout/area feasibility proxy for the fixed 16-wavelength, 2-bank, 2-tile configuration. This is not physical-design closure.",
            )
            rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
        else:
            skipped.append({"figure": "fig_layout_area_feasibility_proxy", "reason": "no fixed 16wl/2bank/2tile rows"})
    else:
        skipped.append({"figure": "fig_layout_area_feasibility_proxy", "reason": "missing layout_area_feasibility_proxy.csv"})

    # 图 11：P2 热调谐压力代理折线图（热漂移温度 → 能量开销代理值）
    thermal_csv = tables_dir / "thermal_tuning_stress.csv"
    if _exists(thermal_csv):
        rows = _thermal_stress_figure_rows(thermal_csv)
        if rows:
            source = _copy_rows(fig_dir / "fig_thermal_tuning_stress.csv", rows, list(rows[0].keys()))
            png = line_panel(
                fig_dir / "fig_thermal_tuning_stress.png",
                rows,
                title="P2 thermal tuning stress proxy",
                x_key="thermal_drift_c",
                y_key="energy_overhead_mj_proxy",
                series_key="variant",
                panel_key="panel",
                ylabel="Energy overhead proxy (mJ)",
                xlabel="Thermal drift (C)",
            )
            sidecar = figure_sidecar(
                output_path=png,
                repo_root=output_dir,
                source_paths=[source, thermal_csv, *_existing([tables_dir / "p2_readiness_summary.json"])],
                evidence_label="local/modelled P2 thermal tuning stress proxy",
                caption="Thermal tuning stress proxy across configured drift values at fixed retune interval and multiplier. This is not packaged-device thermal validation.",
            )
            rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
        else:
            skipped.append({"figure": "fig_thermal_tuning_stress", "reason": "no fixed retune/multiplier rows"})
    else:
        skipped.append({"figure": "fig_thermal_tuning_stress", "reason": "missing thermal_tuning_stress.csv"})

    # 图 12：P2 额外模型家族 MAC 占比横向条形图（映射合理性检查）
    family_csv = tables_dir / "additional_model_family_operator_summary.csv"
    if _exists(family_csv):
        rows = _additional_family_figure_rows(family_csv)
        if rows:
            source = _copy_rows(fig_dir / "fig_additional_model_family_mapping.csv", rows, list(rows[0].keys()))
            png = horizontal_grouped_bar(
                fig_dir / "fig_additional_model_family_mapping.png",
                rows,
                title="P2 additional model-family mapping check",
                group_key="variant",
                series_key="metric",
                value_key="share_percent",
                xlabel="MAC share (%)",
            )
            sidecar = figure_sidecar(
                output_path=png,
                repo_root=output_dir,
                source_paths=[source, family_csv, *_existing([tables_dir / "p2_readiness_summary.json"])],
                evidence_label="local/modelled P2 additional-family mapping proxy",
                caption="Additional-family hook trace summary showing PDPU-candidate and electronic-remainder MAC shares. This supports bounded mapping plausibility only.",
            )
            rendered.append({"figure": _run_record(png, output_dir), "sidecar": _run_record(sidecar, output_dir)})
        else:
            skipped.append({"figure": "fig_additional_model_family_mapping", "reason": "no trace-backed additional-family rows"})
    else:
        skipped.append({"figure": "fig_additional_model_family_mapping", "reason": "missing additional_model_family_operator_summary.csv"})

    manifest_path = output_dir / "render_experiment_figures_manifest.json"
    manifest.update(  # 汇总渲染结果：状态、图清单、跳过清单
        {
            "status": "ok",
            "figures_dir": _run_record(fig_dir, output_dir),
            "rendered": rendered,
            "skipped": skipped,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; figure rendering is short and deterministic.",
        }
    )
    write_json(manifest_path, manifest)
    return {"manifest": manifest_path, "rendered": rendered, "skipped": skipped}


def main() -> None:
    """命令行入口：解析 --output-dir，调用 run() 并把结果 JSON 打印到标准输出。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    # --config 仅为兼容 run_all 批量入口，本脚本不使用
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir))  # 执行全部渲染
    # 输出结果里 pathlib.Path 转成相对路径，再以 JSON 打印便于流水线解析
    print(json.dumps({k: relative(v) if isinstance(v, pathlib.Path) else v for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()  # 以脚本方式直接运行时，从这里进入
