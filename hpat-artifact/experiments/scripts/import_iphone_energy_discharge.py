"""导入 iPhone 电池放电实测数据，用手机放电曲线校准（标定）边缘设备的能耗模型。

做什么：读取 MobileViTCoreMLBench 采集的 iPhone 电池放电日志（summary CSV 阶段汇总表
       和可选的 samples CSV 细粒度采样表），把"电池电量下降比例"换算成每个模型测试
       阶段的能耗估计（能量 J、平均功率 W、单次推理毫焦 mJ），并扣除空闲阶段
       （什么都不跑时的系统耗电）的影响，输出汇总 CSV 和记录元信息的 JSON 清单。

数据从哪来：
    --summary-csv：每个测试阶段的汇总（phase 行，含 idle/active 标记、时长、
                   电池电量变化、推理次数、平均延迟等）。
    --samples-csv（可选）：更细的电池采样点（battery_state、battery_level）。
    --capacity-wh：电池容量（瓦时 Wh），用于把"电量比例"换算成能量；
       也可用 --nominal-capacity-wh × --battery-health-factor（电池健康度系数）
       折算出实际容量。

输出到哪：--out-dir 目录下的 iphone_energy_discharge_summary.csv（能耗汇总表）
        和 iphone_energy_discharge_manifest.json（元信息清单）。

怎么运行（示例）：
    python import_iphone_energy_discharge.py \
        --summary-csv run/summary.csv --samples-csv run/samples.csv \
        --nominal-capacity-wh 12.9 --battery-health-factor 0.91 \
        --out-dir run/imported

注意事项：本脚本产出的是"电池放电曲线"级别的粗粒度估计（手机整机耗电），
        不是外部功率计（power meter）校准的精确读数，只能作为边缘设备基线
       （baseline，即对比用的基准数据）的参考，不能当作 HPAT（光子张量处理器）
        本身或芯片某条供电轨的能耗证据。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from statistics import fmean
from typing import Any


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 CSV 文件并返回"字典列表"（每行一个字典，列名作键）。

    参数：path —— 要读取的 CSV 文件路径。
    返回：list[dict[str, str]] —— 每行数据表示为一个 {列名: 单元格文本} 的字典。
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        # csv.DictReader 会把首行当作列名，之后逐行解析成字典
        return list(csv.DictReader(f))


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """把字典列表按指定列顺序写成 CSV 文件。

    参数：path —— 输出文件路径（会自动创建父目录）；
         rows —— 要写入的行（每行一个字典）；
         fields —— 列名顺序（同时决定表头的列顺序）。
    返回：无。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()  # 先写表头
        writer.writerows(rows)  # 再写所有数据行


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    """安全地把单元格文本转成浮点数；空字符串返回默认值（避免脏数据导致报错）。

    参数：row —— 一行字典；key —— 列名；default —— 单元格为空时的回退值。
    返回：解析出的浮点数。
    """
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    """安全地把单元格文本转成整数；空字符串返回默认值。

    注意：先转 float 再转 int，可兼容 "1.0" 这类带小数点的文本。
    参数：row —— 一行字典；key —— 列名；default —— 单元格为空时的回退值。
    返回：解析出的整数。
    """
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def summarize(
    summary_csv: pathlib.Path,
    samples_csv: pathlib.Path | None,
    capacity_wh: float | None,
    nominal_capacity_wh: float | None = None,
    battery_health_factor: float | None = None,
    capacity_reference_url: str = "",
    low_battery_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """核心汇总函数：把 iPhone 电池放电日志换算成能耗估计（校准模型的核心逻辑）。

    做什么（分步）：
      1) 读取 summary CSV，把 idle（空闲）和 active（跑模型）两类 phase（阶段）分开；
      2) 用 idle phase 的"电量变化比例 ÷ 时长"算出全局空闲耗电速率（idle_rate），
         代表手机不跑模型时每秒自然掉电的比例；
      3) 对每个 active phase，用"总电量下降 − 空闲等效电量下降"得到模型自身造成的
         电量下降（model_delta），再乘电池容量换算成能量/功率；
      4) 输出四类聚合行：单个 phase、按相同模型配置分组的聚合、整个 active 时间段
         合并的"综合窗口"行、以及按时长分摊到各模型配置的行；
      5) 同时生成 manifest（元信息清单），记录采样点状态、低电量告警等质量控制信息。

    参数：
        summary_csv —— 阶段汇总 CSV 路径（必填）。
        samples_csv —— 细粒度采样 CSV 路径（可选，用于统计电池采样状态与电量点）。
        capacity_wh —— 电池实际容量（瓦时）。为 None 时只保留"电量比例"证据，
                       不换算成焦耳/瓦特。
        nominal_capacity_wh —— 出厂标称容量（可选，用于健康度修正）。
        battery_health_factor —— 电池健康度系数（如 0.91 表示最大容量只剩 91%）。
        capacity_reference_url —— 容量/健康度数据来源的参考链接（可选，写进清单）。
        low_battery_threshold —— 低电量阈值（默认 0.20，即电量 20% 以下视为低电量）。

    返回：
        (out_rows, manifest) 二元组：
        out_rows —— 汇总行列表（供写 CSV）；
        manifest —— 记录运行质量、容量来源、边界声明的字典（供写 JSON）。
    """
    # 第一步：读取所有阶段行，并给每一行追加 _row_index，便于之后回溯定位源行
    rows = []
    for index, row in enumerate(_read_csv(summary_csv)):
        indexed = dict(row)
        indexed["_row_index"] = str(index)
        rows.append(indexed)
    # 按 phase 列把行分成"空闲行"和"跑模型行"两组
    idle_rows = [row for row in rows if row.get("phase") == "idle"]
    active_rows = [row for row in rows if row.get("phase") == "active"]
    # 没有空闲/活动行就无法做"扣除空闲"的校准，直接报错说明
    if not idle_rows:
        raise ValueError("summary CSV has no idle phase row")
    if not active_rows:
        raise ValueError("summary CSV has no active phase rows")

    # 第二步：计算全局空闲耗电速率（单位：电量比例/秒）
    idle_rates = []
    for row in idle_rows:
        duration_s = _float(row, "duration_s")
        delta = _float(row, "battery_delta_fraction")
        if duration_s > 0:
            idle_rates.append(delta / duration_s)  # 每次空闲阶段自己的掉电速率
    idle_rate = fmean(idle_rates) if idle_rates else 0.0  # 取所有空闲速率的均值

    # 第三步：统计细粒度采样数据（如果提供了 samples_csv），用于质量控制信息
    sample_rows = _read_csv(samples_csv) if samples_csv else []
    sample_count = 0
    # 采样点出现的电池状态集合（如充电/放电/满电等）
    sample_battery_states = sorted({row.get("battery_state", "") for row in sample_rows}) if samples_csv else []
    # 采样点电量值列表（排除空值）
    sample_battery_levels = [float(row["battery_level"]) for row in sample_rows if row.get("battery_level", "") != ""]
    # 去重并按从高到低排序，方便看整段测试覆盖的电量范围
    unique_sample_battery_levels = sorted(set(sample_battery_levels), reverse=True)
    # 统计电量值发生变化的次数（相邻采样点不同则算一次"跳变"）
    battery_transition_count = sum(
        1 for previous, current in zip(sample_battery_levels, sample_battery_levels[1:]) if current != previous
    )
    min_sample_battery_level = min(sample_battery_levels) if sample_battery_levels else None
    max_sample_battery_level = max(sample_battery_levels) if sample_battery_levels else None
    # 是否观测到低电量区间（iOS 在低电量时会干预功耗，结果可信度要打折）
    low_battery_observed = min_sample_battery_level is not None and min_sample_battery_level <= low_battery_threshold
    # 有多少 active 阶段是在低电量区间内开始或结束的
    low_battery_active_phase_count = sum(
        1
        for row in active_rows
        if _float(row, "start_battery_level") <= low_battery_threshold
        or _float(row, "end_battery_level") <= low_battery_threshold
    )
    if samples_csv:
        sample_count = len(sample_rows)

    # 第四步：判断"阶段归属"是否可信。若既有电量下降>0 的阶段又有=0 的阶段，
    # 说明电池电量步进（iOS 量化粒度）太粗，无法把电量变化准确归因到具体阶段
    nonzero_active_delta_count = sum(_float(row, "battery_delta_fraction") > 0 for row in active_rows)
    zero_active_delta_count = sum(_float(row, "battery_delta_fraction") == 0 for row in active_rows)
    phase_attribution_uncertain = bool(nonzero_active_delta_count and zero_active_delta_count)

    # 第五步：逐阶段计算能耗并生成"单 phase 级"汇总行
    out_rows: list[dict[str, Any]] = []
    phase_records: list[dict[str, Any]] = []  # 保留中间量，供后面的分组/合并聚合用
    for row in active_rows:
        duration_s = _float(row, "duration_s")
        raw_delta = _float(row, "battery_delta_fraction")  # 该阶段实际掉电比例
        idle_delta = idle_rate * duration_s  # 空闲等效掉电比例（若没跑模型会掉多少）
        model_delta = max(raw_delta - idle_delta, 0.0)  # 扣除空闲后，模型自身造成的掉电比例
        iterations = _int(row, "iterations")
        # 电量比例 × 电池容量(Wh) × 3600 秒/小时 → 能量（焦耳 J）
        energy_j = model_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        raw_energy_j = raw_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        idle_energy_j = idle_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        # 能量 ÷ 时长 → 平均功率（瓦 W）
        raw_avg_power_w = raw_energy_j / duration_s if raw_energy_j is not None and duration_s > 0 else None
        idle_equivalent_avg_power_w = idle_energy_j / duration_s if idle_energy_j is not None and duration_s > 0 else None
        idle_subtracted_avg_power_w = energy_j / duration_s if energy_j is not None and duration_s > 0 else None
        # 能量 ÷ 推理次数 → 单次推理能耗（毫焦 mJ）
        energy_mj_per_inference = (energy_j * 1000.0 / iterations) if energy_j is not None and iterations > 0 else None
        # 给每行打状态标签：可信度不足时如实标记，而不是假装精确
        if model_delta > 0 and iterations > 0 and phase_attribution_uncertain:
            status = "coarse_battery_step_phase_attribution_uncertain"  # 电量步进粗、归属不确定
        elif model_delta > 0 and iterations > 0:
            status = "ok"  # 正常可用
        else:
            status = "insufficient_battery_resolution"  # 电池分辨率不足，算不出有效能耗

        out_rows.append(
            {
                "model_variant": row.get("model_variant", ""),
                "input_shape": row.get("input_shape", ""),
                "runtime": row.get("runtime", ""),
                "precision": row.get("precision", ""),
                "compute_units": row.get("compute_units", ""),
                "compute_unit_scope": row.get("compute_unit_scope", ""),
                "compute_unit_claim_boundary": row.get("compute_unit_claim_boundary", ""),
                "aggregation_level": "phase",
                "source_phase_count": 1,
                "idle_subtraction_source": "global_idle_mean",
                "duration_s": f"{duration_s:.6f}",
                "iterations": iterations,
                "mean_latency_ms": row.get("mean_latency_ms", ""),
                "start_battery_level": row.get("start_battery_level", ""),
                "end_battery_level": row.get("end_battery_level", ""),
                "raw_battery_delta_fraction": f"{raw_delta:.8f}",
                "idle_rate_fraction_per_s": f"{idle_rate:.12f}",
                "idle_equivalent_delta_fraction": f"{idle_delta:.8f}",
                "idle_subtracted_delta_fraction": f"{model_delta:.8f}",
                "battery_capacity_wh": "" if capacity_wh is None else f"{capacity_wh:.6f}",
                "raw_avg_power_w_estimate": "" if raw_avg_power_w is None else f"{raw_avg_power_w:.6f}",
                "idle_equivalent_avg_power_w_estimate": "" if idle_equivalent_avg_power_w is None else f"{idle_equivalent_avg_power_w:.6f}",
                "idle_subtracted_avg_power_w_estimate": "" if idle_subtracted_avg_power_w is None else f"{idle_subtracted_avg_power_w:.6f}",
                "energy_j_estimate": "" if energy_j is None else f"{energy_j:.6f}",
                "energy_mj_per_inference_estimate": "" if energy_mj_per_inference is None else f"{energy_mj_per_inference:.6f}",
                "status": status,
                "evidence_label": "iPhone battery-discharge energy estimate with idle subtraction; not external power-meter calibrated",
                "claim_boundary": "Use as edge baseline context only. Do not present as HPAT energy or calibrated phone SoC rail energy.",
            }
        )
        # 保存中间量，供后续"同配置分组聚合 / 综合窗口合并"复用
        phase_records.append(
            {
                "source_row": row,
                "duration_s": duration_s,
                "raw_delta": raw_delta,
                "idle_delta": idle_delta,
                "model_delta": model_delta,
                "iterations": iterations,
                "mean_latency_ms": _float(row, "mean_latency_ms") if str(row.get("mean_latency_ms", "")).strip() else None,
                "status": status,
            }
        )

    # 第六步：把"重复测同一个模型配置"的多阶段聚合起来（同一配置出现多次时，
    # 单阶段的电池步进噪声会互相抵消，聚合结果更稳）
    grouped_phase_records: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in phase_records:
        row = record["source_row"]
        # 用模型名称/输入形状/运行时/精度/计算单元等字段组成"分组键"
        key = (
            row.get("model_variant", ""),
            row.get("input_shape", ""),
            row.get("runtime", ""),
            row.get("precision", ""),
            row.get("compute_units", ""),
            row.get("compute_unit_scope", ""),
            row.get("compute_unit_claim_boundary", ""),
        )
        grouped_phase_records.setdefault(key, []).append(record)
    for key, records in sorted(grouped_phase_records.items()):
        model_variant, input_shape, runtime, precision, compute_units, compute_scope, compute_boundary = key
        # 组内各量求和（时长、电量变化、次数），再按同一套公式换算能耗
        duration_s = sum(record["duration_s"] for record in records)
        raw_delta = sum(record["raw_delta"] for record in records)
        idle_delta = sum(record["idle_delta"] for record in records)
        model_delta = max(sum(record["model_delta"] for record in records), 0.0)
        iterations = sum(record["iterations"] for record in records)
        energy_j = model_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        raw_energy_j = raw_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        idle_energy_j = idle_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        raw_avg_power_w = raw_energy_j / duration_s if raw_energy_j is not None and duration_s > 0 else None
        idle_equivalent_avg_power_w = idle_energy_j / duration_s if idle_energy_j is not None and duration_s > 0 else None
        idle_subtracted_avg_power_w = energy_j / duration_s if energy_j is not None and duration_s > 0 else None
        energy_mj_per_inference = (energy_j * 1000.0 / iterations) if energy_j is not None and iterations > 0 else None
        # 平均延迟按"推理次数"加权平均，推理次数多的阶段权重更大
        latency_weight = sum(record["iterations"] for record in records if record["mean_latency_ms"] is not None)
        mean_latency_ms = (
            sum(record["mean_latency_ms"] * record["iterations"] for record in records if record["mean_latency_ms"] is not None) / latency_weight
            if latency_weight > 0
            else None
        )
        out_rows.append(
            {
                "model_variant": model_variant,
                "input_shape": input_shape,
                "runtime": runtime,
                "precision": precision,
                "compute_units": compute_units,
                "compute_unit_scope": compute_scope,
                "compute_unit_claim_boundary": compute_boundary,
                "aggregation_level": "paired_group_aggregate",
                "source_phase_count": len(records),
                "idle_subtraction_source": "global_idle_mean",
                "duration_s": f"{duration_s:.6f}",
                "iterations": iterations,
                "mean_latency_ms": "" if mean_latency_ms is None else f"{mean_latency_ms:.6f}",
                "start_battery_level": "",
                "end_battery_level": "",
                "raw_battery_delta_fraction": f"{raw_delta:.8f}",
                "idle_rate_fraction_per_s": f"{idle_rate:.12f}",
                "idle_equivalent_delta_fraction": f"{idle_delta:.8f}",
                "idle_subtracted_delta_fraction": f"{model_delta:.8f}",
                "battery_capacity_wh": "" if capacity_wh is None else f"{capacity_wh:.6f}",
                "raw_avg_power_w_estimate": "" if raw_avg_power_w is None else f"{raw_avg_power_w:.6f}",
                "idle_equivalent_avg_power_w_estimate": "" if idle_equivalent_avg_power_w is None else f"{idle_equivalent_avg_power_w:.6f}",
                "idle_subtracted_avg_power_w_estimate": "" if idle_subtracted_avg_power_w is None else f"{idle_subtracted_avg_power_w:.6f}",
                "energy_j_estimate": "" if energy_j is None else f"{energy_j:.6f}",
                "energy_mj_per_inference_estimate": "" if energy_mj_per_inference is None else f"{energy_mj_per_inference:.6f}",
                "status": "paired_group_aggregate" if model_delta > 0 and iterations > 0 else "insufficient_battery_resolution",
                "evidence_label": "Grouped iPhone battery-discharge estimate over repeated same model/compute-unit phases; not external power-meter calibrated",
                "claim_boundary": "Grouped phone-level discharge context only. Do not present as pure GPU/ANE power, HPAT energy, or calibrated SoC rail energy.",
            }
        )

    # 第七步：整个 active 时间段作为一个"综合窗口"整体核算，得出总体能耗。
    # 综合窗口的好处是不依赖单阶段电量步进，只依赖窗口首尾电量差。
    if active_rows:
        combined_start = _float(active_rows[0], "start_battery_level")
        combined_end = _float(active_rows[-1], "end_battery_level")
        combined_duration_s = sum(_float(row, "duration_s") for row in active_rows)
        combined_iterations = sum(_int(row, "iterations") for row in active_rows)
        # 窗口内总掉电 = 起点电量 − 终点电量（不足 0 则取 0，防止负数）
        combined_raw_delta = max(combined_start - combined_end, 0.0)
        combined_idle_delta = idle_rate * combined_duration_s
        combined_model_delta = max(combined_raw_delta - combined_idle_delta, 0.0)
        combined_energy_j = combined_model_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        combined_raw_energy_j = combined_raw_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        combined_idle_energy_j = combined_idle_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
        combined_raw_power_w = combined_raw_energy_j / combined_duration_s if combined_raw_energy_j is not None and combined_duration_s > 0 else None
        combined_idle_power_w = combined_idle_energy_j / combined_duration_s if combined_idle_energy_j is not None and combined_duration_s > 0 else None
        combined_model_power_w = combined_energy_j / combined_duration_s if combined_energy_j is not None and combined_duration_s > 0 else None
        combined_energy_mj_per_inference = (
            combined_energy_j * 1000.0 / combined_iterations
            if combined_energy_j is not None and combined_iterations > 0
            else None
        )
        out_rows.append(
            {
                "model_variant": "MobileViT-active-window-combined",
                "input_shape": "mixed",
                "runtime": active_rows[0].get("runtime", ""),
                "precision": active_rows[0].get("precision", ""),
                "compute_units": "mixed",
                "compute_unit_scope": "mixed Core ML compute-unit policies",
                "compute_unit_claim_boundary": "Combined row spans multiple Core ML policies; not isolated GPU-only or ANE-only execution.",
                "aggregation_level": "combined_active_window",
                "source_phase_count": len(active_rows),
                "idle_subtraction_source": "global_idle_mean",
                "duration_s": f"{combined_duration_s:.6f}",
                "iterations": combined_iterations,
                "mean_latency_ms": "",
                "start_battery_level": f"{combined_start:.6f}",
                "end_battery_level": f"{combined_end:.6f}",
                "raw_battery_delta_fraction": f"{combined_raw_delta:.8f}",
                "idle_rate_fraction_per_s": f"{idle_rate:.12f}",
                "idle_equivalent_delta_fraction": f"{combined_idle_delta:.8f}",
                "idle_subtracted_delta_fraction": f"{combined_model_delta:.8f}",
                "battery_capacity_wh": "" if capacity_wh is None else f"{capacity_wh:.6f}",
                "raw_avg_power_w_estimate": "" if combined_raw_power_w is None else f"{combined_raw_power_w:.6f}",
                "idle_equivalent_avg_power_w_estimate": "" if combined_idle_power_w is None else f"{combined_idle_power_w:.6f}",
                "idle_subtracted_avg_power_w_estimate": "" if combined_model_power_w is None else f"{combined_model_power_w:.6f}",
                "energy_j_estimate": "" if combined_energy_j is None else f"{combined_energy_j:.6f}",
                "energy_mj_per_inference_estimate": "" if combined_energy_mj_per_inference is None else f"{combined_energy_mj_per_inference:.6f}",
                "status": "combined_active_window_only" if combined_model_delta > 0 else "insufficient_battery_resolution",
                "evidence_label": "iPhone battery-discharge combined active-window estimate with idle subtraction; not per-model calibrated",
                "claim_boundary": "Combined active-window context only. Per-model attribution is not supported by coarse battery-level steps.",
            }
        )

        # 第八步：按"模型 × 计算单元"把综合窗口的总能耗按"时长占比"分摊到各个模型。
        # 这只是一个快速测试用的粗分摊（duration-allocated），不是逐模型实测。
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in active_rows:
            grouped.setdefault((row.get("model_variant", ""), row.get("compute_units", "")), []).append(row)
        for (model_variant, compute_units), model_rows in sorted(grouped.items()):
            model_duration_s = sum(_float(row, "duration_s") for row in model_rows)
            model_iterations = sum(_int(row, "iterations") for row in model_rows)
            # 该模型时长占总时长的比例
            duration_fraction = model_duration_s / combined_duration_s if combined_duration_s > 0 else 0.0
            # 按比例从综合窗口的总量里"切"出一份给该模型
            allocated_delta = combined_model_delta * duration_fraction
            allocated_raw_delta = combined_raw_delta * duration_fraction
            allocated_idle_delta = combined_idle_delta * duration_fraction
            allocated_energy_j = allocated_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
            allocated_raw_energy_j = allocated_raw_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
            allocated_idle_energy_j = allocated_idle_delta * capacity_wh * 3600.0 if capacity_wh is not None else None
            allocated_power_w = allocated_energy_j / model_duration_s if allocated_energy_j is not None and model_duration_s > 0 else None
            allocated_raw_power_w = allocated_raw_energy_j / model_duration_s if allocated_raw_energy_j is not None and model_duration_s > 0 else None
            allocated_idle_power_w = allocated_idle_energy_j / model_duration_s if allocated_idle_energy_j is not None and model_duration_s > 0 else None
            allocated_mj_per_inference = (
                allocated_energy_j * 1000.0 / model_iterations
                if allocated_energy_j is not None and model_iterations > 0
                else None
            )
            mean_latency_values = [
                _float(row, "mean_latency_ms")
                for row in model_rows
                if str(row.get("mean_latency_ms", "")).strip()
            ]
            out_rows.append(
                {
                    "model_variant": f"{model_variant}-duration-allocated",
                    "input_shape": model_rows[0].get("input_shape", ""),
                    "runtime": model_rows[0].get("runtime", ""),
                    "precision": model_rows[0].get("precision", ""),
                    "compute_units": compute_units,
                    "compute_unit_scope": model_rows[0].get("compute_unit_scope", ""),
                    "compute_unit_claim_boundary": model_rows[0].get("compute_unit_claim_boundary", ""),
                    "aggregation_level": "duration_allocated",
                    "source_phase_count": len(model_rows),
                    "idle_subtraction_source": "combined_active_duration_fraction",
                    "duration_s": f"{model_duration_s:.6f}",
                    "iterations": model_iterations,
                    "mean_latency_ms": "" if not mean_latency_values else f"{fmean(mean_latency_values):.6f}",
                    "start_battery_level": "",
                    "end_battery_level": "",
                    "raw_battery_delta_fraction": f"{allocated_raw_delta:.8f}",
                    "idle_rate_fraction_per_s": f"{idle_rate:.12f}",
                    "idle_equivalent_delta_fraction": f"{allocated_idle_delta:.8f}",
                    "idle_subtracted_delta_fraction": f"{allocated_delta:.8f}",
                    "battery_capacity_wh": "" if capacity_wh is None else f"{capacity_wh:.6f}",
                    "raw_avg_power_w_estimate": "" if allocated_raw_power_w is None else f"{allocated_raw_power_w:.6f}",
                    "idle_equivalent_avg_power_w_estimate": "" if allocated_idle_power_w is None else f"{allocated_idle_power_w:.6f}",
                    "idle_subtracted_avg_power_w_estimate": "" if allocated_power_w is None else f"{allocated_power_w:.6f}",
                    "energy_j_estimate": "" if allocated_energy_j is None else f"{allocated_energy_j:.6f}",
                    "energy_mj_per_inference_estimate": "" if allocated_mj_per_inference is None else f"{allocated_mj_per_inference:.6f}",
                    "status": "allocated_from_combined_active_window_by_duration",
                    "evidence_label": "Duration-allocated estimate from combined iPhone battery-discharge window; not per-model measured",
                    "claim_boundary": "Quick testing record only. Do not use as calibrated per-model power/energy evidence.",
                }
            )

    # 第九步：汇总所有质量控制信息到 manifest（元信息清单）
    manifest = {
        "summary_csv": str(summary_csv),
        "samples_csv": "" if samples_csv is None else str(samples_csv),
        "sample_count": sample_count,
        "sample_battery_states": sample_battery_states,
        "unique_sample_battery_levels": unique_sample_battery_levels,
        "battery_transition_count": battery_transition_count,
        "phase_attribution_uncertain": phase_attribution_uncertain,
        "idle_phase_count": len(idle_rows),
        "active_phase_count": len(active_rows),
        "idle_rate_fraction_per_s": idle_rate,
        "battery_capacity_wh": capacity_wh,
        "nominal_capacity_wh": nominal_capacity_wh,
        "battery_health_factor": battery_health_factor,
        "effective_capacity_wh": capacity_wh,
        "capacity_reference_url": capacity_reference_url,
        "capacity_correction": (
            "effective Wh = nominal Wh * Battery Health maximum capacity"
            if nominal_capacity_wh is not None and battery_health_factor is not None
            else ""
        ),
        "min_sample_battery_level": min_sample_battery_level,
        "max_sample_battery_level": max_sample_battery_level,
        "low_battery_threshold": low_battery_threshold,
        "low_battery_observed": low_battery_observed,
        "low_battery_active_phase_count": low_battery_active_phase_count,
        "run_quality_notes": [
            "Battery level is quantized by iOS; individual phase attribution is coarse."
        ]
        + (
            [
                "Low-battery range observed near the end of the run; treat tail phases as higher risk for OS power-management effects."
            ]
            if low_battery_observed
            else []
        ),
        "energy_claim_boundary": "Battery-discharge estimate; calibrated W/J claims require external power meter or a validated device power source.",
        "rows": out_rows,
    }
    return out_rows, manifest


def main() -> None:
    """命令行入口：解析参数、计算有效容量、调用 summarize 并写出 CSV 与 JSON。

    做什么：读取 --summary-csv（及可选的 --samples-csv），把能耗估计结果写入
        --out-dir 下的 iphone_energy_discharge_summary.csv 和
        iphone_energy_discharge_manifest.json。
    参数：全部来自命令行（见 argparse 定义）。
    返回：无。
    """
    parser = argparse.ArgumentParser(description="Import iPhone battery-discharge energy logs from MobileViTCoreMLBench.")
    parser.add_argument("--summary-csv", required=True, type=pathlib.Path)
    parser.add_argument("--samples-csv", type=pathlib.Path)
    parser.add_argument("--capacity-wh", type=float, help="Battery nominal energy in Wh. Omit to keep only battery-fraction evidence.")
    parser.add_argument("--nominal-capacity-wh", type=float, help="New-battery nominal energy in Wh, before Battery Health correction.")
    parser.add_argument("--battery-health-factor", type=float, help="Battery Health maximum capacity as a fraction, e.g. 0.91.")
    parser.add_argument("--capacity-reference-url", default="", help="Reference URL for the capacity/health correction.")
    parser.add_argument("--low-battery-threshold", type=float, default=0.20)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    # 若没直接给容量，但给了"标称容量 × 健康度"，就用两者相乘得到有效容量
    capacity_wh = args.capacity_wh
    if capacity_wh is None and args.nominal_capacity_wh is not None and args.battery_health_factor is not None:
        capacity_wh = args.nominal_capacity_wh * args.battery_health_factor

    # 调用核心汇总函数，得到能耗行和元信息清单
    rows, manifest = summarize(
        args.summary_csv,
        args.samples_csv,
        capacity_wh,
        nominal_capacity_wh=args.nominal_capacity_wh,
        battery_health_factor=args.battery_health_factor,
        capacity_reference_url=args.capacity_reference_url,
        low_battery_threshold=args.low_battery_threshold,
    )
    # 定义输出 CSV 的列顺序（固定顺序保证每次输出列一致）
    fields = [
        "model_variant",
        "input_shape",
        "runtime",
        "precision",
        "compute_units",
        "compute_unit_scope",
        "compute_unit_claim_boundary",
        "aggregation_level",
        "source_phase_count",
        "idle_subtraction_source",
        "duration_s",
        "iterations",
        "mean_latency_ms",
        "start_battery_level",
        "end_battery_level",
        "raw_battery_delta_fraction",
        "idle_rate_fraction_per_s",
        "idle_equivalent_delta_fraction",
        "idle_subtracted_delta_fraction",
        "battery_capacity_wh",
        "raw_avg_power_w_estimate",
        "idle_equivalent_avg_power_w_estimate",
        "idle_subtracted_avg_power_w_estimate",
        "energy_j_estimate",
        "energy_mj_per_inference_estimate",
        "status",
        "evidence_label",
        "claim_boundary",
    ]
    # 写能耗汇总 CSV 与元信息 JSON 两个输出文件
    _write_csv(args.out_dir / "iphone_energy_discharge_summary.csv", rows, fields)
    with (args.out_dir / "iphone_energy_discharge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
