"""边缘设备（Edge）基线（baseline，即对比用的基准数据）导入脚本。

做什么：
    把外部采集的手机等边缘设备实测数据（延迟 latency 与功耗 power 的 CSV 日志）
    导入到本实验库，经过格式校验、字段规范化（normalize，统一成标准列名）
    后，生成汇总统计表（均值/中位数/分位数等），供后续校准边缘能耗模型使用。

数据从哪来（输入）：
    --latency-csv   边缘设备延迟采样日志（CSV），必填；
    --power-csv     边缘设备功耗采样日志（CSV），可选；
    --device-config 设备元信息配置文件（JSON 或 key: value 文本），必填（配合延迟 CSV 时）。

输出到哪（输出）：
    <output-dir>/raw/edge_latency_samples.csv         规范化后的延迟原始样本；
    <output-dir>/raw/edge_power_samples.csv           规范化后的功耗原始样本（若有）；
    <output-dir>/tables/mobilevit_edge_baseline_measured.csv  汇总统计表；
    <output-dir>/edge_baseline_import_manifest.json   本次导入的清单（manifest，记录来源、校验状态、SHA256）；
    同时把汇总表复制到仓库根目录 tables/ 下。

怎么运行：
    python run_edge_baseline_import.py --output-dir <输出目录> --latency-csv <延迟CSV> --device-config <设备配置>

设计要点：
    1) 严格字段校验：缺列、数值非法都会把本次导入标记为 blocked（受阻）并写明原因，
       保证进入实验库的每一行数据都是"来源明确、作者本人实测"的可信数据。
    2) 每个输出数据行都带有 evidence_label（证据标签），说明它是 MobileViT 边缘基线
       上下文、并非 HPAT 部署数据，防止后续被误当成 HPAT 光子芯片本身的实验证据。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.mobilevit_loader import latency_stats


# 设备配置文件中必须提供的字段列表（缺失任一字段，本次导入即被判定为 blocked）。
# 这些元信息用于说明"数据是在什么设备、什么条件下测出来的"，是溯源（provenance，
# 即数据来源可查证）的基础。
DEVICE_REQUIRED_FIELDS = [
    "device_id",
    "device_label",
    "device_type",
    "os",
    "runtime",
    "precision",
    "power_method",
    "thermal_mode",
    "power_mode",
]

# 延迟 CSV 中必须存在的列名（模型变体、输入尺寸、批大小、运行框架、精度、延迟毫秒数）。
EDGE_LATENCY_REQUIRED_FIELDS = [
    "model_variant",
    "input_shape",
    "batch_size",
    "runtime",
    "precision",
    "latency_ms",
]

# 规范化后的延迟原始样本在输出 CSV 中的完整列顺序（比输入多出设备元信息与证据标签）。
EDGE_LATENCY_RAW_FIELDS = [
    "model_variant",
    "input_shape",
    "batch_size",
    "runtime",
    "precision",
    "compute_units",
    "compute_unit_scope",
    "compute_unit_claim_boundary",
    "latency_ms",
    "sample_index",
    "warmup",
    "measured_iterations",
    "device_id",
    "device_label",
    "power_method",
    "thermal_mode",
    "power_mode",
    "evidence_label",
]

# 规范化后的功耗原始样本在输出 CSV 中的完整列顺序。
EDGE_POWER_RAW_FIELDS = [
    "timestamp_s",
    "power_w",
    "model_variant",
    "device_id",
    "measurement_window",
    "idle_subtracted",
    "sampling_rate_hz",
    "evidence_label",
]

# 论文中必须覆盖的 MobileViT 模型变体（即版本）。若导入数据中缺少其中任一，
# 说明覆盖不完整，基线声明（claim）不能被支持。
REQUIRED_MODEL_VARIANTS = ["MobileViT-XXS", "MobileViT-XS", "MobileViT-S"]
# 功耗 CSV 中必须存在的列名（时间戳、功率瓦数、测量窗口、是否扣除空闲功耗、采样率）。
EDGE_POWER_REQUIRED_FIELDS = ["timestamp_s", "power_w", "measurement_window", "idle_subtracted", "sampling_rate_hz"]

# 最终汇总统计表（每行对应"一个模型变体 + 一个输入配置"的组合）的完整列顺序。
EDGE_SUMMARY_FIELDS = [
    "model_variant",
    "input_shape",
    "batch_size",
    "runtime",
    "precision",
    "compute_units",
    "compute_unit_scope",
    "compute_unit_claim_boundary",
    "device_id",
    "device_label",
    "status",
    "blocked_reason",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p05",
    "latency_ms_p95",
    "latency_ms_min",
    "latency_ms_max",
    "sample_count",
    "warmup",
    "measured_iterations",
    "power_collection_status",
    "power_method",
    "thermal_mode",
    "power_mode",
    "coverage_status",
    "claim_eligible",
    "missing_variants",
    "power_claim_eligible",
    "evidence_label",
]


def _load_device_config(path: pathlib.Path) -> dict[str, Any]:
    """读取设备元信息配置文件，兼容 JSON 与简单的 "key: value" 文本两种格式。

    用途：
        设备配置文件描述"这次实测用的手机/边缘设备是什么、测量条件是什么"，
        例如设备 ID、型号、操作系统、运行框架、精度、功耗测量方式、热模式和功耗模式。
        这些信息最终会拼进每一行导入数据，作为溯源（provenance）记录。

    参数：
        path: 配置文件的路径。

    返回：
        dict 形式的设备元信息（字段名 -> 字段值）。读取失败或格式错误时，
        JSON 的解析异常会原样抛出，交由调用方（run）记录到清单中。
    """
    # JSON 配置：优先取出 "device" 键下的内容，没有则把整个 JSON 当作设备信息。
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload.get("device"), dict):
            return dict(payload["device"])
        return dict(payload)

    # 文本配置：逐行解析，形如 "device_id: iphone-12"；支持 "#" 注释和引号。
    config: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()  # 去掉 # 号后的注释部分
        if not line or ":" not in line:
            continue  # 空行或没有冒号的行直接跳过
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip().strip('"').strip("'")  # 去掉首尾空白和引号
    return config


def _validate_device_config(config: dict[str, Any]) -> list[str]:
    """校验设备配置是否完整。

    参数：
        config: _load_device_config 返回的设备元信息字典。

    返回：
        缺失字段名列表；若为空列表，表示配置完整、校验通过。
    """
    # 逐个检查必填字段：取到的是空串/空值也视为缺失。
    return [field for field in DEVICE_REQUIRED_FIELDS if not str(config.get(field, "")).strip()]


def _normalize_latency_rows(rows: list[dict[str, Any]], device: dict[str, Any]) -> list[dict[str, Any]]:
    """校验并规范化延迟（latency，即单次推理耗时）采样数据。

    做什么：
        1) 校验 CSV 非空、必填列齐全；
        2) 逐行检查 latency_ms 必须是正数；
        3) 把每行补全设备元信息和证据标签，得到统一列结构的规范化样本。

    为什么：
        外部采集的日志格式可能五花八门，规范化后才能保证后续统计、
        汇总和论文基线声明建立在可靠、可溯源的数据上。

    参数：
        rows:   read_csv 读出的原始延迟行（每行一个 dict）。
        device: 设备元信息，会合并进每一行输出。

    返回：
        规范化后的延迟样本列表。校验失败时抛出 ValueError（并附行号），
        由 run 捕获后记入清单并中止本次导入。
    """
    if not rows:
        raise ValueError("edge latency CSV contains no data rows")
    missing = [field for field in EDGE_LATENCY_REQUIRED_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"edge latency CSV missing required columns: {', '.join(missing)}")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        # 延迟必须是能转成 float 的数值；空值或文本都会导致 TypeError/ValueError。
        try:
            latency_ms = float(row.get("latency_ms", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"edge latency row {index + 1}: latency_ms must be numeric") from exc
        if latency_ms <= 0:
            raise ValueError(f"edge latency row {index + 1}: latency_ms must be positive")
        normalized.append(
            {
                "model_variant": str(row["model_variant"]).strip(),
                "input_shape": str(row["input_shape"]).strip(),
                "batch_size": str(row["batch_size"]).strip(),
                "runtime": str(row["runtime"]).strip(),
                "precision": str(row["precision"]).strip(),
                # 计算单元（compute unit）相关字段是可选元信息，缺省填空串。
                "compute_units": str(row.get("compute_units", "")).strip(),
                "compute_unit_scope": str(row.get("compute_unit_scope", "")).strip(),
                "compute_unit_claim_boundary": str(row.get("compute_unit_claim_boundary", "")).strip(),
                # 统一保留 6 位小数，避免浮点格式不统一。
                "latency_ms": f"{latency_ms:.6f}",
                "sample_index": row.get("sample_index", index),  # 无编号时用行号兜底
                "warmup": row.get("warmup", ""),
                "measured_iterations": row.get("measured_iterations", ""),
                "device_id": str(device.get("device_id", "")).strip(),
                "device_label": str(device.get("device_label", "")).strip(),
                "power_method": str(device.get("power_method", "")).strip(),
                "thermal_mode": str(device.get("thermal_mode", "")).strip(),
                "power_mode": str(device.get("power_mode", "")).strip(),
                # 证据标签：明确这些数据是 MobileViT 边缘基线上下文，不是 HPAT 部署数据，
                # 防止后续分析把手机实测结果误当成 HPAT 光子芯片的证据。
                "evidence_label": "author-measured MobileViT edge baseline context; not HPAT deployment",
            }
        )
    return normalized


def _normalize_power_rows(rows: list[dict[str, Any]], device: dict[str, Any]) -> list[dict[str, Any]]:
    """校验并规范化功耗（power，单位瓦特）采样数据。

    做什么：
        1) 功耗日志可能缺失（rows 为空），此时返回空列表表示"本次只导入延迟"；
        2) 校验必填列齐全、必填值非空；
        3) 逐行检查 timestamp_s / power_w / sampling_rate_hz 必须是数值，
           且 power_w >= 0、sampling_rate_hz > 0（采样率不可能是零或负数）；
        4) 补全设备信息与证据标签。

    参数：
        rows:   read_csv 读出的原始功耗行（每行一个 dict）。
        device: 设备元信息。

    返回：
        规范化后的功耗样本列表；校验失败时抛出 ValueError 并由 run 记入清单。
    """
    if not rows:
        return []
    missing = [field for field in EDGE_POWER_REQUIRED_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"edge power CSV missing required columns: {', '.join(missing)}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        # 必填字段的值不能是空串，缺值说明该行记录不完整。
        missing_values = [field for field in EDGE_POWER_REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing_values:
            raise ValueError(f"edge power row {index + 1}: missing required values: {', '.join(missing_values)}")
        try:
            timestamp_s = float(row.get("timestamp_s", ""))
            power_w = float(row.get("power_w", ""))
            sampling_rate_hz = float(row.get("sampling_rate_hz", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"edge power row {index + 1}: timestamp_s, power_w, and sampling_rate_hz must be numeric") from exc
        if power_w < 0 or sampling_rate_hz <= 0:
            raise ValueError(f"edge power row {index + 1}: power_w must be nonnegative and sampling_rate_hz must be positive")
        normalized.append(
            {
                "timestamp_s": f"{timestamp_s:.6f}",
                "power_w": f"{power_w:.6f}",
                "model_variant": row.get("model_variant", ""),
                "device_id": str(device.get("device_id", "")).strip(),
                "measurement_window": row.get("measurement_window", ""),
                "idle_subtracted": row.get("idle_subtracted", ""),
                # 采样率：行内没给就退回设备配置里的 power_sampling_rate_hz，再没有则填空串。
                "sampling_rate_hz": row.get("sampling_rate_hz", device.get("power_sampling_rate_hz", "")),
                "evidence_label": "raw edge power sample for MobileViT baseline context; not HPAT deployment",
            }
        )
    return normalized


def _coverage(raw_rows: list[dict[str, Any]], power_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """评估本次导入数据的覆盖度（coverage），判断能否支撑论文中的基线声明。

    做什么（分步骤）：
        1. 统计延迟样本里出现过的模型变体，与论文要求的 3 个 MobileViT 变体比对，
           找出缺了哪些（missing_variants）；
        2. 找出那些缺少 warmup（预热轮数）或 measured_iterations（实测轮数）元信息的
           变体（missing_timing_meta），这类样本无法说明测量过程是否稳定；
        3. 判断功耗数据能否支撑功耗声明（power_claim_eligible）：要求功耗样本非空，
           且每条都有测量窗口、是否扣除空闲功耗、采样率三项信息；
        4. 综合给出状态：无缺失即 "ready"（就绪，可支撑基线声明），有缺失即 "partial"（部分）。

    参数：
        raw_rows:   规范化后的延迟样本列表。
        power_rows: 规范化后的功耗样本列表（可能为空）。

    返回：
        包含 coverage_status / claim_eligible / missing_variants /
        missing_timing_metadata_variants / power_claim_eligible 的字典。
    """
    variants = sorted({row["model_variant"] for row in raw_rows})  # 延迟样本里实际出现的变体
    missing_variants = [variant for variant in REQUIRED_MODEL_VARIANTS if variant not in variants]
    # 找出缺少计时元信息的变体（warmup 或 measured_iterations 为空）。
    missing_timing_meta = [
        row["model_variant"]
        for row in raw_rows
        if not str(row.get("warmup", "")).strip() or not str(row.get("measured_iterations", "")).strip()
    ]
    # 功耗声明需满足：有功功耗样本，且所有样本的测量条件（窗口/扣空闲/采样率）齐全。
    power_claim_eligible = bool(power_rows) and all(
        str(row.get("measurement_window", "")).strip()
        and str(row.get("idle_subtracted", "")).strip()
        and str(row.get("sampling_rate_hz", "")).strip()
        for row in power_rows
    )
    status = "ready"
    if missing_variants or missing_timing_meta:
        status = "partial"
    return {
        "coverage_status": status,
        "claim_eligible": status == "ready",  # 完整无缺，才能支撑基线声明
        "missing_variants": missing_variants,
        "missing_timing_metadata_variants": sorted(set(missing_timing_meta)),
        "power_claim_eligible": power_claim_eligible,
    }


def _summary_rows(raw_rows: list[dict[str, Any]], power_status: str, coverage: dict[str, Any]) -> list[dict[str, Any]]:
    """把规范化后的延迟样本聚合成"每配置一行"的汇总统计表。

    做什么：
        1. 按 (模型变体, 输入形状, 批大小, 运行框架, 精度, 计算单元三件套) 分组
           ——同一配置下的多次重复采样归为一组；
        2. 对每组的延迟样本计算 mean（均值）、median（中位数）、p05/p95（5%/95% 分位数）、
           min/max（最小/最大值）等统计量；
        3. 合并覆盖度与功耗状态，形成论文可直接引用的汇总行。

    参数：
        raw_rows:    规范化后的延迟样本列表。
        power_status: 功耗采集状态（"power-samples" 有功功耗样本 / "latency-only" 仅延迟）。
        coverage:    _coverage 的返回值（覆盖度信息）。

    返回：
        汇总行列表，每行一个配置，字段见 EDGE_SUMMARY_FIELDS。
    """
    # 分组键：把"同一测量配置"的样本归到一起（8 元组）。
    buckets: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (
            row["model_variant"],
            row["input_shape"],
            row["batch_size"],
            row["runtime"],
            row["precision"],
            row.get("compute_units", ""),
            row.get("compute_unit_scope", ""),
            row.get("compute_unit_claim_boundary", ""),
        )
        buckets.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (model, shape, batch, runtime, precision, compute_units, compute_scope, compute_boundary), rows in sorted(buckets.items()):
        samples = [float(row["latency_ms"]) for row in rows]  # 该组全部延迟样本
        stats = latency_stats(samples)  # 计算统计量（来自 hpat_eval.mobilevit_loader）
        first = rows[0]  # 设备等元信息取组内第一行即可（同组应一致）
        summaries.append(
            {
                "model_variant": model,
                "input_shape": shape,
                "batch_size": batch,
                "runtime": runtime,
                "precision": precision,
                "compute_units": compute_units,
                "compute_unit_scope": compute_scope,
                "compute_unit_claim_boundary": compute_boundary,
                "device_id": first["device_id"],
                "device_label": first["device_label"],
                "status": "ok",
                "blocked_reason": "",
                "latency_ms_mean": f"{stats['mean']:.6f}",
                "latency_ms_median": f"{stats['median']:.6f}",
                "latency_ms_p05": f"{stats['p05']:.6f}",
                "latency_ms_p95": f"{stats['p95']:.6f}",
                "latency_ms_min": f"{stats['min']:.6f}",
                "latency_ms_max": f"{stats['max']:.6f}",
                "sample_count": len(samples),
                "warmup": first.get("warmup", ""),
                "measured_iterations": first.get("measured_iterations", ""),
                "power_collection_status": power_status,
                "power_method": first["power_method"],
                "thermal_mode": first["thermal_mode"],
                "power_mode": first["power_mode"],
                "coverage_status": coverage["coverage_status"],
                # 布尔值转成 "yes"/"no"，方便人读和表格展示。
                "claim_eligible": "yes" if coverage["claim_eligible"] else "no",
                "missing_variants": ";".join(coverage["missing_variants"]),
                "power_claim_eligible": "yes" if coverage["power_claim_eligible"] else "no",
                "evidence_label": "author-measured MobileViT edge baseline context; not HPAT deployment",
            }
        )
    return summaries


def run(
    output_dir: pathlib.Path,
    latency_csv: pathlib.Path | None,
    power_csv: pathlib.Path | None,
    device_config: pathlib.Path | None,
) -> dict[str, pathlib.Path]:
    """边缘基线导入的主流程：校验输入 -> 规范化 -> 汇总统计 -> 写出产物与清单。

    主流程（分步骤）：
        1. 创建输出目录结构（raw/、tables/），初始化清单（manifest）；
        2. 没有延迟 CSV：本次导入被标记为 blocked（受阻），写出空表并结束；
        3. 有延迟 CSV 但缺设备配置：同样 blocked 并抛出 ValueError；
        4. 读取设备配置并校验必填字段，缺失则 blocked；
        5. 规范化延迟/功耗行，任何校验错误都会被记入清单后重新抛出；
        6. 计算覆盖度、聚合汇总统计，写出原始样本表、汇总表（含项目目录一份）；
        7. 更新清单（状态、采样数、覆盖度、输出文件列表）并序列化为 JSON。

    参数：
        output_dir:    输出根目录。
        latency_csv:   延迟采样 CSV 路径；为 None 表示本次仅初始化空结果。
        power_csv:     功耗采样 CSV 路径；为 None 表示无功耗数据。
        device_config: 设备元信息配置文件路径；有延迟 CSV 时必须提供。

    返回：
        字典，键包括 raw_latency_csv / csv / project_csv / manifest，
        值为本次实际写出的文件路径。
    """
    ensure_dir(output_dir)
    raw_dir = ensure_dir(output_dir / "raw")        # 原始规范化样本目录
    tables_dir = ensure_dir(output_dir / "tables")  # 汇总表目录
    manifest_path = output_dir / "edge_baseline_import_manifest.json"
    out_csv = tables_dir / "mobilevit_edge_baseline_measured.csv"
    project_csv = REPO_ROOT / "tables" / "mobilevit_edge_baseline_measured.csv"  # 同步到仓库 tables/
    manifest = base_manifest("edge_baseline_import", "blocked or author-measured MobileViT edge baseline context")
    manifest.update(
        {
            # 记录输入文件相对路径及其 SHA256 校验和，保证结果可溯源、可复现。
            "latency_csv": relative(latency_csv) if latency_csv else "",
            "latency_csv_sha256": sha256_file(latency_csv) if latency_csv else None,
            "power_csv": relative(power_csv) if power_csv else "",
            "power_csv_sha256": sha256_file(power_csv) if power_csv else None,
            "device_config": relative(device_config) if device_config else "",
            "device_config_sha256": sha256_file(device_config) if device_config else None,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; importing externally collected edge logs is a short parsing step.",
        }
    )

    # 分支 1：完全没有延迟 CSV，只能写出一份"无数据"的空表，声明保持关闭。
    if not latency_csv:
        write_csv(out_csv, [], EDGE_SUMMARY_FIELDS)
        write_csv(project_csv, [], EDGE_SUMMARY_FIELDS)
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "No external edge latency CSV was provided.",
                "outputs": [relative(out_csv), relative(project_csv)],
                "promotion_note": "No author-measured edge baseline claim is enabled until raw edge/mobile samples are imported.",
            }
        )
        write_json(manifest_path, manifest)
        return {"csv": out_csv, "project_csv": project_csv, "manifest": manifest_path}

    # 分支 2：有延迟数据但缺设备元信息，无法说明测量环境，同样 blocked。
    if not device_config:
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "Edge latency CSV was provided but device metadata config is missing.",
                "promotion_note": "Device, runtime, precision, thermal, and power-mode metadata are required.",
            }
        )
        write_json(manifest_path, manifest)
        raise ValueError(manifest["blocked_reason"])

    device = _load_device_config(device_config)
    missing_device = _validate_device_config(device)
    if missing_device:
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "Device config missing required fields: " + ", ".join(missing_device),
                "promotion_note": "Complete edge device metadata before importing measured edge rows.",
            }
        )
        write_json(manifest_path, manifest)
        raise ValueError(manifest["blocked_reason"])

    # 分支 3：正式导入。规范化过程中任何 ValueError 都会写入清单并中止。
    try:
        latency_rows = _normalize_latency_rows(read_csv(latency_csv), device)
        power_rows = _normalize_power_rows(read_csv(power_csv), device) if power_csv else []
    except ValueError as exc:
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": str(exc),
                "promotion_note": "Fix edge latency/power schema and required metadata before importing measured edge rows.",
            }
        )
        write_json(manifest_path, manifest)
        raise
    # 功耗采集状态：有功功耗样本记 "power-samples"，否则 "latency-only"（仅延迟）。
    power_status = "power-samples" if power_rows else "latency-only"
    coverage = _coverage(latency_rows, power_rows)
    summary_rows = _summary_rows(latency_rows, power_status, coverage)
    raw_latency_out = raw_dir / "edge_latency_samples.csv"
    raw_power_out = raw_dir / "edge_power_samples.csv"
    write_csv(raw_latency_out, latency_rows, EDGE_LATENCY_RAW_FIELDS)
    if power_rows:
        write_csv(raw_power_out, power_rows, EDGE_POWER_RAW_FIELDS)
    write_csv(out_csv, summary_rows, EDGE_SUMMARY_FIELDS)
    write_csv(project_csv, summary_rows, EDGE_SUMMARY_FIELDS)  # 同步到仓库 tables/
    output_list = [relative(raw_latency_out), relative(out_csv), relative(project_csv)]
    if power_rows:
        output_list.insert(1, relative(raw_power_out))
    manifest.update(
        {
            # 覆盖完整则 "ok"，有缺失则 "partial"（只能部分支撑声明）。
            "status": "ok" if coverage["claim_eligible"] else "partial",
            "device": device,
            "latency_sample_count": len(latency_rows),
            "power_sample_count": len(power_rows),
            "power_collection_status": power_status,
            "coverage": coverage,
            "outputs": output_list,
            # 说明本次导入能支撑到什么程度（只支撑延迟基线，还是连功耗也支撑）。
            "promotion_note": (
                "Enables author-measured MobileViT edge latency baseline context only; no power/energy claim is enabled without valid synchronized power samples."
                if coverage["claim_eligible"]
                else "Edge import is partial and cannot support measured edge baseline claims until all required variants and timing metadata are present."
            ),
        }
    )
    write_json(manifest_path, manifest)
    return {"raw_latency_csv": raw_latency_out, "csv": out_csv, "project_csv": project_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run 主流程。

    支持的参数：
        --output-dir    输出目录（必填）；
        --latency-csv   延迟采样 CSV（可选，缺省则生成空结果）；
        --power-csv     功耗采样 CSV（可选）；
        --device-config 设备元信息配置文件（可选，但配合延迟 CSV 时实际必须）；
        --config        为兼容 run_all 而保留的占位参数，本脚本不使用。

    运行结束后把输出文件路径以 JSON 形式打印到标准输出。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latency-csv", default="")
    parser.add_argument("--power-csv", default="")
    parser.add_argument("--device-config", default="")
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.latency_csv) if args.latency_csv else None,
        pathlib.Path(args.power_csv) if args.power_csv else None,
        pathlib.Path(args.device_config) if args.device_config else None,
    )
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))  # 相对路径打印，便于其他脚本消费


if __name__ == "__main__":
    main()  # 只有被直接执行时才进入命令行入口
