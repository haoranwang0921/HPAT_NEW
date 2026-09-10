"""E-local 固定子集精度实验（run_e_local_fixed_subset.py）。

实验目的：在"固定标注图片子集"（可复现、来源可追溯）上跑 MobileViT，
用 HPAT 映射边界注入非理想性，测出精度/预测变化率，形成 e-local
（P1 档边缘本地推理）的"固定子集鲁棒性"证据线。本脚本是
run_nonideality_accuracy_sweep.py 的"精包装"：强制使用 hpat-mapping
注入边界，并在产出上补充子集来源标签（external_data_label）与
"最坏情况"摘要表。

- 输入：--dataset-root + --subset-file（固定子集），
  --operator-activity-csv（HPAT 映射所需算子活动表）。
- 产出（--output-dir 下）：
  * tables/nonideality_accuracy_fixed_subset.csv（实测明细）；
  * tables/nonideality_accuracy_summary.csv（每效果最坏情况）；
  * 继承的 trials/by_severity/safe_region/failure_thresholds CSV；
  * e_local_fixed_subset_manifest.json；并同步写入仓库 tables/。
  若无子集则写"阻塞"状态表。
- 命令：python run_e_local_fixed_subset.py --output-dir <目录>
  --dataset-root <根> --subset-file <文件> [--config <配置>] ...
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections import defaultdict
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.e_local import E_LOCAL_CLAIM_BOUNDARY
from run_nonideality_accuracy_sweep import (
    BY_SEVERITY_FIELDS,
    FAILURE_THRESHOLD_FIELDS,
    MEASURED_NONIDEALITY_FIELDS,
    SAFE_REGION_FIELDS,
    TRIAL_NONIDEALITY_FIELDS,
    run as run_nonideality,
)


FIXED_SUBSET_SUMMARY_FIELDS = [
    "variant",
    "effect",
    "worst_sweep_variable",
    "worst_sweep_value",
    "sample_count",
    "label_count",
    "clean_top1",
    "worst_perturbed_top1",
    "worst_top1_delta",
    "clean_top5",
    "worst_perturbed_top5",
    "worst_top5_delta",
    "prediction_change_rate",
    "evidence_label",
    "claim_boundary",
]


def _safe_float(value: Any) -> float:
    """把值转成浮点数；空值/None 返回 0.0。"""
    if value in ("", None):
        return 0.0
    return float(value)


def _subset_external_labels(subset_file: pathlib.Path | None) -> list[str]:
    """读取子集 CSV 中的 external_data_label（外部数据来源标签）去重列表。

    参数 subset_file：子集 CSV 路径。
    返回：去重排序后的外部数据来源标签列表；无则返回空列表。
    """
    if not subset_file or not subset_file.exists() or subset_file.suffix.lower() != ".csv":
        return []
    labels = sorted(
        {
            row.get("external_data_label", "").strip()
            for row in read_csv(subset_file)
            if row.get("external_data_label", "").strip()
        }
    )
    return labels


def _maybe_source_ledger(dataset_root: pathlib.Path | None) -> pathlib.Path | None:
    """若数据根目录下有 fixed_subset_source_ledger.csv（来源台账）则返回其路径。

    该台账记录固定子集里每张图的原始出处，用于保证子集可追溯。
    """
    if not dataset_root:
        return None
    candidate = dataset_root / "fixed_subset_source_ledger.csv"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """从实测明细中提取每个 (模型, 效果) 的"最坏情况"行。

    最坏 = |top1_delta| 最大的扫描点，便于论文里快速看到影响上限。
    参数 rows：实测明细行列表。
    返回：摘要行列表。
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["effect"])].append(row)
    out: list[dict[str, Any]] = []
    for (variant, effect), effect_rows in sorted(grouped.items()):
        worst = max(effect_rows, key=lambda row: abs(_safe_float(row.get("top1_delta"))))
        out.append(
            {
                "variant": variant,
                "effect": effect,
                "worst_sweep_variable": worst.get("sweep_variable", ""),
                "worst_sweep_value": worst.get("sweep_value", ""),
                "sample_count": worst.get("sample_count", ""),
                "label_count": worst.get("label_count", ""),
                "clean_top1": worst.get("clean_top1", ""),
                "worst_perturbed_top1": worst.get("perturbed_top1", ""),
                "worst_top1_delta": worst.get("top1_delta", ""),
                "clean_top5": worst.get("clean_top5", ""),
                "worst_perturbed_top5": worst.get("perturbed_top5", ""),
                "worst_top5_delta": worst.get("top5_delta", ""),
                "prediction_change_rate": worst.get("prediction_change_rate", ""),
                "evidence_label": worst.get("evidence_label", ""),
                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            }
        )
    return out


def _append_source_label(path: pathlib.Path, fieldnames: list[str], external_label_note: str) -> None:
    """给已有 CSV 的每一行追加"固定子集来源"标签，并重写该文件。

    参数：
        path：目标 CSV。
        fieldnames：列名列表（重写时保持表头不变）。
        external_label_note：来源说明文本（空则跳过）。
    """
    if not external_label_note or not path.exists():
        return
    rows = read_csv(path)
    for row in rows:
        if "evidence_label" in row:
            row["evidence_label"] = f"{row.get('evidence_label', '')}; fixed subset source={external_label_note}"
        if "claim_boundary" in row:
            row["claim_boundary"] = (
                "Fixed-subset robustness/sensitivity only; not full ImageNet validation, "
                "not silicon validation, and not measured HPAT edge/mobile deployment."
            )
    write_csv(path, rows, fieldnames)


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    dataset_root: pathlib.Path | None,
    subset_file: pathlib.Path | None,
    max_samples: int | None,
    seed: int | None,
    device: str,
    pretrained: bool,
    model_variant: str,
    operator_activity_csv: pathlib.Path | None,
    min_boundary_coverage: float,
    batch_size: int,
    save_logits: str,
    save_prediction_detail: str,
    repeat_seeds: str,
    safe_top1_drop_threshold: float,
    safe_top5_drop_threshold: float,
    safe_prediction_change_threshold: float,
) -> dict[str, pathlib.Path]:
    """固定子集 e-local 实验主流程（包装 run_nonideality 精度扫描）。

    参数：见各参数名。
    返回：产出文件路径字典（含 fixed_subset_csv、summary_csv 及各继承表）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    manifest_path = output_dir / "e_local_fixed_subset_manifest.json"
    manifest = base_manifest("e_local_fixed_subset", "fixed-subset E-local non-ideality accuracy lane")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset_root": relative(dataset_root) if dataset_root else "",
            "subset_file": relative(subset_file) if subset_file else "",
            "subset_file_sha256": sha256_file(subset_file) if subset_file else None,
            "subset_external_data_labels": _subset_external_labels(subset_file),
            "subset_source_ledger": relative(_maybe_source_ledger(dataset_root)) if _maybe_source_ledger(dataset_root) else "",
            "max_samples": max_samples,
            "seed": seed,
            "requested_device": device,
            "pretrained": pretrained,
            "model_variant": model_variant,
            "operator_activity_csv": relative(operator_activity_csv) if operator_activity_csv else "",
            "operator_activity_csv_sha256": sha256_file(operator_activity_csv) if operator_activity_csv else None,
            "min_boundary_coverage": min_boundary_coverage,
            "batch_size": batch_size,
            "save_logits": save_logits,
            "save_prediction_detail": save_prediction_detail,
            "repeat_seeds": repeat_seeds,
            "safe_thresholds": {
                "top1_drop": safe_top1_drop_threshold,
                "top5_drop": safe_top5_drop_threshold,
                "prediction_change_rate": safe_prediction_change_threshold,
            },
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        }
    )
    # 未提供固定子集 → 只写"阻塞"状态表，不跑耗时实验
    if not dataset_root or not subset_file:
        status_csv = tables_dir / "nonideality_accuracy_fixed_subset_status.csv"
        project_status_csv = REPO_ROOT / "tables" / "nonideality_accuracy_fixed_subset_status.csv"
        rows = [
            {
                "status": "blocked",
                "blocked_reason": "No fixed labeled validation subset was supplied.",
                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            }
        ]
        write_csv(status_csv, rows, ["status", "blocked_reason", "claim_boundary"])
        write_csv(project_status_csv, rows, ["status", "blocked_reason", "claim_boundary"])
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "No fixed labeled validation subset was supplied.",
                "outputs": [relative(status_csv), relative(project_status_csv)],
                "promotion_note": "Provide --dataset-root and --subset-file to collect fixed-subset robustness evidence.",
                "caffeinate_used": False,
                "caffeinate_reason": "Not used; blocked before a long dataset run.",
            }
        )
        write_json(manifest_path, manifest)
        return {"status_csv": status_csv, "project_status_csv": project_status_csv, "manifest": manifest_path}

    # 核心工作委托给 run_nonideality_accuracy_sweep.run（强制 hpat-mapping 边界）
    outputs = run_nonideality(
        output_dir=output_dir,
        config_path=config_path,
        dataset_root=dataset_root,
        subset_file=subset_file,
        max_samples=max_samples,
        seed=seed,
        device=device,
        pretrained=pretrained,
        model_variant=model_variant,
        operator_activity_csv=operator_activity_csv,
        injection_boundary="hpat-mapping",
        min_boundary_coverage=min_boundary_coverage,
        batch_size=batch_size,
        save_logits=save_logits,
        save_prediction_detail=save_prediction_detail,
        repeat_seeds=repeat_seeds,
        safe_top1_drop_threshold=safe_top1_drop_threshold,
        safe_top5_drop_threshold=safe_top5_drop_threshold,
        safe_prediction_change_threshold=safe_prediction_change_threshold,
    )
    measured = outputs["measured_csv"]
    rows = read_csv(measured)
    # 给结果补充"外部数据来源"标签，保证外部子集的结论被明确标注
    external_labels = _subset_external_labels(subset_file)
    external_label_note = "; ".join(external_labels)
    if external_label_note:
        for row in rows:
            row["evidence_label"] = (
                f"{row.get('evidence_label', '')}; fixed subset source={external_label_note}"
            )
        # 对其余继承表也追加来源标签（本次运行目录与仓库公共目录各一份）
        for key, fieldnames in [
            ("trials_csv", TRIAL_NONIDEALITY_FIELDS),
            ("by_severity_csv", BY_SEVERITY_FIELDS),
            ("safe_region_csv", SAFE_REGION_FIELDS),
            ("failure_thresholds_csv", FAILURE_THRESHOLD_FIELDS),
        ]:
            _append_source_label(outputs[key], fieldnames, external_label_note)
            project_path = REPO_ROOT / "tables" / outputs[key].name
            _append_source_label(project_path, fieldnames, external_label_note)
    # 写出固定子集专用表：实测明细 + 最坏情况摘要
    fixed_csv = tables_dir / "nonideality_accuracy_fixed_subset.csv"
    project_fixed_csv = REPO_ROOT / "tables" / "nonideality_accuracy_fixed_subset.csv"
    summary_csv = tables_dir / "nonideality_accuracy_summary.csv"
    project_summary_csv = REPO_ROOT / "tables" / "nonideality_accuracy_summary.csv"
    summary_rows = _summary_rows(rows)
    write_csv(fixed_csv, rows, MEASURED_NONIDEALITY_FIELDS)
    write_csv(project_fixed_csv, rows, MEASURED_NONIDEALITY_FIELDS)
    write_csv(summary_csv, summary_rows, FIXED_SUBSET_SUMMARY_FIELDS)
    write_csv(project_summary_csv, summary_rows, FIXED_SUBSET_SUMMARY_FIELDS)
    manifest.update(
        {
            "status": "ready_with_limitations",
            "outputs": [
                relative(fixed_csv),
                relative(project_fixed_csv),
                relative(summary_csv),
                relative(project_summary_csv),
                relative(outputs["trials_csv"]),
                relative(REPO_ROOT / "tables" / outputs["trials_csv"].name),
                relative(outputs["by_severity_csv"]),
                relative(REPO_ROOT / "tables" / outputs["by_severity_csv"].name),
                relative(outputs["safe_region_csv"]),
                relative(REPO_ROOT / "tables" / outputs["safe_region_csv"].name),
                relative(outputs["failure_thresholds_csv"]),
                relative(REPO_ROOT / "tables" / outputs["failure_thresholds_csv"].name),
                relative(outputs["manifest"]),
            ],
            "row_count": len(rows),
            "summary_row_count": len(summary_rows),
            "trials_csv": relative(outputs["trials_csv"]),
            "by_severity_csv": relative(outputs["by_severity_csv"]),
            "safe_region_csv": relative(outputs["safe_region_csv"]),
            "failure_thresholds_csv": relative(outputs["failure_thresholds_csv"]),
            "subset_external_data_labels": external_labels,
            "subset_source_ledger": relative(_maybe_source_ledger(dataset_root)) if _maybe_source_ledger(dataset_root) else "",
            # 免责说明：只支持记录范围内固定子集/hook 注入的非理想性论断
            "promotion_note": (
                "Supports fixed-subset, hook-injected non-ideality claims only within the recorded subset, "
                "model/checkpoint, device, and HPAT mapping boundary. External public subset rows must remain "
                "labeled as external and do not support full ImageNet robustness claims."
            ),
            "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED") == "1",
            "caffeinate_reason": (
                "Caller set HPAT_CAFFEINATE_USED=1; command was expected to be wrapped in caffeinate."
                if os.environ.get("HPAT_CAFFEINATE_USED") == "1"
                else "This wrapper does not caffeinate itself; wrap real dataset runs with caffeinate -dimsu."
            ),
        }
    )
    source_ledger = _maybe_source_ledger(dataset_root)
    if source_ledger:
        manifest["outputs"].append(relative(source_ledger))
    write_json(manifest_path, manifest)
    return {
        "fixed_subset_csv": fixed_csv,
        "project_fixed_subset_csv": project_fixed_csv,
        "summary_csv": summary_csv,
        "project_summary_csv": project_summary_csv,
        "trials_csv": outputs["trials_csv"],
        "by_severity_csv": outputs["by_severity_csv"],
        "safe_region_csv": outputs["safe_region_csv"],
        "failure_thresholds_csv": outputs["failure_thresholds_csv"],
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
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
    parser.add_argument("--operator-activity-csv", default=str(REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"))
    parser.add_argument("--min-boundary-coverage", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-logits", default="full", choices=["summary", "none", "full"])
    parser.add_argument("--save-prediction-detail", default="basic", choices=["basic", "topk-margin"])
    parser.add_argument("--repeat-seeds", default="")
    parser.add_argument("--safe-top1-drop-threshold", type=float, default=5.0)
    parser.add_argument("--safe-top5-drop-threshold", type=float, default=5.0)
    parser.add_argument("--safe-prediction-change-threshold", type=float, default=10.0)
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
        args.min_boundary_coverage,
        args.batch_size,
        args.save_logits,
        args.save_prediction_detail,
        args.repeat_seeds,
        args.safe_top1_drop_threshold,
        args.safe_top5_drop_threshold,
        args.safe_prediction_change_threshold,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
