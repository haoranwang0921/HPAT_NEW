"""E-local 注入边界消融（run_e_local_boundary_ablation.py）。

实验目的：对比两种"非理想性注入边界"对精度结果的影响：
  1) hpat-mapping：只在 HPAT 真实映射层注入（可主张的边界）；
  2) all-linear-smoke：在所有 Linear 层注入（纯工程诊断边界）。
用同一固定子集分别跑一遍，输出"对比摘要表"和"差值表"（all-linear
减去 hpat-mapping 的精度降幅），用来说明"全 Linear 注入会高估多少
非理想性影响"。

- 输入：固定子集（--dataset-root/--subset-file）、可选的
  --hpat-run-dir 指向上一次 hpat-mapping 运行目录（默认自动找最新）。
- 产出（--output-dir 下）：tables/nonideality_boundary_ablation_summary.csv、
  tables/nonideality_boundary_ablation_delta.csv、
  e_local_boundary_ablation_manifest.json；并同步写入仓库 tables/。
- 命令：python run_e_local_boundary_ablation.py --output-dir <目录>
  [--hpat-run-dir <目录>] [--dataset-root <根>] [--subset-file <文件>] ...
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_csv, write_json
from run_nonideality_accuracy_sweep import BY_SEVERITY_FIELDS, DATASET_CLAIM_BOUNDARY, run as run_nonideality


SUMMARY_FIELDS = [
    "boundary",
    "boundary_claim_use",
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "trial_count",
    "sample_count",
    "label_count",
    "clean_top1_mean",
    "perturbed_top1_mean",
    "top1_drop_mean",
    "clean_top5_mean",
    "perturbed_top5_mean",
    "top5_drop_mean",
    "prediction_change_rate_mean",
    "mean_relative_error_mean",
    "device",
    "evidence_label",
    "claim_boundary",
]

DELTA_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "all_linear_minus_hpat_top1_drop",
    "all_linear_minus_hpat_top5_drop",
    "all_linear_minus_hpat_prediction_change_rate",
    "hpat_top1_drop",
    "all_linear_top1_drop",
    "evidence_label",
    "claim_boundary",
]


def _float(value: Any, default: float = 0.0) -> float:
    """把值转成浮点数；空值/None 返回默认值。"""
    if value in ("", None):
        return default
    return float(value)


def _fmt(value: float) -> str:
    """格式化浮点数为 4 位小数。"""
    return f"{value:.4f}"


def _latest_hpat_run_dir() -> pathlib.Path:
    """在 results 目录里找最新一个"含 hpat-mapping 结果"的 e-local 运行目录。

    判定条件：tables/nonideality_accuracy_by_severity.csv 与
    nonideality_injection_boundary_manifest.json 都存在。
    找不到则报错（提示先跑 run_e_local_fixed_subset）。
    """
    candidates = sorted(
        [
            path
            for path in (REPO_ROOT / "experiments" / "results").glob("run_*")
            if (path / "tables" / "nonideality_accuracy_by_severity.csv").exists()
            and (path / "nonideality_injection_boundary_manifest.json").exists()
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError("No prior HPAT-mapping E-local run was found.")
    return candidates[-1]


def _read_by_severity(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 by_severity CSV 并校验非空；否则报错。"""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def _summary_rows(boundary: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """把 by_severity 行改写成"带边界标注"的摘要行。

    参数：
        boundary：注入边界名（hpat-mapping / all-linear-smoke）。
        rows：按严重度汇总的行。
    返回：摘要行列表（带 boundary 与是否可主张标记）。
    """
    if boundary == "hpat-mapping":
        claim_use = "claim_eligible_fixed_subset_hpat_mapping_with_limitations"
        label_suffix = "; HPAT mapping boundary"
    else:
        claim_use = "diagnostic_only_not_hpat_claim"
        label_suffix = "; all-linear diagnostic boundary only"
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "boundary": boundary,
                "boundary_claim_use": claim_use,
                "variant": row.get("variant", ""),
                "effect": row.get("effect", ""),
                "sweep_variable": row.get("sweep_variable", ""),
                "sweep_value": row.get("sweep_value", ""),
                "trial_count": row.get("trial_count", ""),
                "sample_count": row.get("sample_count", ""),
                "label_count": row.get("label_count", ""),
                "clean_top1_mean": row.get("clean_top1_mean", ""),
                "perturbed_top1_mean": row.get("perturbed_top1_mean", ""),
                "top1_drop_mean": row.get("top1_drop_mean", ""),
                "clean_top5_mean": row.get("clean_top5_mean", ""),
                "perturbed_top5_mean": row.get("perturbed_top5_mean", ""),
                "top5_drop_mean": row.get("top5_drop_mean", ""),
                "prediction_change_rate_mean": row.get("prediction_change_rate_mean", ""),
                "mean_relative_error_mean": row.get("mean_relative_error_mean", ""),
                "device": row.get("device", ""),
                "evidence_label": f"{row.get('evidence_label', '')}{label_suffix}",
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def _delta_rows(hpat_rows: list[dict[str, str]], all_linear_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """计算两种边界的差值：all-linear 的指标减去 hpat-mapping 的指标。

    只对两边都存在的 (variant, effect, variable, value) 组合计算。
    差值 > 0 表示"全 Linear 注入高估了影响"。
    """
    hpat = {
        (row["variant"], row["effect"], row["sweep_variable"], row["sweep_value"]): row
        for row in hpat_rows
    }
    out: list[dict[str, Any]] = []
    for row in all_linear_rows:
        key = (row["variant"], row["effect"], row["sweep_variable"], row["sweep_value"])
        baseline = hpat.get(key)
        if baseline is None:
            continue
        out.append(
            {
                "variant": row["variant"],
                "effect": row["effect"],
                "sweep_variable": row["sweep_variable"],
                "sweep_value": row["sweep_value"],
                "all_linear_minus_hpat_top1_drop": _fmt(_float(row.get("top1_drop_mean")) - _float(baseline.get("top1_drop_mean"))),
                "all_linear_minus_hpat_top5_drop": _fmt(_float(row.get("top5_drop_mean")) - _float(baseline.get("top5_drop_mean"))),
                "all_linear_minus_hpat_prediction_change_rate": _fmt(
                    _float(row.get("prediction_change_rate_mean")) - _float(baseline.get("prediction_change_rate_mean"))
                ),
                "hpat_top1_drop": baseline.get("top1_drop_mean", ""),
                "all_linear_top1_drop": row.get("top1_drop_mean", ""),
                "evidence_label": "Boundary diagnostic delta; all-linear is not HPAT-mapping claim evidence.",
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def _run_all_linear(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    dataset_root: pathlib.Path,
    subset_file: pathlib.Path,
    max_samples: int | None,
    seed: int | None,
    device: str,
    pretrained: bool,
    model_variant: str,
    batch_size: int,
    save_logits: str,
    save_prediction_detail: str,
    repeat_seeds: str,
) -> dict[str, pathlib.Path]:
    """在 all-linear-smoke 边界下跑一次精度扫描（复用以非理想性精度扫描）。

    临时把 HPAT_WRITE_PROJECT_TABLES 设为 "0"（不管外面开没开），
    避免工程诊断实验污染仓库公共表格；退出时恢复原值。
    """
    previous = os.environ.get("HPAT_WRITE_PROJECT_TABLES")
    os.environ["HPAT_WRITE_PROJECT_TABLES"] = "0"
    try:
        return run_nonideality(
            output_dir=output_dir,
            config_path=config_path,
            dataset_root=dataset_root,
            subset_file=subset_file,
            max_samples=max_samples,
            seed=seed,
            device=device,
            pretrained=pretrained,
            model_variant=model_variant,
            operator_activity_csv=None,
            injection_boundary="all-linear-smoke",
            min_boundary_coverage=0.0,
            batch_size=batch_size,
            save_logits=save_logits,
            save_prediction_detail=save_prediction_detail,
            repeat_seeds=repeat_seeds,
        )
    finally:
        # 无论成功失败都恢复原环境变量，保证不泄漏状态
        if previous is None:
            os.environ.pop("HPAT_WRITE_PROJECT_TABLES", None)
        else:
            os.environ["HPAT_WRITE_PROJECT_TABLES"] = previous


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    hpat_run_dir: pathlib.Path | None,
    dataset_root: pathlib.Path,
    subset_file: pathlib.Path,
    max_samples: int | None,
    seed: int | None,
    device: str,
    pretrained: bool,
    model_variant: str,
    batch_size: int,
    save_logits: str,
    save_prediction_detail: str,
    repeat_seeds: str,
) -> dict[str, pathlib.Path]:
    """执行边界消融主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        hpat_run_dir：已有的 hpat-mapping 运行目录（None 自动找最新）。
        dataset_root/subset_file：固定子集。
        其余参数透传给 run_nonideality 精度扫描。
    返回：产出文件路径字典（summary_csv / delta_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 复用已有的 hpat-mapping 结果，不必重跑一遍
    source_hpat_run = hpat_run_dir or _latest_hpat_run_dir()
    hpat_source = source_hpat_run / "tables" / "nonideality_accuracy_by_severity.csv"
    hpat_rows = _read_by_severity(hpat_source)

    # 单独跑 all-linear 边界（写临时目录，不污染项目表格）
    all_linear_dir = ensure_dir(output_dir / "all_linear_smoke")
    all_linear_outputs = _run_all_linear(
        all_linear_dir,
        config_path,
        dataset_root,
        subset_file,
        max_samples,
        seed,
        device,
        pretrained,
        model_variant,
        batch_size,
        save_logits,
        save_prediction_detail,
        repeat_seeds,
    )
    all_linear_source = all_linear_outputs["by_severity_csv"]
    all_linear_rows = _read_by_severity(all_linear_source)

    # 生成摘要与差值两表
    summary = _summary_rows("hpat-mapping", hpat_rows) + _summary_rows("all-linear-smoke", all_linear_rows)
    delta = _delta_rows(hpat_rows, all_linear_rows)

    summary_csv = tables_dir / "nonideality_boundary_ablation_summary.csv"
    delta_csv = tables_dir / "nonideality_boundary_ablation_delta.csv"
    project_summary_csv = REPO_ROOT / "tables" / "nonideality_boundary_ablation_summary.csv"
    project_delta_csv = REPO_ROOT / "tables" / "nonideality_boundary_ablation_delta.csv"
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    write_csv(delta_csv, delta, DELTA_FIELDS)
    write_csv(project_summary_csv, summary, SUMMARY_FIELDS)
    write_csv(project_delta_csv, delta, DELTA_FIELDS)

    manifest_path = output_dir / "e_local_boundary_ablation_manifest.json"
    manifest = base_manifest("e_local_boundary_ablation", "HPAT-boundary versus all-linear diagnostic robustness ablation")
    manifest.update(
        {
            "status": "ready_with_limitations",
            "hpat_run_dir": relative(source_hpat_run),
            "hpat_by_severity_csv": relative(hpat_source),
            "hpat_by_severity_sha256": sha256_file(hpat_source),
            "all_linear_run_dir": relative(all_linear_dir),
            "all_linear_by_severity_csv": relative(all_linear_source),
            "all_linear_by_severity_sha256": sha256_file(all_linear_source),
            "outputs": [
                relative(summary_csv),
                relative(delta_csv),
                relative(project_summary_csv),
                relative(project_delta_csv),
                relative(all_linear_outputs["manifest"]),
            ],
            "row_count": len(summary),
            "delta_row_count": len(delta),
            "claim_boundary": DATASET_CLAIM_BOUNDARY,
            # 免责说明：hpat-mapping 行保留可主张资格；all-linear 行只是诊断
            "promotion_note": (
                "HPAT-mapping rows retain fixed-subset claim eligibility with limitations. "
                "All-linear rows are diagnostic sensitivity only and must not be described as HPAT-mapped execution."
            ),
            "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED") == "1",
            "caffeinate_reason": (
                "Caller set HPAT_CAFFEINATE_USED=1; boundary ablation is expected to be wrapped in caffeinate."
                if os.environ.get("HPAT_CAFFEINATE_USED") == "1"
                else "Not recorded; wrap full boundary ablation in caffeinate -dimsu."
            ),
        }
    )
    write_json(manifest_path, manifest)
    return {"summary_csv": summary_csv, "delta_csv": delta_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--hpat-run-dir", default="")
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "experiments" / "data" / "imagenette160_fixed_subset"))
    parser.add_argument("--subset-file", default=str(REPO_ROOT / "experiments" / "data" / "imagenette160_fixed_subset" / "fixed_subset.csv"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model-variant", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-logits", default="summary", choices=["summary", "none", "full"])
    parser.add_argument("--save-prediction-detail", default="topk-margin", choices=["basic", "topk-margin"])
    parser.add_argument("--repeat-seeds", default="20260706,20260707,20260708,20260709,20260710")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.hpat_run_dir) if args.hpat_run_dir else None,
        pathlib.Path(args.dataset_root),
        pathlib.Path(args.subset_file),
        args.max_samples,
        args.seed,
        args.device,
        args.pretrained,
        args.model_variant,
        args.batch_size,
        args.save_logits,
        args.save_prediction_detail,
        args.repeat_seeds,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
