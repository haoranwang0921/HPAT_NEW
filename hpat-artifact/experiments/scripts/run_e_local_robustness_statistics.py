"""E-local 鲁棒性统计（run_e_local_robustness_statistics.py）。

实验目的：对一次 e-local 固定子集实验的"逐样本明细"（top-k/margin
预测详情 CSV）做深加工统计，产出五张表：
  1) 各模型/类别的"清洁精度"（imagenette_clean_accuracy_by_model.csv）；
  2) 逐类别 × 逐严重度的精度变化（nonideality_accuracy_classwise.csv）；
  3) bootstrap 置信区间（nonideality_accuracy_bootstrap_ci.csv）——
     用自助重采样估计 top-1/top-5 下降等指标的不确定性；
  4) 每个效果下"受影响最大的类别"（nonideality_accuracy_worst_class.csv）；
  5) margin 漂移分布（nonideality_margin_drift_by_severity.csv）——
     扰动前后 top-1 置信度间隔的变化。

- 输入：上游 e-local 运行目录 raw/nonideality_prediction_topk_margin.csv
  （需先以 --save-prediction-detail topk-margin 跑固定子集实验）。
- 产出（--output-dir 下）：tables/ 下五张 CSV +
  e_local_robustness_statistics_manifest.json；并同步写入仓库 tables/。
- 命令：python run_e_local_robustness_statistics.py --output-dir <目录>
  [--run-dir <上游运行目录>] [--bootstrap-repeats 500] ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_csv, write_json
from run_nonideality_accuracy_sweep import DATASET_CLAIM_BOUNDARY


# 统一证据标签：本脚本的统计都来自固定外部公共子集 + hook 注入的 logits
EVIDENCE_LABEL = (
    "Imagenette validation external-public fixed-subset robustness statistics; "
    "derived from hook-injected MobileViT logits"
)

# 以下均为 CSV 表头定义
CLEAN_FIELDS = [
    "variant",
    "class_name",
    "synset",
    "sample_count",
    "clean_top1",
    "clean_top5",
    "evidence_label",
    "claim_boundary",
]

CLASSWISE_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "class_name",
    "synset",
    "sample_count",
    "trial_count",
    "clean_top1_mean",
    "perturbed_top1_mean",
    "top1_drop_mean",
    "clean_top5_mean",
    "perturbed_top5_mean",
    "top5_drop_mean",
    "prediction_change_rate_mean",
    "margin_delta_mean",
    "top5_jaccard_mean",
    "evidence_label",
    "claim_boundary",
]

BOOTSTRAP_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "metric",
    "bootstrap_repeats",
    "bootstrap_sample_cap",
    "sample_count",
    "trial_count",
    "mean",
    "p05",
    "p50",
    "p95",
    "evidence_label",
    "claim_boundary",
]

WORST_CLASS_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "worst_sweep_value",
    "worst_class_name",
    "worst_synset",
    "worst_top1_drop_mean",
    "worst_prediction_change_rate_mean",
    "sample_count",
    "trial_count",
    "evidence_label",
    "claim_boundary",
]

MARGIN_FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "sample_count",
    "trial_count",
    "margin_delta_mean",
    "margin_delta_p05",
    "margin_delta_p50",
    "margin_delta_p95",
    "top1_confidence_delta_mean",
    "top5_jaccard_mean",
    "evidence_label",
    "claim_boundary",
]


def _float(value: Any, default: float = 0.0) -> float:
    """把值转成浮点数；空值/None 返回默认值。"""
    if value in ("", None):
        return default
    return float(value)


def _fmt(value: float, digits: int = 4) -> str:
    """格式化浮点数为指定小数位。"""
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float:
    """求均值（空列表返回 0.0）。"""
    return sum(values) / max(len(values), 1)


def _percentile(values: list[float], pct: float) -> float:
    """计算百分位数（线性插值；空列表返回 0.0）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _latest_run_dir() -> pathlib.Path:
    """在 results 目录找最新一个含非理想性原始产出的 e-local 运行目录。

    判定条件：raw/nonideality_prediction_changes.csv 存在。
    """
    result_root = REPO_ROOT / "experiments" / "results"
    candidates = sorted(
        [path for path in result_root.glob("run_*") if (path / "raw" / "nonideality_prediction_changes.csv").exists()],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError("No E-local run directory with nonideality raw outputs was found.")
    return candidates[-1]


def _source_path(run_dir: pathlib.Path, filename: str) -> pathlib.Path:
    """找明细文件：优先本次运行目录 raw/ 下，其次仓库 tables/ 下。

    参数：
        run_dir：上游 e-local 运行目录。
        filename：文件名。
    返回：找到的路径（两处都没有则返回本次运行目录的候选路径，供后续报错）。
    """
    candidate = run_dir / "raw" / filename
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    project_candidate = REPO_ROOT / "tables" / filename
    if project_candidate.exists() and project_candidate.stat().st_size > 0:
        return project_candidate
    return candidate


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    """提取明细行的分组键：模型、效果、扫描变量、取值。"""
    return (row["variant"], row["effect"], row["sweep_variable"], row["sweep_value"])


def _class_key(row: dict[str, str]) -> tuple[str, str]:
    """提取类别键：优先用 class_name，其次 synset，再退到 label_N。"""
    class_name = row.get("class_name", "").strip() or row.get("synset", "").strip() or f"label_{row.get('label', '')}"
    return class_name, row.get("synset", "").strip()


def _clean_accuracy_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """统计"清洁"（未扰动）精度：按模型×类别分组，含整体（__overall__）行。

    参数 rows：topk_margin 明细行（同一张图可能有多条 trial 记录，
    先按 (variant, sample_index) 去重取一条）。
    返回：每类别/整体的清洁 top-1、top-5 精度行。
    """
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        seen[(row["variant"], row["sample_index"])] = row
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in seen.values():
        class_name, synset = _class_key(row)
        grouped[(row["variant"], class_name, synset)].append(row)
        # 追加一个整体分组，用于输出所有类别的平均精度
        grouped[(row["variant"], "__overall__", "")].append(row)
    out: list[dict[str, Any]] = []
    for (variant, class_name, synset), group in sorted(grouped.items()):
        top1 = [_float(row.get("clean_correct_top1")) for row in group if row.get("clean_correct_top1") not in ("", None)]
        top5 = [_float(row.get("clean_correct_top5")) for row in group if row.get("clean_correct_top5") not in ("", None)]
        out.append(
            {
                "variant": variant,
                "class_name": class_name,
                "synset": synset,
                "sample_count": len(group),
                "clean_top1": _fmt(100.0 * _mean(top1)) if top1 else "",
                "clean_top5": _fmt(100.0 * _mean(top5)) if top5 else "",
                "evidence_label": EVIDENCE_LABEL,
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def _classwise_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """按 模型×效果×严重度×类别 分组，统计扰动前后精度与变化指标。

    参数 rows：topk_margin 明细行。
    返回：逐类别的精度变化行列表。
    """
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        class_name, synset = _class_key(row)
        grouped[(*_key(row), class_name, synset)].append(row)
    out: list[dict[str, Any]] = []
    for (variant, effect, variable, value, class_name, synset), group in sorted(grouped.items()):
        clean_top1 = [_float(row.get("clean_correct_top1")) for row in group if row.get("clean_correct_top1") not in ("", None)]
        pert_top1 = [_float(row.get("perturbed_correct_top1")) for row in group if row.get("perturbed_correct_top1") not in ("", None)]
        clean_top5 = [_float(row.get("clean_correct_top5")) for row in group if row.get("clean_correct_top5") not in ("", None)]
        pert_top5 = [_float(row.get("perturbed_correct_top5")) for row in group if row.get("perturbed_correct_top5") not in ("", None)]
        change = [_float(row.get("changed")) for row in group]
        margin = [_float(row.get("margin_delta")) for row in group if row.get("margin_delta") not in ("", None)]
        jaccard = [_float(row.get("top5_jaccard")) for row in group if row.get("top5_jaccard") not in ("", None)]
        out.append(
            {
                "variant": variant,
                "effect": effect,
                "sweep_variable": variable,
                "sweep_value": value,
                "class_name": class_name,
                "synset": synset,
                "sample_count": len({row["sample_index"] for row in group}),
                "trial_count": len({(row.get("trial_seed", ""), row.get("trial_index", "")) for row in group}),
                "clean_top1_mean": _fmt(100.0 * _mean(clean_top1)) if clean_top1 else "",
                "perturbed_top1_mean": _fmt(100.0 * _mean(pert_top1)) if pert_top1 else "",
                "top1_drop_mean": _fmt(max(0.0, 100.0 * (_mean(clean_top1) - _mean(pert_top1)))) if clean_top1 and pert_top1 else "",
                "clean_top5_mean": _fmt(100.0 * _mean(clean_top5)) if clean_top5 else "",
                "perturbed_top5_mean": _fmt(100.0 * _mean(pert_top5)) if pert_top5 else "",
                "top5_drop_mean": _fmt(max(0.0, 100.0 * (_mean(clean_top5) - _mean(pert_top5)))) if clean_top5 and pert_top5 else "",
                "prediction_change_rate_mean": _fmt(100.0 * _mean(change)),
                "margin_delta_mean": f"{_mean(margin):.8f}" if margin else "",
                "top5_jaccard_mean": f"{_mean(jaccard):.8f}" if jaccard else "",
                "evidence_label": EVIDENCE_LABEL,
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def _metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    """从明细行中提取指定指标的逐样本值列表。

    支持的 metric：top1_drop / top5_drop / prediction_change_rate / margin_delta。
    """
    if metric == "top1_drop":
        return [
            100.0 * (_float(row.get("clean_correct_top1")) - _float(row.get("perturbed_correct_top1")))
            for row in rows
            if row.get("clean_correct_top1") not in ("", None) and row.get("perturbed_correct_top1") not in ("", None)
        ]
    if metric == "top5_drop":
        return [
            100.0 * (_float(row.get("clean_correct_top5")) - _float(row.get("perturbed_correct_top5")))
            for row in rows
            if row.get("clean_correct_top5") not in ("", None) and row.get("perturbed_correct_top5") not in ("", None)
        ]
    if metric == "prediction_change_rate":
        return [100.0 * _float(row.get("changed")) for row in rows]
    if metric == "margin_delta":
        return [_float(row.get("margin_delta")) for row in rows if row.get("margin_delta") not in ("", None)]
    raise ValueError(metric)


def _bootstrap_summary(values: list[float], repeats: int, sample_cap: int, seed: int) -> tuple[float, float, float, float]:
    """对一组样本做 bootstrap（自助重采样）估计均值及其 90% 区间。

    做法：有放回地抽取与原样本同数量的样本 repeats 次，每次算均值，
    取这些均值的 p05/p50/p95 作为置信区间。样本超过 sample_cap 时先随机截取。

    返回：(原始均值, p05, p50, p95)。
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    rng = random.Random(seed)
    source = values if len(values) <= sample_cap else rng.sample(values, sample_cap)
    draws: list[float] = []
    for _ in range(repeats):
        draws.append(_mean([source[rng.randrange(len(source))] for _sample in range(len(source))]))
    return _mean(values), _percentile(draws, 5), _percentile(draws, 50), _percentile(draws, 95)


def _bootstrap_rows(rows: list[dict[str, str]], repeats: int, sample_cap: int, seed: int) -> list[dict[str, Any]]:
    """对每个 严重度×指标 组合做 bootstrap 置信区间估计。

    参数：
        rows：topk_margin 明细行。
        repeats：重采样次数。
        sample_cap：每组最多用多少个样本做重采样。
        seed：随机种子（随分组偏移，保证可复现且各组独立）。
    返回：bootstrap 结果行列表。
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        variant, effect, variable, value = key
        for metric in ["top1_drop", "top5_drop", "prediction_change_rate", "margin_delta"]:
            values = _metric_values(group, metric)
            if not values:
                continue
            mean, p05, p50, p95 = _bootstrap_summary(values, repeats, sample_cap, seed + group_index * 17 + len(metric))
            out.append(
                {
                    "variant": variant,
                    "effect": effect,
                    "sweep_variable": variable,
                    "sweep_value": value,
                    "metric": metric,
                    "bootstrap_repeats": repeats,
                    "bootstrap_sample_cap": min(sample_cap, len(values)),
                    "sample_count": len({row["sample_index"] for row in group}),
                    "trial_count": len({(row.get("trial_seed", ""), row.get("trial_index", "")) for row in group}),
                    "mean": f"{mean:.8f}",
                    "p05": f"{p05:.8f}",
                    "p50": f"{p50:.8f}",
                    "p95": f"{p95:.8f}",
                    "evidence_label": EVIDENCE_LABEL,
                    "claim_boundary": DATASET_CLAIM_BOUNDARY,
                }
            )
    return out


def _worst_class_rows(classwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从逐类别统计中挑出每个 (模型, 效果, 扫描变量) 的"最受影响类别"。

    最受影响 = top1_drop_mean 最大（并列时看 prediction_change_rate_mean）。
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in classwise:
        grouped[(str(row["variant"]), str(row["effect"]), str(row["sweep_variable"]))].append(row)
    out: list[dict[str, Any]] = []
    for (variant, effect, variable), group in sorted(grouped.items()):
        worst = max(
            group,
            key=lambda row: (_float(row.get("top1_drop_mean")), _float(row.get("prediction_change_rate_mean"))),
        )
        out.append(
            {
                "variant": variant,
                "effect": effect,
                "sweep_variable": variable,
                "worst_sweep_value": worst.get("sweep_value", ""),
                "worst_class_name": worst.get("class_name", ""),
                "worst_synset": worst.get("synset", ""),
                "worst_top1_drop_mean": worst.get("top1_drop_mean", ""),
                "worst_prediction_change_rate_mean": worst.get("prediction_change_rate_mean", ""),
                "sample_count": worst.get("sample_count", ""),
                "trial_count": worst.get("trial_count", ""),
                "evidence_label": EVIDENCE_LABEL,
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def _margin_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """按严重度汇总"margin 漂移"（扰动前后 top-1 间隔变化）分布。

    参数 rows：topk_margin 明细行。
    返回：每严重度的 margin_delta 均值/分位数、置信度变化均值、Jaccard 均值。
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for (variant, effect, variable, value), group in sorted(grouped.items()):
        margin = [_float(row.get("margin_delta")) for row in group if row.get("margin_delta") not in ("", None)]
        confidence_delta = [
            _float(row.get("perturbed_top1_confidence")) - _float(row.get("clean_top1_confidence"))
            for row in group
            if row.get("perturbed_top1_confidence") not in ("", None)
            and row.get("clean_top1_confidence") not in ("", None)
        ]
        jaccard = [_float(row.get("top5_jaccard")) for row in group if row.get("top5_jaccard") not in ("", None)]
        out.append(
            {
                "variant": variant,
                "effect": effect,
                "sweep_variable": variable,
                "sweep_value": value,
                "sample_count": len({row["sample_index"] for row in group}),
                "trial_count": len({(row.get("trial_seed", ""), row.get("trial_index", "")) for row in group}),
                "margin_delta_mean": f"{_mean(margin):.8f}" if margin else "",
                "margin_delta_p05": f"{_percentile(margin, 5):.8f}" if margin else "",
                "margin_delta_p50": f"{_percentile(margin, 50):.8f}" if margin else "",
                "margin_delta_p95": f"{_percentile(margin, 95):.8f}" if margin else "",
                "top1_confidence_delta_mean": f"{_mean(confidence_delta):.8f}" if confidence_delta else "",
                "top5_jaccard_mean": f"{_mean(jaccard):.8f}" if jaccard else "",
                "evidence_label": EVIDENCE_LABEL,
                "claim_boundary": DATASET_CLAIM_BOUNDARY,
            }
        )
    return out


def run(
    output_dir: pathlib.Path,
    run_dir: pathlib.Path | None,
    repeats: int,
    sample_cap: int,
    seed: int,
) -> dict[str, pathlib.Path]:
    """鲁棒性统计主流程：读取明细并生成五张统计表。

    参数：
        output_dir：结果输出目录。
        run_dir：上游 e-local 运行目录（None 自动找最新）。
        repeats：bootstrap 重采样次数。
        sample_cap：bootstrap 每组样本上限。
        seed：bootstrap 随机种子。
    返回：产出文件路径字典（含 manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 定位上游明细文件：topk-margin 详情是统计的唯一数据源
    source_run_dir = run_dir or _latest_run_dir()
    topk_source = _source_path(source_run_dir, "nonideality_prediction_topk_margin.csv")
    if not topk_source.exists() or topk_source.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing top-k/margin prediction detail CSV: {topk_source}. "
            "Run E-local with --save-prediction-detail topk-margin first."
        )
    rows = read_csv(topk_source)
    # 依次计算五张表
    clean = _clean_accuracy_rows(rows)
    classwise = _classwise_rows(rows)
    bootstrap = _bootstrap_rows(rows, repeats=repeats, sample_cap=sample_cap, seed=seed)
    worst = _worst_class_rows(classwise)
    margin = _margin_rows(rows)

    outputs: dict[str, pathlib.Path] = {}
    for name, data, fields in [
        ("imagenette_clean_accuracy_by_model.csv", clean, CLEAN_FIELDS),
        ("nonideality_accuracy_classwise.csv", classwise, CLASSWISE_FIELDS),
        ("nonideality_accuracy_bootstrap_ci.csv", bootstrap, BOOTSTRAP_FIELDS),
        ("nonideality_accuracy_worst_class.csv", worst, WORST_CLASS_FIELDS),
        ("nonideality_margin_drift_by_severity.csv", margin, MARGIN_FIELDS),
    ]:
        path = tables_dir / name
        project_path = REPO_ROOT / "tables" / name
        write_csv(path, data, fields)
        write_csv(project_path, data, fields)
        outputs[name] = path

    manifest_path = output_dir / "e_local_robustness_statistics_manifest.json"
    manifest = base_manifest("e_local_robustness_statistics", EVIDENCE_LABEL)
    manifest.update(
        {
            "status": "ready_with_limitations",
            "source_run_dir": relative(source_run_dir),
            "source_topk_margin_csv": relative(topk_source),
            "source_topk_margin_sha256": sha256_file(topk_source),
            "bootstrap_repeats": repeats,
            "bootstrap_sample_cap": sample_cap,
            "bootstrap_seed": seed,
            "row_counts": {
                "raw_topk_margin": len(rows),
                "clean_accuracy": len(clean),
                "classwise": len(classwise),
                "bootstrap_ci": len(bootstrap),
                "worst_class": len(worst),
                "margin_drift": len(margin),
            },
            "outputs": [relative(path) for path in outputs.values()]
            + [relative(REPO_ROOT / "tables" / name) for name in outputs],
            "claim_boundary": DATASET_CLAIM_BOUNDARY,
            # 免责说明：统计只加强固定子集鲁棒性的解读，不升格为全量验证
            "promotion_note": (
                "Derived statistics strengthen fixed-subset robustness interpretation only. "
                "They do not promote the run to full ImageNet validation, silicon robustness, "
                "or measured edge/mobile deployment evidence."
            ),
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is post-processing over an existing E-local run.",
        }
    )
    write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--bootstrap-sample-cap", type=int, default=2048)
    parser.add_argument("--bootstrap-seed", type=int, default=20260706)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.run_dir) if args.run_dir else None,
        max(1, args.bootstrap_repeats),
        max(1, args.bootstrap_sample_cap),
        args.bootstrap_seed,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
