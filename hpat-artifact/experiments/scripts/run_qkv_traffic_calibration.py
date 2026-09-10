"""注意力 QKV 流量校准（run_qkv_traffic_calibration.py）。

实验目的：对 Transformer/MobileViT 注意力层，把"实际产生的 Q/K/V
数据传输量"校准成表。若上游提供了算子活动表（torch/timm 真实算子
跟踪得到的流量），则优先用真实轨迹；否则退化为用配置里的模型参数
（token 数、head 数、位宽等）做代理估算。本脚本是 P1 档（边缘本地
推理）能耗论证的输入之一。

- 输入：hpat_experiment_config.json、可选 mobilevit_operator_activity.csv
  （算子活动表，来自真实 torch hook 跟踪）。
- 产出（--output-dir 下）：qkv_traffic_calibrated.csv、tables/ 下同份、
  qkv_traffic_calibration_manifest.json；并同步写入仓库 tables/。
- 命令：python run_qkv_traffic_calibration.py --output-dir <目录>
  [--config <配置>] [--operator-activity-csv <文件>] [--trace-mode auto|config|trace]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.p1 import P1_CLAIM_BOUNDARY, QKV_CALIBRATED_FIELDS, qkv_calibrated_rows


def _torch_hook_rows(path: pathlib.Path) -> int:
    """统计给定 CSV 中来源为 torch_hooks（真实算子跟踪）的行数。

    参数 path：算子活动表路径。
    返回：来源为 torch_hooks 的行数；读取失败返回 0。
    """
    try:
        return sum(1 for row in read_csv(path) if row.get("trace_source") == "torch_hooks")
    except Exception:
        return 0


def _operator_activity_path(output_dir: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path | None:
    """挑选算子活动表：优先用显式传入的，其次找本次运行/仓库里的默认表。

    选表原则：优先选含 torch_hooks 真实轨迹的行（更可信）；
    都没有则返回 None，让调用方走纯配置的代理模式。

    参数：
        output_dir：本次运行输出目录。
        explicit：调用方显式指定的表路径。
    返回：选中的表路径，找不到返回 None。
    """
    if explicit:
        return explicit if explicit.exists() and explicit.stat().st_size > 0 else None
    candidates = []
    candidates.extend(
        [
            output_dir / "tables" / "mobilevit_operator_activity.csv",
            REPO_ROOT / "tables" / "mobilevit_operator_activity.csv",
        ]
    )
    existing = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    # 优先返回含真实 torch hook 轨迹的表
    for path in existing:
        if _torch_hook_rows(path) > 0:
            return path
    return existing[0] if existing else None


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    operator_activity_csv: pathlib.Path | None = None,
    trace_mode: str = "auto",
) -> dict[str, pathlib.Path]:
    """执行 QKV 流量校准主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        operator_activity_csv：可选的真实算子活动表。
        trace_mode：auto/config/trace 三选一，控制流量来源策略。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    operator_path = _operator_activity_path(output_dir, operator_activity_csv)
    operator_rows = read_csv(operator_path) if operator_path else []
    # 核心计算委托给 hpat_eval.p1.qkv_calibrated_rows（有轨迹用轨迹，否则用配置代理）
    rows = qkv_calibrated_rows(config, operator_rows=operator_rows, trace_mode=trace_mode)

    out_csv = output_dir / "qkv_traffic_calibrated.csv"
    tables_csv = tables_dir / "qkv_traffic_calibrated.csv"
    project_csv = REPO_ROOT / "tables" / "qkv_traffic_calibrated.csv"
    manifest_path = output_dir / "qkv_traffic_calibration_manifest.json"
    write_csv(out_csv, rows, QKV_CALIBRATED_FIELDS)
    write_csv(tables_csv, rows, QKV_CALIBRATED_FIELDS)
    write_csv(project_csv, rows, QKV_CALIBRATED_FIELDS)

    manifest = base_manifest("qkv_traffic_calibration", "local/modelled P1 Q/K/V traffic calibration")
    manifest.update(
        {
            # 只要含 torch_hooks 行就标 trace-backed（真实轨迹背书），否则标 proxy
            "status": "trace-backed" if any(row.get("trace_source") == "torch_hooks" for row in rows) else "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "operator_activity_csv": relative(operator_path) if operator_path else "",
            "operator_activity_csv_sha256": sha256_file(operator_path) if operator_path else None,
            "trace_mode": trace_mode,
            "outputs": [relative(out_csv), relative(tables_csv), relative(project_csv)],
            "row_count": len(rows),
            "traceability": (
                "Rows are derived from local torch/timm operator activity when hook rows are available; otherwise configured MobileViT variants and proxy settings are used."
            ),
            "claim_boundary": P1_CLAIM_BOUNDARY,
            "promotion_note": "Replace proxy token/config values with simulator-exported layer activity before using as calibrated physical-design evidence.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled traffic calibration run.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": out_csv, "project_csv": project_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数（含可选的活动表与 trace 模式）并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--operator-activity-csv", default="")
    parser.add_argument("--trace-mode", default="auto", choices=["auto", "config", "trace"])
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.operator_activity_csv) if args.operator_activity_csv else None,
        args.trace_mode,
    )
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
