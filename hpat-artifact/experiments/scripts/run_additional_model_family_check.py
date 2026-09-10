"""额外模型家族映射检查（run_additional_model_family_check.py）。

实验目的：验证 HPAT 的算子映射方法不止能处理 MobileViT，还能把
其它模型家族（additional family）的算子映射到光子矩阵乘法器上。
做法：用 torch/timm 真实跑一遍模型（trace 算子），看每个算子能否
映射到 HPAT 的算子类型。属于 P2 档"映射合理性"检查，不代表 HPAT
在这些模型上全面优于电子方案。

- 输入：hpat_experiment_config.json（含额外家族模型清单）；需要
  torch、timm 已安装，否则记录 blocked 状态。
- 产出（--output-dir 下）：
  * tables/additional_model_family_operator_activity.csv（算子活动）；
  * tables/additional_model_family_operator_summary.csv（汇总）；
  * additional_model_family_check_manifest.json；并同步写入仓库 tables/。
- 命令：python run_additional_model_family_check.py --output-dir <目录>
  [--config <配置>] [--device auto|cpu|mps|cuda] [--precision fp32|fp16|bf16|int8]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.activity_trace import trace_timm_operator_rows
from hpat_eval.mobilevit_loader import missing_packages
from hpat_eval.p2 import (
    ADDITIONAL_MODEL_OPERATOR_FIELDS,
    ADDITIONAL_MODEL_SUMMARY_FIELDS,
    P2_CLAIM_BOUNDARY,
    additional_family_trace_config,
    additional_model_family_summary_rows,
    attach_family_to_operator_rows,
)


def run(output_dir: pathlib.Path, config_path: pathlib.Path, device: str, precision: str) -> dict[str, pathlib.Path]:
    """执行额外模型家族检查主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        device：运行 torch trace 的设备（auto/cpu/mps/cuda）。
        precision：模型推理精度（fp32/fp16/bf16/int8）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    manifest = base_manifest("additional_model_family_check", "local/modelled P2 additional model-family mapping check")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "requested_device": device,
            "precision": precision,
            "claim_boundary": P2_CLAIM_BOUNDARY,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; default additional-family hook trace is a short batch-1 pass.",
        }
    )

    operator_rows = []
    backend = ""
    device_reason = ""
    status = "blocked"
    blocked_reason = ""
    # 先检查依赖：缺 torch/timm 就直接记 blocked，避免后续导入报错
    missing = missing_packages(["torch", "timm"])
    if missing:
        blocked_reason = f"Missing Python packages for torch/timm trace: {', '.join(missing)}"
    else:
        try:
            # 用 torch hook 真实跑一遍模型，得到算子活动；再给算子挂上模型家族标签
            trace_config = additional_family_trace_config(config)
            traced_rows, trace_meta = trace_timm_operator_rows(trace_config, requested_device=device, precision=precision)
            operator_rows = attach_family_to_operator_rows(traced_rows, config)
            backend = str(trace_meta.get("backend", ""))
            device_reason = str(trace_meta.get("device_reason", ""))
            failures = trace_meta.get("failures") or []
            # 状态分级：无失败且有行 → ok；有部分失败 → partial；没跑出任何行 → blocked
            status = "ok" if operator_rows and not failures else ("partial" if operator_rows else "blocked")
            blocked_reason = "; ".join(f"{item.get('model')}: {item.get('reason')}" for item in failures)
            manifest["trace_meta"] = trace_meta
        except Exception as exc:
            # 整个 trace 流程抛异常也记为 blocked，原因写入清单
            blocked_reason = str(exc)
            status = "blocked"

    # 汇总行：无论成功/部分/阻塞，都生成一条带状态说明的汇总
    summary_rows = additional_model_family_summary_rows(
        operator_rows=operator_rows,
        config=config,
        backend=backend,
        precision=precision,
        status=status,
        blocked_reason=blocked_reason,
        device_reason=device_reason,
    )

    activity_csv = tables_dir / "additional_model_family_operator_activity.csv"
    summary_csv = tables_dir / "additional_model_family_operator_summary.csv"
    project_activity_csv = REPO_ROOT / "tables" / "additional_model_family_operator_activity.csv"
    project_summary_csv = REPO_ROOT / "tables" / "additional_model_family_operator_summary.csv"
    write_csv(activity_csv, operator_rows, ADDITIONAL_MODEL_OPERATOR_FIELDS)
    write_csv(summary_csv, summary_rows, ADDITIONAL_MODEL_SUMMARY_FIELDS)
    write_csv(project_activity_csv, operator_rows, ADDITIONAL_MODEL_OPERATOR_FIELDS)
    write_csv(project_summary_csv, summary_rows, ADDITIONAL_MODEL_SUMMARY_FIELDS)

    manifest_path = output_dir / "additional_model_family_check_manifest.json"
    manifest.update(
        {
            "status": status,
            "blocked_reason": blocked_reason,
            "backend": backend,
            "device_reason": device_reason,
            "outputs": [relative(activity_csv), relative(summary_csv), relative(project_activity_csv), relative(project_summary_csv)],
            "operator_row_count": len(operator_rows),
            "summary_row_count": len(summary_rows),
            # 免责说明：额外家族的行只验证映射合理性，不支持 HPAT 全面优于他人的结论
            "promotion_note": "Additional-family rows test mapping plausibility only; they do not support broad HPAT superiority claims.",
        }
    )
    write_json(manifest_path, manifest)
    return {
        "activity_csv": activity_csv,
        "summary_csv": summary_csv,
        "project_activity_csv": project_activity_csv,
        "project_summary_csv": project_summary_csv,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数（含 device / precision）并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "int8"])
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config), args.device, args.precision)
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
