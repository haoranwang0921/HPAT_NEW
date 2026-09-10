"""MobileViT 算子活动跟踪（run_mobilevit_activity_trace.py）。

实验目的：用 torch/timm 真实跑一遍 MobileViT（batch=1），通过 hook 记录
每个算子（卷积/Linear/注意力等）的形状与乘加次数，产出"算子活动表"。
这张表是后续所有能耗核算（energy accounting）、QKV 流量校准、
trace-driven 活动等实验的共同上游数据源。
若 torch/timm 缺失或 trace 失败，则退化为"配置生成的代理算子行"。

- 输入：hpat_experiment_config.json（含模型变体配置）。
- 产出（--output-dir 下）：tables/mobilevit_operator_activity.csv、
  mobilevit_activity_manifest.json；按项目写入策略（ok-only/always/never）
  决定是否同步写入仓库 tables/。
- 命令：python run_mobilevit_activity_trace.py --output-dir <目录>
  [--config <配置>] [--device auto|cpu|mps|cuda] [--precision fp32|fp16|bf16|int8]
  [--project-write-policy ok-only|always|never]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import (
    REPO_ROOT,
    base_manifest,
    ensure_dir,
    load_json,
    project_writes_enabled,
    relative,
    sha256_file,
    write_csv,
    write_json,
)
from hpat_eval.activity_trace import OPERATOR_ACTIVITY_FIELDS, proxy_operator_rows, trace_timm_operator_rows
from hpat_eval.mobilevit_loader import missing_packages


def _should_write_project(status: str, policy: str) -> tuple[bool, str]:
    """按策略决定是否把结果写入仓库公共 tables/ 目录。

    规则：没开 HPAT_WRITE_PROJECT_TABLES=1 一律不写；
    开了之后按 policy 判断——always 总是写、never 从不写、
    ok-only 只在 trace 状态为 ok 时写（避免用代理数据覆盖公共表）。

    返回：(是否写, 原因字符串)。
    """
    if not project_writes_enabled():
        return False, "HPAT_WRITE_PROJECT_TABLES=1 was not explicitly set"
    if policy == "always":
        return True, "project write policy is always"
    if policy == "never":
        return False, "project write policy is never"
    if status == "ok":
        return True, "trace status ok and project write policy is ok-only"
    return False, "trace did not reach ok; ok-only policy preserves any existing project-level trace table"


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    device: str,
    precision: str,
    project_write_policy: str = "ok-only",
) -> dict[str, pathlib.Path]:
    """执行 MobileViT 算子活动跟踪主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        device：trace 所用设备。
        precision：推理精度。
        project_write_policy：项目表写入策略。
    返回：产出文件路径字典（csv / 可选 project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    manifest = base_manifest("mobilevit_activity_trace", "local trace or model/config-derived proxy")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "requested_device": device,
            "precision": precision,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; default activity trace/proxy is a short batch-1 pass.",
        }
    )

    # 依赖缺失 → 用代理行，标记 proxy 状态
    missing = missing_packages(["torch", "timm"])
    if missing:
        rows = proxy_operator_rows(config, backend="dependency-blocked proxy", precision=precision)
        manifest.update(
            {
                "status": "proxy",
                "blocked_reason": f"Missing Python packages for torch/timm trace: {', '.join(missing)}",
                "promotion_note": "Rows are config-derived proxies and cannot be cited as measured MobileViT operator activity.",
            }
        )
    else:
        try:
            # 真实 trace：跑模型抓算子活动；trace 成功且非空 → ok
            rows, trace_meta = trace_timm_operator_rows(config, requested_device=device, precision=precision)
            if not rows:
                # trace 没有抓到任何行 → 退化为代理行
                rows = proxy_operator_rows(config, backend="trace-failed proxy", precision=precision)
                manifest.update(
                    {
                        "status": "proxy",
                        "trace_meta": trace_meta,
                        "promotion_note": "No hook rows were collected; generated config-derived proxy rows.",
                    }
                )
            else:
                manifest.update(
                    {
                        "status": "ok",
                        "trace_meta": trace_meta,
                        "promotion_note": "Local torch/timm operator trace only. Do not cite as edge evidence.",
                    }
                )
        except Exception as exc:
            # trace 抛异常 → 也退化为代理行并记录原因
            rows = proxy_operator_rows(config, backend="trace-error proxy", precision=precision)
            manifest.update(
                {
                    "status": "proxy",
                    "blocked_reason": str(exc),
                    "promotion_note": "Torch/timm trace failed; generated config-derived proxy rows.",
                }
            )

    out_csv = tables_dir / "mobilevit_operator_activity.csv"
    project_csv = REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"
    write_csv(out_csv, rows, OPERATOR_ACTIVITY_FIELDS)
    # 按策略决定是否覆盖仓库公共表（默认 ok-only，避免代理数据污染）
    should_write_project, project_write_reason = _should_write_project(manifest.get("status", "proxy"), project_write_policy)
    if should_write_project:
        write_csv(project_csv, rows, OPERATOR_ACTIVITY_FIELDS)
    manifest_path = output_dir / "mobilevit_activity_manifest.json"
    outputs = [relative(out_csv)]
    if should_write_project:
        outputs.append(relative(project_csv))
    manifest.update(
        {
            "outputs": outputs,
            "project_write_policy": project_write_policy,
            "project_write_performed": should_write_project,
            "project_write_reason": project_write_reason,
            "project_csv": relative(project_csv),
            "row_count": len(rows),
        }
    )
    write_json(manifest_path, manifest)
    result = {"csv": out_csv, "manifest": manifest_path}
    if should_write_project:
        result["project_csv"] = project_csv
    return result


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--project-write-policy", default="ok-only", choices=["ok-only", "always", "never"])
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        args.device,
        args.precision,
        args.project_write_policy,
    )
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
