"""P2 档就绪度汇总（run_p2_readiness_summary.py）。

实验目的：检查 P2 档（更多设计细节）的四类证据是否就绪：
  P2-layout-area          版图面积可行性代理
  P2-thermal-tuning       热调谐压力代理
  P2-public-edge-context  公共边缘上下文来源
  P2-additional-model-family  额外模型家族映射检查
对每个实验判定 status / claim_eligible / evidence_tier / blockers，并输出
JSON 与 Markdown 报告。它只是"元数据门禁"，不做计算。

- 输入：仓库/运行目录 tables/ 下的 layout_area_feasibility_proxy.csv、
  thermal_tuning_stress.csv、public_edge_context_sources.csv、
  additional_model_family_operator_summary.csv 等。
- 产出（--output-dir 下）：tables/p2_readiness_summary.json/.md，
  并同步写入仓库 tables/；p2_readiness_summary_manifest.json。
- 命令：python run_p2_readiness_summary.py --output-dir <目录>
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_json, write_text
from hpat_eval.p2 import P2_CLAIM_BOUNDARY


P2_READINESS_SCHEMA_VERSION = "p2-readiness-v1"


def _exists(path: pathlib.Path) -> bool:
    """判断文件是否存在且非空。"""
    return path.exists() and path.stat().st_size > 0


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 CSV；文件不存在则返回空列表。"""
    return read_csv(path) if _exists(path) else []


def _table_path(output_dir: pathlib.Path, name: str) -> pathlib.Path:
    """找表：优先本次运行目录 tables/ 下，其次仓库公共 tables/ 下。"""
    run_path = output_dir / "tables" / name
    if _exists(run_path):
        return run_path
    return REPO_ROOT / "tables" / name


def _p2_rows(output_dir: pathlib.Path) -> list[dict[str, Any]]:
    """判定四个 P2 实验的就绪状态并组装成行。

    参数 output_dir：本次运行输出目录。
    返回：就绪度行列表。
    """
    layout = _rows(_table_path(output_dir, "layout_area_feasibility_proxy.csv"))
    thermal = _rows(_table_path(output_dir, "thermal_tuning_stress.csv"))
    sources = _rows(_table_path(output_dir, "public_edge_context_sources.csv"))
    family = _rows(_table_path(output_dir, "additional_model_family_operator_summary.csv"))

    # 额外模型家族：有任意一行 row_count>0 视为成功；全部 blocked 视为失败
    family_ok = bool(family) and any(int(float(row.get("row_count") or 0)) > 0 for row in family)
    family_blocked = bool(family) and all(row.get("status") == "blocked" for row in family)
    return [
        {
            "experiment": "P2-layout-area",
            "status": "ready_with_limitations" if layout else "blocked",
            "claim_eligible": bool(layout),
            "evidence_tier": "local/modelled layout proxy" if layout else "not claimable",
            "trace_readiness": "not required",
            "literature_readiness": "not required",
            "layout_readiness": "proxy only; no layout closure",
            "meaning": "Layout/area proxy rows expose MRR, converter, interconnect, budget, path-loss, and thermal-density pressure.",
            "blockers": "No floorplan, routing, DRC/LVS, thermal layout, or fabricated silicon validation.",
            "promotion_requirements": "Layout database or floorplan model with calibrated device and routing assumptions.",
        },
        {
            "experiment": "P2-thermal-tuning",
            "status": "ready_with_limitations" if thermal else "blocked",
            "claim_eligible": bool(thermal),
            "evidence_tier": "local/modelled thermal stress proxy" if thermal else "not claimable",
            "trace_readiness": "not required",
            "literature_readiness": "not required",
            "layout_readiness": "proxy only; no packaged-device thermal validation",
            "meaning": "Thermal stress rows bound drift, calibration multiplier, retune interval, latency overhead, and energy overhead proxies.",
            "blockers": "No packaged-device thermal measurement, heater control trace, or calibrated temperature map.",
            "promotion_requirements": "Measured or simulator-calibrated thermal placement/tuning evidence tied to HPAT layout assumptions.",
        },
        {
            "experiment": "P2-public-edge-context",
            "status": "context_ready" if sources else "blocked",
            "claim_eligible": bool(sources),
            "evidence_tier": "literature-context / official public benchmark" if sources else "not claimable",
            "trace_readiness": "not required",
            "literature_readiness": "refreshed source ledger" if sources else "missing source ledger",
            "layout_readiness": "not layout evidence",
            "meaning": "Public edge/mobile sources refresh motivation and methodology context only.",
            "blockers": "Not an HPAT baseline and not comparable speedup evidence.",
            "promotion_requirements": "Keep as context only; collect author-measured comparable baselines before stronger claims.",
        },
        {
            "experiment": "P2-additional-model-family",
            "status": "ready_with_limitations" if family_ok else ("blocked" if family_blocked else "partial"),
            "claim_eligible": family_ok,
            "evidence_tier": "local torch/timm trace mapping proxy" if family_ok else "blocked or partial local trace",
            "trace_readiness": "trace-backed" if family_ok else "blocked",
            "literature_readiness": "EfficientFormer context source refreshed",
            "layout_readiness": "not layout evidence",
            "meaning": "Additional-family trace checks whether the linear/MVM mapping insight extends beyond the primary MobileViT rows.",
            "blockers": "" if family_ok else "Torch/timm trace was not available or produced no rows.",
            "promotion_requirements": "HPAT simulator/export support and calibrated energy model for the additional model family before quantitative claims.",
        },
    ]


def _md(summary: dict[str, Any]) -> str:
    """把就绪度汇总渲染成 Markdown 表格文本。"""
    lines = [
        "# P2 Readiness Summary",
        "",
        f"Schema: `{summary['schema_version']}`",
        f"Overall P2 ready with limitations: `{str(summary['overall_ready_with_limitations']).lower()}`",
        "",
        "| Experiment | Status | Claim eligible | Evidence tier | Trace | Literature | Layout |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in summary["experiments"]:
        lines.append(
            f"| {row['experiment']} | {row['status']} | {str(row['claim_eligible']).lower()} | "
            f"{row['evidence_tier']} | {row['trace_readiness']} | {row['literature_readiness']} | {row['layout_readiness']} |"
        )
    lines.extend(["", "## Claim Boundary", "", summary["claim_boundary"], "", "## Blockers And Promotion Requirements", ""])
    for row in summary["experiments"]:
        lines.append(f"- {row['experiment']}: {row['blockers'] or 'no current blocker for bounded P2 use'} Promotion: {row['promotion_requirements']}")
    return "\n".join(lines) + "\n"


def run(output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行 P2 就绪度汇总主流程。

    参数 output_dir：结果输出目录。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    rows = _p2_rows(output_dir)
    summary = {
        "schema_version": P2_READINESS_SCHEMA_VERSION,
        "experiments": rows,
        "overall_ready_with_limitations": all(row["claim_eligible"] for row in rows),
        "claim_boundary": P2_CLAIM_BOUNDARY,
    }
    json_path = tables_dir / "p2_readiness_summary.json"
    project_json = REPO_ROOT / "tables" / "p2_readiness_summary.json"
    md_path = tables_dir / "p2_readiness_summary.md"
    project_md = REPO_ROOT / "tables" / "p2_readiness_summary.md"
    write_json(json_path, summary)
    write_json(project_json, summary)
    text = _md(summary)
    md_path.write_text(text, encoding="utf-8")
    write_text(project_md, text)

    manifest_path = output_dir / "p2_readiness_summary_manifest.json"
    manifest = base_manifest("p2_readiness_summary", "P2 readiness and claim-boundary gate")
    manifest.update(
        {
            "status": "ready_with_limitations" if summary["overall_ready_with_limitations"] else "partial",
            "p2_readiness_json": relative(json_path),
            "p2_readiness_json_sha256": sha256_file(json_path),
            "outputs": [relative(json_path), relative(project_json), relative(md_path), relative(project_md)],
            "claim_boundary": summary["claim_boundary"],
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; readiness summary is a short metadata gate.",
        }
    )
    write_json(manifest_path, manifest)
    return {"json": json_path, "project_json": project_json, "md": md_path, "project_md": project_md, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir))
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
