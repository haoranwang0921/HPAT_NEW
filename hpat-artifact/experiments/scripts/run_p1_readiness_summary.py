"""P1 档就绪度汇总（run_p1_readiness_summary.py）。

实验目的：检查 P1 档（边缘本地推理，e-local 相关）的四类证据是否就绪：
  P1-E5  QKV 流量校准表（是否 trace-backed）
  P1-E6  单位能耗标定（是否含文献数值来源）
  P1-E7  架构消融表
  P1-E8  规模物理资源代理表
对每个实验判定 status / claim_eligible / evidence_tier / blockers，并输出
机器可读 JSON 与 Markdown 报告。它只是"元数据门禁"，不做计算。

- 输入：仓库/运行目录 tables/ 下的 qkv_traffic_calibrated.csv、
  energy_unit_costs.csv、hpat_architecture_ablation.csv、scalability_physical_proxy.csv 等。
- 产出（--output-dir 下）：tables/p1_readiness_summary.json/.md，
  并同步写入仓库 tables/；p1_readiness_summary_manifest.json。
- 命令：python run_p1_readiness_summary.py --output-dir <目录>
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, write_json, write_text


P1_READINESS_SCHEMA_VERSION = "p1-readiness-v1"


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


def _p1_rows(output_dir: pathlib.Path) -> list[dict[str, Any]]:
    """判定四个 P1 实验的就绪状态并组装成行。

    参数 output_dir：本次运行输出目录。
    返回：就绪度行列表（每行含 experiment/status/claim_eligible/blockers 等）。
    """
    qkv = _rows(_table_path(output_dir, "qkv_traffic_calibrated.csv"))
    unit_costs = _rows(_table_path(output_dir, "energy_unit_costs.csv"))
    ledger = _rows(_table_path(output_dir, "energy_unit_cost_source_ledger.csv"))
    ablation = _rows(_table_path(output_dir, "hpat_architecture_ablation.csv"))
    scale = _rows(_table_path(output_dir, "scalability_physical_proxy.csv"))

    # P1-E5：QKV 流量表是否有真实 torch hook 轨迹行
    qkv_trace = any(row.get("trace_source") == "torch_hooks" for row in qkv)
    qkv_status = "ready_with_limitations" if qkv_trace else "proxy"
    qkv_tier = "trace-backed local" if qkv_trace else "config-derived local/modelled"
    qkv_blockers = (
        "HPAT simulator activity export still absent; not physical-design calibrated"
        if qkv_trace
        else "No torch_hooks Q/K/V trace rows in calibrated traffic table"
    )

    # P1-E6：单位能耗的来源角色——是否有文献数值来源
    roles = {row.get("source_role", "") for row in ledger} | {row.get("source_role", "") for row in unit_costs}
    has_lit_numeric = "literature_numeric" in roles
    has_lit_context = "literature_context" in roles
    cost_status = "ready_with_limitations" if has_lit_numeric else "literature_context_modelled"
    cost_tier = "literature-calibrated modelled" if has_lit_numeric else "literature-context/modelled"
    cost_blockers = (
        "Directly comparable HPAT simulator-derived or cited unit costs still absent"
        if not has_lit_numeric
        else "Still not silicon-measured HPAT energy"
    )

    # P1-E7 / P1-E8：消融表与规模代理表是否非空
    ablation_ready = bool(ablation)
    scale_ready = bool(scale) and all("mrr_area_proxy_mm2" in row for row in scale[:1])
    return [
        {
            "experiment": "P1-E5",
            "status": qkv_status,
            "claim_eligible": bool(qkv),
            "evidence_tier": qkv_tier,
            "trace_readiness": "trace-backed" if qkv_trace else "config-proxy",
            "literature_readiness": "not required",
            "layout_readiness": "not layout evidence",
            "meaning": "Q/K/V traffic and reprogramming overhead table is available with bounded labels.",
            "blockers": qkv_blockers,
            "promotion_requirements": "HPAT simulator/exported per-layer activity and physical unit-cost reconciliation.",
        },
        {
            "experiment": "P1-E6",
            "status": cost_status,
            "claim_eligible": bool(unit_costs),
            "evidence_tier": cost_tier,
            "trace_readiness": "not required",
            "literature_readiness": "literature-numeric" if has_lit_numeric else ("literature-context" if has_lit_context else "internal-only"),
            "layout_readiness": "not layout evidence",
            "meaning": "Unit-cost ranges and source ledger are available, but numeric values remain modelled unless a source is marked literature_numeric.",
            "blockers": cost_blockers,
            "promotion_requirements": "Directly comparable cited/simulator unit-cost values with unit, scope, precision, and technology assumptions.",
        },
        {
            "experiment": "P1-E7",
            "status": "ready_with_limitations" if ablation_ready else "blocked",
            "claim_eligible": ablation_ready,
            "evidence_tier": "local/modelled architecture ablation" if ablation_ready else "not claimable",
            "trace_readiness": "not required",
            "literature_readiness": "not required",
            "layout_readiness": "not layout evidence",
            "meaning": "Architecture ablation table covers broadcast, weight residency, WDM, PDPU, precision, electronic remainder, and calibration interval.",
            "blockers": "" if ablation_ready else "Architecture ablation table missing",
            "promotion_requirements": "Simulator-backed latency/energy and hardware activity validation for stronger quantitative claims.",
        },
        {
            "experiment": "P1-E8",
            "status": "ready_with_limitations" if scale_ready else "proxy",
            "claim_eligible": bool(scale),
            "evidence_tier": "local/modelled physical resource proxy" if scale_ready else "partial proxy",
            "trace_readiness": "not required",
            "literature_readiness": "literature-context optional",
            "layout_readiness": "proxy only; no layout closure",
            "meaning": "Scalability resource proxies include MRR, converter, memory-bandwidth, thermal, footprint, link-budget, and laser-power proxy fields.",
            "blockers": "No layout database, floorplan, thermal trace, or silicon validation.",
            "promotion_requirements": "Layout/floorplan model, link budget, thermal model, or simulator export tied to HPAT physical assumptions.",
        },
    ]


def _md(summary: dict[str, Any]) -> str:
    """把就绪度汇总渲染成 Markdown 表格文本。

    参数 summary：就绪度汇总字典。
    返回：Markdown 字符串。
    """
    lines = [
        "# P1 Readiness Summary",
        "",
        f"Schema: `{summary['schema_version']}`",
        f"Overall P1 ready with limitations: `{str(summary['overall_ready_with_limitations']).lower()}`",
        "",
        "| Experiment | Status | Claim eligible | Evidence tier | Trace | Literature | Layout |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in summary["experiments"]:
        lines.append(
            f"| {row['experiment']} | {row['status']} | {str(row['claim_eligible']).lower()} | "
            f"{row['evidence_tier']} | {row['trace_readiness']} | {row['literature_readiness']} | {row['layout_readiness']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "P1 artifacts support bounded architecture-level, local/modelled, trace-backed, or literature-context discussion only. They do not create simulator-calibrated physical-design energy, layout closure, fabricated-silicon evidence, edge deployment evidence, or measured HPAT speedup.",
            "",
            "## Blockers And Promotion Requirements",
            "",
        ]
    )
    for row in summary["experiments"]:
        lines.append(f"- {row['experiment']}: {row['blockers'] or 'no current blocker for bounded P1 use'} Promotion: {row['promotion_requirements']}")
    return "\n".join(lines) + "\n"


def run(output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行 P1 就绪度汇总主流程。

    参数 output_dir：结果输出目录。
    返回：产出文件路径字典（json / project_json / md / project_md / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    rows = _p1_rows(output_dir)
    # 整体就绪 = 每个实验都可主张（claim_eligible）
    summary = {
        "schema_version": P1_READINESS_SCHEMA_VERSION,
        "experiments": rows,
        "overall_ready_with_limitations": all(row["claim_eligible"] for row in rows),
        "claim_boundary": (
            "P1 artifacts remain local/modelled, trace-backed, or literature-context only; "
            "not silicon, not edge deployment, not layout closure, not measured HPAT speedup."
        ),
    }
    json_path = tables_dir / "p1_readiness_summary.json"
    project_json = REPO_ROOT / "tables" / "p1_readiness_summary.json"
    md_path = tables_dir / "p1_readiness_summary.md"
    project_md = REPO_ROOT / "tables" / "p1_readiness_summary.md"
    write_json(json_path, summary)
    write_json(project_json, summary)
    text = _md(summary)
    md_path.write_text(text, encoding="utf-8")
    write_text(project_md, text)

    manifest_path = output_dir / "p1_readiness_summary_manifest.json"
    manifest = base_manifest("p1_readiness_summary", "P1 readiness and claim-boundary gate")
    manifest.update(
        {
            "status": "ready_with_limitations" if summary["overall_ready_with_limitations"] else "partial",
            "p1_readiness_json": relative(json_path),
            "p1_readiness_json_sha256": sha256_file(json_path),
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
