"""刷新外部边缘上下文数据（离线静态刷新，不联网）。

本脚本做什么：
    论文背景里会引用一批"外部边缘/移动场景"的公开上下文（外部设备的
    推理压力、公开 benchmark 等）。本脚本从 hpat_eval 内置的公共数据源
    重新生成 5 张表 + 1 份边界说明：
      - external_edge_mobile_context.csv        外部边缘/移动上下文主表；
      - external_edge_context_verified.csv      已核验的外部上下文；
      - external_edge_context_sources.csv       外部来源清单（含来源类型）；
      - external_edge_context_claim_boundary.md 结论边界说明（Markdown）；
      - edge_mobile_baseline_status.csv         作者实测边缘/移动基线的状态。
    同时生成一份 manifest，声明这些数据只作动机/相关背景使用，不得用于
    计算 HPAT 加速比。

数据从哪来：
    hpat_eval.p2.public_edge_context_rows() / public_edge_context_source_rows()
    （库内预置的公开数据），以及 --config 指定的实验配置（本脚本仅记录其
    路径用于溯源，不读取其内容）。

产出什么：
    上述 5 张表与边界说明，同时写到本轮输出目录和仓库 tables/ 各一份，
    外加 external_edge_context_manifest.json。

怎么运行：
    python experiments/scripts/refresh_external_edge_context.py --output-dir <输出目录>
    本脚本不联网，可安全本地运行。
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, relative, write_csv, write_json, write_text
from hpat_eval.e_local import (
    EDGE_MOBILE_BASELINE_STATUS_FIELDS,
    E_LOCAL_CLAIM_BOUNDARY,
    EXTERNAL_CONTEXT_EVIDENCE_LABEL,
    EXTERNAL_EDGE_CONTEXT_FIELDS,
    EXTERNAL_EDGE_CONTEXT_SOURCE_FIELDS,
    EXTERNAL_EDGE_CONTEXT_VERIFIED_FIELDS,
)
from hpat_eval.p2 import public_edge_context_rows, public_edge_context_source_rows


def _filter(row: dict, fields: list[str]) -> dict:
    """按指定字段列表从一行数据里挑出关心的列（缺失列补空串）。

    参数：
        row: 原始数据字典。
        fields: 想保留的字段名列表。
    返回：
        dict：只含 fields 里出现的键；原行没有的键统一补 ""。
    """
    return {field: row.get(field, "") for field in fields}


def _boundary_markdown(rows: list[dict]) -> str:
    """生成结论边界说明的 Markdown 文本。

    参数：
        rows: 外部上下文行（需含 source/evidence_tier/hpat_use 三列）。
    返回：
        str：可直接写入 .md 文件的完整 Markdown 文本。
    """
    lines = [
        "# External Edge/Mobile Context Claim Boundary",
        "",
        "Status: E-local literature/official-benchmark context only.",
        "",
        "These rows contextualize mobile and edge inference pressure. They are not author-measured HPAT baselines and must not be used to compute HPAT speedup.",
        "",
        "Author-measured edge/mobile baseline status: blocked until real device logs from the authors' measurement setup are supplied. External public data cannot be relabeled as author-measured evidence.",
        "",
        "| Source | Evidence tier | HPAT use |",
        "|---|---|---|",
    ]
    # 逐行渲染成 Markdown 表格
    for row in rows:
        lines.append(f"| {row['source']} | {row['evidence_tier']} | {row['hpat_use']} |")
    lines.extend(["", f"Claim boundary: {E_LOCAL_CLAIM_BOUNDARY}", ""])
    return "\n".join(lines)


def _baseline_status_rows() -> list[dict[str, str]]:
    """生成"作者实测边缘/移动基线"的状态行。

    返回：
        list[dict]：两条固定状态——作者实测基线（当前因缺少真实设备日志而
        阻塞，不可用于加速比结论）与外部公开上下文（可作背景引用但同样
        不可用于加速比结论）。
    """
    return [
        {
            "lane": "author_measured_edge_mobile_baseline",
            "status": "blocked_no_author_device_logs",
            "evidence_tier": "not claimable",
            "claim_eligible_for_speedup": "false",
            "artifact": "",
            "safe_use": "No paper-facing author-measured edge/mobile baseline claim until raw device/runtime/precision logs are supplied.",
            "must_not_imply": "author-measured HPAT edge/mobile baseline, author-measured mobile/edge deployment, or measured HPAT speedup",
            "external_data_label": "none",
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        },
        {
            "lane": "external_public_edge_mobile_context",
            "status": "context_ready",
            "evidence_tier": "external_public_context",
            "claim_eligible_for_speedup": "false",
            "artifact": "tables/external_edge_mobile_context.csv",
            "safe_use": "Use for mobile/edge motivation, related-work context, and evaluation-methodology framing only.",
            "must_not_imply": "author-measured HPAT edge/mobile baseline, direct HPAT speedup comparison, or superiority over submitted edge systems",
            "external_data_label": "external_public_context; source-specific public benchmark/literature context",
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        },
    ]


def _verified_context_rows(rows: list[dict], source_rows: list[dict]) -> list[dict[str, str]]:
    """把外部上下文主表扩成"已核验上下文"表（补充来源类型与安全使用说明）。

    参数：
        rows: 外部上下文主表行（含 source_id、source_url_or_doi、
            verified_date、measurement_scope、device_platform、model_task 等列）。
        source_rows: 来源清单行，用于按 source_id 反查 source_type。
    返回：
        list[dict]：已核验上下文行，统一附带 external_data_label 与
        claim_boundary（声明是外部公开背景、非作者实测 HPAT 基线）。
    """
    # 先建 source_id -> source_type 的映射，避免每行都去遍历来源表
    source_type_by_id = {row.get("source_id", ""): row.get("source_type", "") for row in source_rows}
    verified: list[dict[str, str]] = []
    for row in rows:
        source_id = row.get("source_id", "")
        verified.append(
            {
                "source_id": source_id,
                "source_url": row.get("source_url_or_doi", ""),
                "verified_date": row.get("verified_date", ""),
                # 有来源类型用来源类型，否则退回该行的证据等级
                "source_type": source_type_by_id.get(source_id, row.get("evidence_tier", "")),
                "metric_scope": row.get("measurement_scope", ""),
                "device_platform": row.get("device_platform", ""),
                "task_scope": row.get("model_task", ""),
                "safe_use": "External context for mobile/edge motivation and methodology framing only.",
                "must_not_imply": (
                    "author-measured HPAT edge/mobile baseline, measured HPAT speedup, "
                    "or superiority over public edge systems"
                ),
                "external_data_label": "external_public_context; not author-measured HPAT baseline",
                "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            }
        )
    return verified


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """重新生成外部边缘上下文的所有产出表并写 manifest。

    参数：
        output_dir: 本轮产出的输出目录（内部再建 tables/ 子目录）。
        config_path: 实验配置路径（仅用于记录溯源，不读取）。
    返回：
        dict：10 个产出文件路径 + manifest 路径的字典（键见返回值结构）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    rows = public_edge_context_rows()
    # 用主表的核验日期去筛选对应的来源清单（同一批数据保持一致）
    verified_date = rows[0]["verified_date"] if rows else ""
    source_rows = public_edge_context_source_rows(verified_date)
    # 只保留各表约定好的字段列，保证表结构稳定
    rows = [_filter(row, EXTERNAL_EDGE_CONTEXT_FIELDS) for row in rows]
    source_rows = [_filter(row, EXTERNAL_EDGE_CONTEXT_SOURCE_FIELDS) for row in source_rows]
    verified_rows = _verified_context_rows(rows, source_rows)

    # 本轮输出目录内的 5 个文件
    context_csv = tables_dir / "external_edge_mobile_context.csv"
    verified_csv = tables_dir / "external_edge_context_verified.csv"
    sources_csv = tables_dir / "external_edge_context_sources.csv"
    boundary_md = tables_dir / "external_edge_context_claim_boundary.md"
    status_csv = tables_dir / "edge_mobile_baseline_status.csv"
    # 仓库正式表对应的 5 个文件（同步更新，供论文引用）
    project_context_csv = REPO_ROOT / "tables" / "external_edge_mobile_context.csv"
    project_verified_csv = REPO_ROOT / "tables" / "external_edge_context_verified.csv"
    project_sources_csv = REPO_ROOT / "tables" / "external_edge_context_sources.csv"
    project_boundary_md = REPO_ROOT / "tables" / "external_edge_context_claim_boundary.md"
    project_status_csv = REPO_ROOT / "tables" / "edge_mobile_baseline_status.csv"
    status_rows = _baseline_status_rows()
    # 五组"本轮产出 + 仓库正式表"各写一份
    write_csv(context_csv, rows, EXTERNAL_EDGE_CONTEXT_FIELDS)
    write_csv(project_context_csv, rows, EXTERNAL_EDGE_CONTEXT_FIELDS)
    write_csv(verified_csv, verified_rows, EXTERNAL_EDGE_CONTEXT_VERIFIED_FIELDS)
    write_csv(project_verified_csv, verified_rows, EXTERNAL_EDGE_CONTEXT_VERIFIED_FIELDS)
    write_csv(sources_csv, source_rows, EXTERNAL_EDGE_CONTEXT_SOURCE_FIELDS)
    write_csv(project_sources_csv, source_rows, EXTERNAL_EDGE_CONTEXT_SOURCE_FIELDS)
    write_csv(status_csv, status_rows, EDGE_MOBILE_BASELINE_STATUS_FIELDS)
    write_csv(project_status_csv, status_rows, EDGE_MOBILE_BASELINE_STATUS_FIELDS)
    text = _boundary_markdown(rows)
    boundary_md.write_text(text, encoding="utf-8")
    write_text(project_boundary_md, text)

    manifest_path = output_dir / "external_edge_context_manifest.json"
    manifest = base_manifest("external_edge_context", EXTERNAL_CONTEXT_EVIDENCE_LABEL)
    manifest.update(
        {
            "status": "context_ready",
            "config": relative(config_path),
            "outputs": [
                relative(context_csv),
                relative(project_context_csv),
                relative(verified_csv),
                relative(project_verified_csv),
                relative(sources_csv),
                relative(project_sources_csv),
                relative(boundary_md),
                relative(project_boundary_md),
                relative(status_csv),
                relative(project_status_csv),
            ],
            "row_count": len(rows),
            "verified_context_row_count": len(verified_rows),
            "source_row_count": len(source_rows),
            "baseline_status_rows": len(status_rows),
            "verified_date": verified_date,
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            "promotion_note": "Use only as mobile/edge motivation and methodology context; never compute HPAT speedup from these rows. Author-measured edge/mobile baseline remains blocked without real device logs.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short static context refresh.",
        }
    )
    write_json(manifest_path, manifest)
    return {
        "context_csv": context_csv,
        "project_context_csv": project_context_csv,
        "verified_csv": verified_csv,
        "project_verified_csv": project_verified_csv,
        "sources_csv": sources_csv,
        "project_sources_csv": project_sources_csv,
        "boundary_md": boundary_md,
        "project_boundary_md": project_boundary_md,
        "status_csv": status_csv,
        "project_status_csv": project_status_csv,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析 --output-dir / --config 并执行刷新。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config))
    # 以相对路径形式打印全部产出
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    # 直接执行本文件时调用命令行入口
    main()
