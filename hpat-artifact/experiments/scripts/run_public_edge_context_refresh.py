"""公共边缘上下文刷新（run_public_edge_context_refresh.py）。

实验目的：刷新/固化"公共边缘/移动基准"（如公开论文、官方基准）的
上下文数据，生成上下文表、来源台账与 Markdown，供论文引用作动机与
方法论背景。它属于 P2 档的文献上下文（literature-context），
不是 HPAT 的基线，绝不能用来计算 HPAT 加速比或优越性。

- 输入：无外部数据（公共上下文由 hpat_eval.p2 内置/维护）。
- 产出（--output-dir 下）：tables/public_edge_context.csv/.md、
  tables/public_edge_context_sources.csv、public_edge_context_refresh_manifest.json；
  并同步写入仓库 tables/。
- 命令：python run_public_edge_context_refresh.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, relative, write_csv, write_json, write_text
from hpat_eval.p2 import (
    P2_CLAIM_BOUNDARY,
    PUBLIC_EDGE_CONTEXT_FIELDS,
    PUBLIC_EDGE_CONTEXT_SOURCE_FIELDS,
    public_edge_context_markdown,
    public_edge_context_rows,
    public_edge_context_source_rows,
)


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行公共边缘上下文刷新主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径（仅记录用）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 从库中取上下文行与来源台账行（来源行记录每条的验证日期）
    rows = public_edge_context_rows()
    source_rows = public_edge_context_source_rows(rows[0]["verified_date"] if rows else None)

    context_csv = tables_dir / "public_edge_context.csv"
    sources_csv = tables_dir / "public_edge_context_sources.csv"
    context_md = tables_dir / "public_edge_context.md"
    project_sources_csv = REPO_ROOT / "tables" / "public_edge_context_sources.csv"
    project_context_md = REPO_ROOT / "tables" / "public_edge_context.md"
    write_csv(context_csv, rows, PUBLIC_EDGE_CONTEXT_FIELDS)
    write_csv(sources_csv, source_rows, PUBLIC_EDGE_CONTEXT_SOURCE_FIELDS)
    write_csv(project_sources_csv, source_rows, PUBLIC_EDGE_CONTEXT_SOURCE_FIELDS)
    text = public_edge_context_markdown(rows)
    context_md.write_text(text, encoding="utf-8")
    write_text(project_context_md, text)

    manifest_path = output_dir / "public_edge_context_refresh_manifest.json"
    manifest = base_manifest("public_edge_context_refresh", "literature-context P2 public edge/mobile context refresh")
    manifest.update(
        {
            "status": "context",
            "config": relative(config_path),
            "outputs": [
                relative(context_csv),
                relative(sources_csv),
                relative(context_md),
                relative(project_sources_csv),
                relative(project_context_md),
            ],
            "row_count": len(rows),
            "source_row_count": len(source_rows),
            "verified_date": rows[0]["verified_date"] if rows else "",
            "claim_boundary": P2_CLAIM_BOUNDARY,
            # 免责说明：公共行只是动机与方法论上下文，不可用于算 HPAT 加速比
            "promotion_note": "Public rows are motivation and methodology context only; they must not be used to compute HPAT speedup or superiority.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short static context-refresh run.",
        }
    )
    write_json(manifest_path, manifest)
    return {
        "context_csv": context_csv,
        "sources_csv": sources_csv,
        "context_md": context_md,
        "project_sources_csv": project_sources_csv,
        "project_context_md": project_context_md,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config))
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
