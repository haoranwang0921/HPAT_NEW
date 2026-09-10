"""E-local 就绪度汇总（run_e_local_readiness_summary.py）。

实验目的：把"边缘本地推理"（e-local，P1 档真实硬件相关）各条证据线
（lane，如 trace-driven 活动、盈亏平衡、算子映射闭合、外部边缘上下文等）
的状态汇总成一张就绪度表：哪些已"带局限就绪"（ready_with_limitations）、
哪些被阻塞（blocked），并生成 CSV/JSON/Markdown 三种格式的汇总报告。
这是 ASP-DAC 投稿时用于加强论证的门禁检查之一。

- 输入：仓库现有各 e-local 实验的产出（由 e_local_readiness_rows 自动检测）。
- 产出（--output-dir 下）：tables/e_local_readiness_summary.csv/.json/.md，
  并同步写入仓库 tables/；e_local_readiness_summary_manifest.json。
- 命令：python run_e_local_readiness_summary.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, relative, write_csv, write_json, write_text
from hpat_eval.e_local import (
    E_LOCAL_CLAIM_BOUNDARY,
    E_LOCAL_READINESS_FIELDS,
    e_local_readiness_markdown,
    e_local_readiness_rows,
)


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行 E-local 就绪度汇总主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径（用于清单记录）。
    返回：各产出文件的路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 自动检测各条证据线的状态（由库函数扫描仓库与本次运行目录）
    rows = e_local_readiness_rows(REPO_ROOT, output_dir)
    # 汇总判定：只要没有"失败"（都是就绪或阻塞）就算整体"带局限就绪"
    overall_ready_with_limitations = all(row["status"] in {"ready_with_limitations", "blocked"} for row in rows)
    ready_count = sum(row["status"] == "ready_with_limitations" for row in rows)

    csv_path = tables_dir / "e_local_readiness_summary.csv"
    project_csv = REPO_ROOT / "tables" / "e_local_readiness_summary.csv"
    json_path = tables_dir / "e_local_readiness_summary.json"
    project_json = REPO_ROOT / "tables" / "e_local_readiness_summary.json"
    md_path = tables_dir / "e_local_readiness_summary.md"
    project_md = REPO_ROOT / "tables" / "e_local_readiness_summary.md"
    write_csv(csv_path, rows, E_LOCAL_READINESS_FIELDS)
    write_csv(project_csv, rows, E_LOCAL_READINESS_FIELDS)
    payload = {
        "schema_version": "e-local-readiness-v1",
        "overall_ready_with_limitations": overall_ready_with_limitations,
        "ready_lane_count": ready_count,
        "lane_count": len(rows),
        "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
        "lanes": rows,
    }
    write_json(json_path, payload)
    write_json(project_json, payload)
    # Markdown 报告便于直接在论文/仓库 README 中引用
    text = e_local_readiness_markdown(rows)
    md_path.write_text(text, encoding="utf-8")
    write_text(project_md, text)

    manifest_path = output_dir / "e_local_readiness_summary_manifest.json"
    manifest = base_manifest("e_local_readiness_summary", "E-local readiness gate for ASP-DAC strengthening")
    manifest.update(
        {
            "status": "ready_with_limitations" if ready_count else "blocked",
            "config": relative(config_path),
            "outputs": [
                relative(csv_path),
                relative(project_csv),
                relative(json_path),
                relative(project_json),
                relative(md_path),
                relative(project_md),
            ],
            "overall_ready_with_limitations": overall_ready_with_limitations,
            "ready_lane_count": ready_count,
            "lane_count": len(rows),
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            # 免责说明：e-local 就绪度只补充、不取代 P0 无硅片就绪度
            "promotion_note": "E-local readiness complements, but does not override, P0 no-silicon readiness.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short readiness summarization run.",
        }
    )
    write_json(manifest_path, manifest)
    return {
        "csv": csv_path,
        "project_csv": project_csv,
        "json": json_path,
        "project_json": project_json,
        "md": md_path,
        "project_md": project_md,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config))
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
