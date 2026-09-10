"""版图面积可行性代理（run_layout_area_feasibility_proxy.py）。

实验目的：估算把 HPAT 光子矩阵乘法器（39 万 MRR 微环）放进目标芯片
面积是否可行，输出面积可行性代理数据。它属于 P2 档（更多设计细节）
的本地建模，只做面积估算，不隐含任何版图布线/DRC/LVS/物理设计闭合。

- 输入：hpat_experiment_config.json（含面积/器件尺寸等参数）。
- 产出（--output-dir 下）：tables/layout_area_feasibility_proxy.csv、
  layout_area_feasibility_proxy_manifest.json；并同步写入仓库 tables/。
- 命令：python run_layout_area_feasibility_proxy.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.p2 import LAYOUT_AREA_FIELDS, P2_CLAIM_BOUNDARY, layout_area_feasibility_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行版图面积可行性代理主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.p2.layout_area_feasibility_rows
    rows = layout_area_feasibility_rows(config)

    tables_csv = tables_dir / "layout_area_feasibility_proxy.csv"
    project_csv = REPO_ROOT / "tables" / "layout_area_feasibility_proxy.csv"
    write_csv(tables_csv, rows, LAYOUT_AREA_FIELDS)
    write_csv(project_csv, rows, LAYOUT_AREA_FIELDS)

    manifest_path = output_dir / "layout_area_feasibility_proxy_manifest.json"
    manifest = base_manifest("layout_area_feasibility_proxy", "local/modelled P2 layout-area feasibility proxy")
    manifest.update(
        {
            "status": "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(tables_csv), relative(project_csv)],
            "row_count": len(rows),
            "claim_boundary": P2_CLAIM_BOUNDARY,
            # 免责说明：面积只是代理估算，不隐含版图/布线/DRC/LVS/物理设计闭合
            "promotion_note": "Area values are proxy estimates only; no floorplan, routing, layout DRC/LVS, or physical-design closure is implied.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled layout-area proxy run.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": tables_csv, "project_csv": project_csv, "manifest": manifest_path}


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
