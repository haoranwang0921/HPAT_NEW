"""规模扩展性扫描（run_scalability_sweep.py）。

实验目的：扫描模型规模（如通道数、token 数等），看 HPAT 光子处理器的
算子映射、能耗/面积/时延等指标如何随规模扩展。这是"建模型"扩展性
敏感性分析，不涉及布局闭合或真实流片。
与 run_scalability_physical_proxy.py 的区别：本脚本扫模型/架构规模参数，
后者用物理资源代理指标（如微环数量、光电转换次数）评估扩展压力。

- 输入：hpat_experiment_config.json（含扩展性扫描参数）。
- 产出（--output-dir 下）：tables/scalability_sweep.csv、
  scalability_sweep_manifest.json；并同步写入仓库 tables/。
- 命令：python run_scalability_sweep.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.scalability import SCALABILITY_FIELDS, scalability_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行规模扩展性扫描主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.scalability.scalability_rows
    rows = scalability_rows(config)
    out_csv = tables_dir / "scalability_sweep.csv"
    project_csv = REPO_ROOT / "tables" / "scalability_sweep.csv"
    write_csv(out_csv, rows, SCALABILITY_FIELDS)
    write_csv(project_csv, rows, SCALABILITY_FIELDS)
    manifest_path = output_dir / "scalability_sweep_manifest.json"
    manifest = base_manifest("scalability_sweep", "modelled scalability sensitivity")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(out_csv), relative(project_csv)],
            "row_count": len(rows),
            # 免责说明：仅建模扩展性，不代表布局闭合/流片部署/热学验证
            "promotion_note": "Modelled scalability only; not layout closure, fabricated deployment, or physical thermal validation.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled sweep.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": out_csv, "project_csv": project_csv, "manifest": manifest_path}


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
