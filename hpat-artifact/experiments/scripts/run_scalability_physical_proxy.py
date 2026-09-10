"""规模扩展性-物理代理（run_scalability_physical_proxy.py）。

实验目的：用"物理资源代理指标"评估规模扩展压力——例如微环谐振器
（MRR，做光矩阵乘法的核心器件）数量、光电转换次数、激光功率分摊等，
看模型变大时物理资源是否吃紧。它属于 P1 档（边缘本地推理）的本地建模
代理，不代表真实布局/热/硅验证。
与 run_scalability_sweep.py 的区别：本脚本关注物理资源压力，前者关注
模型/架构规模参数的敏感性。

- 输入：hpat_experiment_config.json（含物理代理参数）。
- 产出（--output-dir 下）：tables/scalability_physical_proxy.csv、
  scalability_physical_proxy_manifest.json；并同步写入仓库 tables/。
- 命令：python run_scalability_physical_proxy.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.p1 import P1_CLAIM_BOUNDARY, SCALABILITY_PHYSICAL_PROXY_FIELDS, scalability_physical_proxy_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行规模扩展性物理代理主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.p1.scalability_physical_proxy_rows
    rows = scalability_physical_proxy_rows(config)

    tables_csv = tables_dir / "scalability_physical_proxy.csv"
    project_csv = REPO_ROOT / "tables" / "scalability_physical_proxy.csv"
    write_csv(tables_csv, rows, SCALABILITY_PHYSICAL_PROXY_FIELDS)
    write_csv(project_csv, rows, SCALABILITY_PHYSICAL_PROXY_FIELDS)

    manifest_path = output_dir / "scalability_physical_proxy_manifest.json"
    manifest = base_manifest("scalability_physical_proxy", "local/modelled P1 scalability physical proxy")
    manifest.update(
        {
            "status": "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(tables_csv), relative(project_csv)],
            "row_count": len(rows),
            "claim_boundary": P1_CLAIM_BOUNDARY,
            # 免责说明：物理代理只暴露资源压力，不等于版图/布局/热/硅验证
            "promotion_note": "Physical proxies expose resource pressure only; they are not floorplan, layout, thermal, or silicon validation.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled scalability proxy run.",
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
