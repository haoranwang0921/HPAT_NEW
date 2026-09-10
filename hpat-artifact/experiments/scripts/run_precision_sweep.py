"""精度扫描实验（run_precision_sweep.py）。

实验目的：扫描光子乘法器的数据位宽（如 4/8/16 bit 等量化精度），
看量化精度变化如何影响计算精度的代理指标。这里的"精度"是合成代理
（synthetic transfer proxy），不是 ImageNet 真实精度，也不是流片验证。

- 输入：hpat_experiment_config.json（含精度扫描参数）。
- 产出（--output-dir 下）：tables/precision_sweep.csv、
  precision_sweep_manifest.json；并同步写入仓库 tables/precision_sweep.csv。
- 命令：python run_precision_sweep.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.nonidealities import PRECISION_FIELDS, precision_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行精度扫描主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.nonidealities.precision_rows
    rows = precision_rows(config)
    out_csv = tables_dir / "precision_sweep.csv"
    project_csv = REPO_ROOT / "tables" / "precision_sweep.csv"
    write_csv(out_csv, rows, PRECISION_FIELDS)
    write_csv(project_csv, rows, PRECISION_FIELDS)
    manifest_path = output_dir / "precision_sweep_manifest.json"
    manifest = base_manifest("precision_sweep", "analytical/modelled precision sensitivity")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(out_csv), relative(project_csv)],
            "row_count": len(rows),
            # 免责说明：仅合成代理指标，不是 ImageNet 精度也不是硅片验证
            "promotion_note": "Synthetic transfer proxy only; not ImageNet top-1 accuracy and not silicon validation.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical sensitivity sweep.",
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
