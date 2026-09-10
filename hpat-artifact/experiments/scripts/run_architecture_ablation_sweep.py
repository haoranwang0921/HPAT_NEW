"""架构消融扫描（run_architecture_ablation_sweep.py）。

实验目的：对 HPAT 的架构设计做"消融"（ablation，即逐项去掉/改变某个
设计特性，观察影响）：例如权重常驻 vs 流式、是否启用广播、波长数、
PDPU 存储体数、tile 数、位宽、是否计入电子部分、校准间隔等。用来
回答"每个架构特性对能耗/性能的贡献有多大"。属 P1 档本地建模敏感性。

- 输入：hpat_experiment_config.json（含架构基线参数）。
- 产出（--output-dir 下）：tables/hpat_architecture_ablation.csv、
  architecture_ablation_manifest.json；并同步写入仓库 tables/。
- 命令：python run_architecture_ablation_sweep.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.p1 import ARCHITECTURE_ABLATION_FIELDS, P1_CLAIM_BOUNDARY, architecture_ablation_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行架构消融扫描主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.p1.architecture_ablation_rows
    rows = architecture_ablation_rows(config)

    tables_csv = tables_dir / "hpat_architecture_ablation.csv"
    project_csv = REPO_ROOT / "tables" / "hpat_architecture_ablation.csv"
    write_csv(tables_csv, rows, ARCHITECTURE_ABLATION_FIELDS)
    write_csv(project_csv, rows, ARCHITECTURE_ABLATION_FIELDS)

    manifest_path = output_dir / "architecture_ablation_manifest.json"
    manifest = base_manifest("architecture_ablation_sweep", "local/modelled P1 architecture ablation")
    manifest.update(
        {
            "status": "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(tables_csv), relative(project_csv)],
            "row_count": len(rows),
            # 记录消融实验对照的架构基线配置，便于复现
            "baseline": {
                "weight_mode": "resident",
                "broadcast_enabled": True,
                "wavelengths": 16,
                "pdpu_banks": 2,
                "tiles": 2,
                "bit_width": 8,
                "electronic_remainder_included": True,
                "calibration_interval_inferences": 1000,
            },
            "claim_boundary": P1_CLAIM_BOUNDARY,
            # 免责说明：消融趋势是确定性的模型/代理敏感性，不是实测硬件结果
            "promotion_note": "Ablation trends are deterministic model/proxy sensitivities, not measured hardware results.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled ablation run.",
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
