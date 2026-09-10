"""热调谐压力测试（run_thermal_tuning_stress.py）。

实验目的：光子矩阵乘法依赖 MRR 微环精确对准共振波长，而温度漂移会让
微环失谐，需要额外"热调谐"（thermal tuning）功耗来补偿。本脚本模拟
温度变化场景，给出热调谐能耗/功率的"压力上界"，评估热管理是否吃紧。
属 P2 档本地建模代理，不代表真实封装器件的热测量。

- 输入：hpat_experiment_config.json（含热/温度假设参数）。
- 产出（--output-dir 下）：tables/thermal_tuning_stress.csv、
  thermal_tuning_stress_manifest.json；并同步写入仓库 tables/。
- 命令：python run_thermal_tuning_stress.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.p2 import P2_CLAIM_BOUNDARY, THERMAL_STRESS_FIELDS, thermal_tuning_stress_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行热调谐压力测试主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.p2.thermal_tuning_stress_rows
    rows = thermal_tuning_stress_rows(config)

    tables_csv = tables_dir / "thermal_tuning_stress.csv"
    project_csv = REPO_ROOT / "tables" / "thermal_tuning_stress.csv"
    write_csv(tables_csv, rows, THERMAL_STRESS_FIELDS)
    write_csv(project_csv, rows, THERMAL_STRESS_FIELDS)

    manifest_path = output_dir / "thermal_tuning_stress_manifest.json"
    manifest = base_manifest("thermal_tuning_stress", "local/modelled P2 thermal tuning stress proxy")
    manifest.update(
        {
            "status": "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(tables_csv), relative(project_csv)],
            "row_count": len(rows),
            "claim_boundary": P2_CLAIM_BOUNDARY,
            # 免责说明：热行只是代理形式给出调谐压力上界，不隐含封装器件热测量/验证
            "promotion_note": "Thermal rows bound tuning stress as a proxy only; no packaged-device thermal measurement or validation is implied.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical/modelled thermal stress run.",
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
