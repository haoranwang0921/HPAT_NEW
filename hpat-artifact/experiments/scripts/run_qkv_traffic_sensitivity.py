"""注意力 QKV 流量敏感性（run_qkv_traffic_sensitivity.py）。

实验目的：用解析公式估算注意力层 Q/K/V 的数据传输量（traffic），
并扫描关键参数（token 数、head 数、位宽、权重流式/编程模式等），
观察流量随参数变化的敏感性。它是"分析型敏感性"实验，不依赖真实硬件。
与 run_qkv_traffic_calibration.py 的区别：前者做校准（尽量用真实轨迹），
本脚本做纯公式的敏感性扫描。

- 输入：hpat_experiment_config.json（含注意力相关参数）。
- 产出（--output-dir 下）：qkv_traffic_sensitivity.csv、tables/ 下同份、
  qkv_traffic_manifest.json；并同步写入仓库 tables/。
- 命令：python run_qkv_traffic_sensitivity.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.traffic import QKV_TRAFFIC_FIELDS, qkv_traffic_rows


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行 QKV 流量敏感性主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / project_csv / manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 核心计算委托给 hpat_eval.traffic.qkv_traffic_rows（纯解析公式）
    rows = qkv_traffic_rows(config)

    out_csv = output_dir / "qkv_traffic_sensitivity.csv"
    tables_csv = tables_dir / "qkv_traffic_sensitivity.csv"
    project_csv = REPO_ROOT / "tables" / "qkv_traffic_sensitivity.csv"
    manifest_path = output_dir / "qkv_traffic_manifest.json"
    write_csv(out_csv, rows, QKV_TRAFFIC_FIELDS)
    write_csv(tables_csv, rows, QKV_TRAFFIC_FIELDS)
    write_csv(project_csv, rows, QKV_TRAFFIC_FIELDS)
    manifest = base_manifest("qkv_traffic_sensitivity", "analytical sensitivity")
    manifest.update(
        {
            # 记录本实验使用的流量估算公式，便于论文核对
            "formula": "B_QKV = N*d*b_a + 3*N*d*b_o + optional weight streaming/programming + explicit bus term",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(out_csv), relative(tables_csv), relative(project_csv)],
            # 免责说明：论文级结论前需用最终模拟器/模型配置替换 token/head/位宽等参数
            "promotion_note": "Replace token counts, heads, bit widths, and programming mode with final simulator/model config before paper-facing quantitative claims.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical smoke run.",
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
