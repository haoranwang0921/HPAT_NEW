"""能耗核算实验（run_energy_accounting.py）。

实验目的：把 HPAT 光子处理器的总能耗"分解到各硬件组件"（如光源、
O/E 光电转换、热调谐等），并与配置文件中声明的能耗预算（energy envelope）
互相核对，同时做"压力场景"敏感性分析（例如某组件能耗上浮 25% 会怎样）。

- 输入：tables/energy_breakdown.csv（各组件能耗明细表）、
  hpat_experiment_config.json（含能耗预算 energy_envelope_mj）。
- 产出（写入 --output-dir 下）：
  * energy_accounting_summary.csv：各组件分模型（XXS/XS/S）能耗及占比；
  * energy_stress_scenarios.csv：加压场景下的总能耗变化；
  * energy_accounting_manifest.json：运行清单（输入文件哈希、核对结果等）。
- 命令：python run_energy_accounting.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, read_csv, relative, sha256_file, write_csv, write_json


# 能耗明细表的固定位置（位于仓库公共 tables 目录）
ENERGY_CSV = REPO_ROOT / "tables" / "energy_breakdown.csv"


def _f(row: dict[str, str], key: str) -> float:
    """把 CSV 行中的某个字符串字段转成浮点数，方便做数值计算。"""
    return float(row[key])


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行能耗核算主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置文件路径。
    返回：三个产出文件的路径字典（summary / stress / manifest）。
    """
    ensure_dir(output_dir)
    config = load_json(config_path)
    rows = read_csv(ENERGY_CSV)

    summary_rows: list[dict[str, Any]] = []
    # 累计三个模型（MobileViT-XXS/XS/S）的组件能耗总和，用于与预算比对
    totals = {"MobileViT-XXS": 0.0, "MobileViT-XS": 0.0, "MobileViT-S": 0.0}
    for row in rows:
        xxs = _f(row, "xxs_mj")
        xs = _f(row, "xs_mj")
        s = _f(row, "s_mj")
        totals["MobileViT-XXS"] += xxs
        totals["MobileViT-XS"] += xs
        totals["MobileViT-S"] += s
        summary_rows.append(
            {
                "component_group": row["component_group"],
                "xxs_mj": f"{xxs:.6f}",
                "xs_mj": f"{xs:.6f}",
                "s_mj": f"{s:.6f}",
                # 各组件能耗占该模型总预算的比例（%）
                "xxs_share_percent": f"{100.0 * xxs / config['energy_envelope_mj']['MobileViT-XXS']:.4f}",
                "xs_share_percent": f"{100.0 * xs / config['energy_envelope_mj']['MobileViT-XS']:.4f}",
                "s_share_percent_recomputed": f"{100.0 * s / config['energy_envelope_mj']['MobileViT-S']:.4f}",
                "claim_status": row["claim_status"],
            }
        )

    # 逐模型核对"各组件加总"与"配置声明的总预算"是否一致（容差 0.01 mJ）
    checks = []
    for model, expected in config["energy_envelope_mj"].items():
        actual = totals[model]
        checks.append(
            {
                "model": model,
                "expected_mj": expected,
                "actual_sum_mj": actual,
                "abs_delta_mj": actual - expected,
                "within_0p01_mj": abs(actual - expected) <= 0.01,
            }
        )

    by_name = {row["component_group"]: row for row in rows}

    def scenario_total(model_key: str, multipliers: dict[str, float]) -> float:
        """按"加压系数表"重算某模型的总能耗。

        参数：
            model_key：模型名（MobileViT-XXS/XS/S 之一）。
            multipliers：{组件名: 放大倍数}，未列出的组件取 1.0。
        返回：放大后的组件能耗之和。
        """
        total = 0.0
        for r in rows:
            key = {"MobileViT-XXS": "xxs_mj", "MobileViT-XS": "xs_mj", "MobileViT-S": "s_mj"}[model_key]
            mult = multipliers.get(r["component_group"], 1.0)
            total += float(r[key]) * mult
        return total

    # 定义几种"悲观加压"场景：某类组件能耗被放大，观察对总能耗的影响
    scenarios = {
        "base": {},
        "optical_source_plus_25pct": {"Optical source and passive loss budget": 1.25},
        "oe_readout_plus_25pct": {"O/E readout": 1.25},
        "converters_plus_25pct": {"DAC and input modulation": 1.25, "O/E readout": 1.25},
        "thermal_and_calibration_double": {"Thermal tuning/tracking": 2.0, "Amortized calibration": 2.0},
    }
    stress_rows: list[dict[str, Any]] = []
    for scenario, multipliers in scenarios.items():
        for model in ["MobileViT-XXS", "MobileViT-XS", "MobileViT-S"]:
            base = config["energy_envelope_mj"][model]
            total = scenario_total(model, multipliers)
            stress_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "total_mj": f"{total:.6f}",
                    "delta_vs_base_mj": f"{total - base:.6f}",
                    "delta_vs_base_percent": f"{100.0 * (total - base) / base:.4f}",
                    "evidence_label": "local/modelled sensitivity",
                }
            )

    summary_path = output_dir / "energy_accounting_summary.csv"
    stress_path = output_dir / "energy_stress_scenarios.csv"
    manifest_path = output_dir / "energy_accounting_manifest.json"

    write_csv(
        summary_path,
        summary_rows,
        [
            "component_group",
            "xxs_mj",
            "xs_mj",
            "s_mj",
            "xxs_share_percent",
            "xs_share_percent",
            "s_share_percent_recomputed",
            "claim_status",
        ],
    )
    write_csv(
        stress_path,
        stress_rows,
        ["scenario", "model", "total_mj", "delta_vs_base_mj", "delta_vs_base_percent", "evidence_label"],
    )

    # 生成清单：记录输入文件、配置及其哈希，以及核对结果是否全部通过
    manifest = base_manifest("energy_accounting", "local/modelled consistency and sensitivity")
    manifest.update(
        {
            "input_csv": relative(ENERGY_CSV),
            "input_csv_sha256": sha256_file(ENERGY_CSV),
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "checks": checks,
            "all_envelope_checks_passed": all(c["within_0p01_mj"] for c in checks),
            "outputs": [relative(summary_path), relative(stress_path)],
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical smoke run.",
        }
    )
    write_json(manifest_path, manifest)
    return {"summary": summary_path, "stress": stress_path, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析 --output-dir / --config 并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config))
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
