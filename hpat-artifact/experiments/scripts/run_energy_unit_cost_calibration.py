"""单位能耗标定（run_energy_unit_cost_calibration.py）。

实验目的：把"每个硬件动作的单次能耗代价"（如一次光电转换、一次 DAC
调制各耗多少能量，即 unit cost）从 YAML 配置标定成表，并据此推算出
能耗不确定性扫描（energy uncertainty sweep），支撑 P1 档（边缘本地推理）
的能耗论断。本脚本不测量硬件，只是把假设显式化、可追溯。

- 输入：experiments/config/energy_unit_costs.yaml（单位能耗配置）、
  实验总配置 hpat_experiment_config.json、可选的上游组件能耗表。
- 产出（--output-dir 下）：
  * tables/energy_unit_costs.csv（单位能耗标定表）；
  * tables/energy_unit_cost_source_ledger.csv（每项代价的来源台账）；
  * tables/energy_uncertainty_sweep.csv / energy_uncertainty_summary.csv（若上游表存在）；
  * energy_unit_cost_calibration_manifest.json（运行清单）。
- 命令：python run_energy_unit_cost_calibration.py --output-dir <目录>
  [--config <配置>] [--unit-costs <YAML>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.evidence_strength import (
    ENERGY_UNCERTAINTY_FIELDS,
    ENERGY_UNCERTAINTY_SUMMARY_FIELDS,
    energy_uncertainty_rows,
    energy_uncertainty_summary_rows,
)
from hpat_eval.p1 import (
    P1_CLAIM_BOUNDARY,
    UNIT_COST_FIELDS,
    UNIT_COST_SOURCE_LEDGER_FIELDS,
    load_json_compatible_yaml,
    unit_cost_source_ledger_rows,
    uncertainty_grid_from_unit_cost_rows,
    unit_cost_rows,
)


def _component_rows(output_dir: pathlib.Path) -> tuple[list[dict[str, str]], pathlib.Path | None]:
    """在"本次运行目录"或"仓库公共目录"中找组件能耗表。

    参数 output_dir：本次运行的输出目录。
    返回：(表内容行列表, 表文件路径)；两处都找不到则返回 ([], None)。
    """
    for path in [
        output_dir / "tables" / "hpat_energy_by_component.csv",
        REPO_ROOT / "tables" / "hpat_energy_by_component.csv",
    ]:
        if path.exists() and path.stat().st_size > 0:
            return read_csv(path), path
    return [], None


def run(output_dir: pathlib.Path, config_path: pathlib.Path, unit_cost_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行单位能耗标定主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验总配置（含能耗预算等）。
        unit_cost_path：单位能耗 YAML 配置路径。
    返回：产出文件路径字典（含 csv、source_ledger、可选 uncertainty 系列、manifest）。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # YAML 配置转成 dict 后，拆成"标定行"与"来源台账行"
    payload = load_json_compatible_yaml(unit_cost_path)
    rows = unit_cost_rows(payload)
    ledger_rows = unit_cost_source_ledger_rows(payload)

    tables_csv = tables_dir / "energy_unit_costs.csv"
    project_csv = REPO_ROOT / "tables" / "energy_unit_costs.csv"
    ledger_csv = tables_dir / "energy_unit_cost_source_ledger.csv"
    project_ledger_csv = REPO_ROOT / "tables" / "energy_unit_cost_source_ledger.csv"
    write_csv(tables_csv, rows, UNIT_COST_FIELDS)
    write_csv(project_csv, rows, UNIT_COST_FIELDS)
    write_csv(ledger_csv, ledger_rows, UNIT_COST_SOURCE_LEDGER_FIELDS)
    write_csv(project_ledger_csv, ledger_rows, UNIT_COST_SOURCE_LEDGER_FIELDS)

    outputs: dict[str, pathlib.Path] = {
        "csv": tables_csv,
        "project_csv": project_csv,
        "source_ledger": ledger_csv,
        "project_source_ledger": project_ledger_csv,
    }
    # 若上游组件能耗表存在，则进一步做能耗不确定性扫描：
    # 用标定出的单位代价生成不确定性网格，套用到组件表上，汇总敏感性。
    component_rows, component_source = _component_rows(output_dir)
    if component_rows:
        p1_config = dict(config)
        p1_config["energy_uncertainty"] = uncertainty_grid_from_unit_cost_rows(rows, config)
        uncertainty = energy_uncertainty_rows(component_rows, p1_config)
        summary = energy_uncertainty_summary_rows(uncertainty)
        uncertainty_csv = tables_dir / "energy_uncertainty_sweep.csv"
        uncertainty_project = REPO_ROOT / "tables" / "energy_uncertainty_sweep.csv"
        summary_csv = tables_dir / "energy_uncertainty_summary.csv"
        summary_project = REPO_ROOT / "tables" / "energy_uncertainty_summary.csv"
        write_csv(uncertainty_csv, uncertainty, ENERGY_UNCERTAINTY_FIELDS)
        write_csv(uncertainty_project, uncertainty, ENERGY_UNCERTAINTY_FIELDS)
        write_csv(summary_csv, summary, ENERGY_UNCERTAINTY_SUMMARY_FIELDS)
        write_csv(summary_project, summary, ENERGY_UNCERTAINTY_SUMMARY_FIELDS)
        outputs.update(
            {
                "energy_uncertainty": uncertainty_csv,
                "project_energy_uncertainty": uncertainty_project,
                "energy_uncertainty_summary": summary_csv,
                "project_energy_uncertainty_summary": summary_project,
            }
        )

    manifest_path = output_dir / "energy_unit_cost_calibration_manifest.json"
    manifest = base_manifest("energy_unit_cost_calibration", "local/modelled P1 unit-cost calibration")
    manifest.update(
        {
            "status": "proxy",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "unit_costs": relative(unit_cost_path),
            "unit_costs_sha256": sha256_file(unit_cost_path),
            "component_source": relative(component_source) if component_source else "",
            "component_source_sha256": sha256_file(component_source) if component_source else None,
            "outputs": [relative(path) for path in outputs.values()],
            "row_count": len(rows),
            "source_ledger_row_count": len(ledger_rows),
            "uncertainty_refreshed": bool(component_rows),
            "claim_boundary": P1_CLAIM_BOUNDARY,
            "promotion_note": "Unit costs are explicit local model/proxy assumptions until replaced by cited or simulator-derived calibration.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short CSV/config calibration run.",
        }
    )
    write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument(
        "--unit-costs",
        default=str(REPO_ROOT / "experiments" / "config" / "energy_unit_costs.yaml"),
        help="JSON-compatible YAML unit-cost file.",
    )
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config), pathlib.Path(args.unit_costs))
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
