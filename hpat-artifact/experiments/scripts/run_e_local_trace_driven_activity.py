"""E-local trace-driven 活动与能耗核算（run_e_local_trace_driven_activity.py）。

实验目的：基于本地 torch/timm hook 跟踪得到的"算子活动"（run_mobilevit_
activity_trace.py 的产出），结合显式单位能耗配置，自底向上核算 HPAT
按层/按组件的能耗，形成 e-local（P1 档）的"trace 驱动"能耗证据线。
可选的 mapping_scenario 决定算子映射范围（只映射 Linear / 合理最大 /
最大全 MAC），产出文件名带对应后缀。

- 输入：tables/mobilevit_operator_activity.csv、energy_unit_costs.yaml、
  hpat_experiment_config.json。
- 产出（--output-dir 下）：raw/hpat_activity_by_layer<suffix>.csv、
  tables/hpat_energy_by_component<suffix>.csv、hpat_energy_by_layer<suffix>.csv、
  e_local_trace_driven_activity_manifest.json；开启项目写入时同步仓库 tables/。
- 命令：python run_e_local_trace_driven_activity.py --output-dir <目录>
  [--operator-activity-csv <文件>] [--mapping-scenario linear_only|reasonable_max|maximal_all_mac]
  [--unit-costs <YAML>] [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from _common import (
    REPO_ROOT,
    base_manifest,
    ensure_dir,
    load_json,
    project_writes_enabled,
    read_csv,
    relative,
    sha256_file,
    write_csv,
    write_json,
)
from hpat_eval.activity_trace import HPAT_ACTIVITY_FIELDS, activity_from_operator_rows
from hpat_eval.e_local import TRACE_DRIVEN_EVIDENCE_LABEL, E_LOCAL_CLAIM_BOUNDARY
from hpat_eval.energy_model import ENERGY_COMPONENT_FIELDS, PER_LAYER_ENERGY_FIELDS, component_energy_rows, resolve_unit_costs
from hpat_eval.p1 import load_json_compatible_yaml


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    operator_activity_csv: pathlib.Path,
    unit_costs_path: pathlib.Path,
    mapping_scenario: str = "linear_only",
) -> dict[str, pathlib.Path]:
    """执行 trace-driven 能耗核算主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        operator_activity_csv：torch hook 算子活动表。
        unit_costs_path：单位能耗 YAML。
        mapping_scenario：映射场景（linear_only / reasonable_max / maximal_all_mac）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    manifest_path = output_dir / "e_local_trace_driven_activity_manifest.json"
    manifest = base_manifest("e_local_trace_driven_activity", TRACE_DRIVEN_EVIDENCE_LABEL)
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "operator_activity_csv": relative(operator_activity_csv),
            "operator_activity_csv_sha256": sha256_file(operator_activity_csv),
            "unit_costs": relative(unit_costs_path),
            "unit_costs_sha256": sha256_file(unit_costs_path),
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            "mapping_scenario": mapping_scenario,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short trace-to-energy accounting run.",
        }
    )
    # 上游算子活动表缺失 → 阻塞并提示先跑 run_mobilevit_activity_trace.py
    if not operator_activity_csv.exists() or operator_activity_csv.stat().st_size == 0:
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "Operator activity CSV is missing; run_mobilevit_activity_trace.py first.",
                "promotion_note": "E-local trace-driven activity requires local torch/timm hook traces.",
            }
        )
        write_json(manifest_path, manifest)
        raise ValueError(manifest["blocked_reason"])

    # 解析单位能耗 → 把算子活动换算成 HPAT 活动 → 分别按层/按组件核算能耗
    unit_cost_payload = load_json_compatible_yaml(unit_costs_path)
    unit_costs, unit_cost_meta = resolve_unit_costs(config, unit_cost_payload)
    op_rows = read_csv(operator_activity_csv)
    activity_rows = activity_from_operator_rows(op_rows, config, mapping_scenario=mapping_scenario)
    component_rows, per_layer_rows = component_energy_rows(
        activity_rows,
        config,
        normalize_to_envelope=False,  # trace 驱动是"自底向上"核算，不做预算归一化
        source_label="local torch/timm trace-driven HPAT activity model",
        unit_costs=unit_costs,
        activity_source="trace_driven_operator_activity",
        unit_cost_status="explicit_modelled_unit_costs_with_metadata",
        evidence_tier="local trace-driven/modelled",
    )

    # 按映射场景选择文件名后缀，避免三种场景的结果互相覆盖
    suffix = (
        "_maximal_mapping"
        if mapping_scenario == "maximal_all_mac"
        else ("_reasonable_max_mapping" if mapping_scenario == "reasonable_max" else "_trace_driven")
    )
    raw_activity = raw_dir / f"hpat_activity_by_layer{suffix}.csv"
    run_component = tables_dir / f"hpat_energy_by_component{suffix}.csv"
    run_layer = tables_dir / f"hpat_energy_by_layer{suffix}.csv"
    project_activity = REPO_ROOT / "tables" / f"hpat_activity_by_layer{suffix}.csv"
    project_component = REPO_ROOT / "tables" / f"hpat_energy_by_component{suffix}.csv"
    project_layer = REPO_ROOT / "tables" / f"hpat_energy_by_layer{suffix}.csv"
    write_csv(raw_activity, activity_rows, HPAT_ACTIVITY_FIELDS)
    write_csv(run_component, component_rows, ENERGY_COMPONENT_FIELDS)
    write_csv(run_layer, per_layer_rows, PER_LAYER_ENERGY_FIELDS)
    write_project = project_writes_enabled()
    if write_project:
        write_csv(project_activity, activity_rows, HPAT_ACTIVITY_FIELDS)
        write_csv(project_component, component_rows, ENERGY_COMPONENT_FIELDS)
        write_csv(project_layer, per_layer_rows, PER_LAYER_ENERGY_FIELDS)

    output_paths = [raw_activity, run_component, run_layer]
    if write_project:
        output_paths.extend([project_activity, project_component, project_layer])

    manifest.update(
        {
            "status": "ready_with_limitations",
            "activity_source": "trace_driven_operator_activity",
            "normalize_to_energy_envelope": False,
            "unit_costs": unit_cost_meta,
            "outputs": [relative(path) for path in output_paths],
            "project_write_performed": write_project,
            "activity_row_count": len(activity_rows),
            "component_row_count": len(component_rows),
            "layer_energy_row_count": len(per_layer_rows),
            # 免责说明：支持自底向上的本地 trace 建模能耗，但仍非硅片/边缘部署证据
            "promotion_note": (
                "Supports bottom-up local trace-driven modelled energy accounting. "
                "It remains non-silicon, non-edge-deployment evidence."
            ),
        }
    )
    write_json(manifest_path, manifest)
    result = {
        "activity_csv": raw_activity,
        "component_csv": run_component,
        "layer_energy_csv": run_layer,
        "manifest": manifest_path,
    }
    if write_project:
        result.update(
            {
                "project_activity_csv": project_activity,
                "project_component_csv": project_component,
                "project_layer_energy_csv": project_layer,
            }
        )
    return result


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument(
        "--operator-activity-csv",
        default=str(REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"),
    )
    parser.add_argument("--mapping-scenario", choices=["linear_only", "reasonable_max", "maximal_all_mac"], default="linear_only")
    parser.add_argument(
        "--unit-costs",
        default=str(REPO_ROOT / "experiments" / "config" / "energy_unit_costs.yaml"),
    )
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.operator_activity_csv),
        pathlib.Path(args.unit_costs),
        args.mapping_scenario,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
