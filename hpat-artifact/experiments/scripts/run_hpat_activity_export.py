"""HPAT 活动导出与能耗核算（run_hpat_activity_export.py）。

实验目的：把"算子活动"（每个算子执行了多少次乘加、读写多少数据）
换算成 HPAT 光子处理器的能耗，按层和按硬件组件分别出表。
活动来源有两种：
  (1) 显式传入 --activity-csv：HPAT 模拟器导出的活动计数（更可信，
      但要求单位能耗参数完整，否则 blocked）；
  (2) 否则用本地 torch/timm hook 跟踪或配置代理生成算子行。
本脚本是能耗核算（energy accounting）链路的枢纽，下游 break-even、
单位能耗标定等都消费它产出的 hpat_energy_by_component.csv。

- 输入：hpat_experiment_config.json、可选 --activity-csv（模拟器导出）、
  --operator-activity-csv（算子活动表）、--unit-costs-json（单位能耗 JSON）。
- 产出（--output-dir 下）：hpat_activity_by_layer.csv、
  hpat_energy_by_component.csv、hpat_energy_by_layer.csv、tables/ 下同份、
  hpat_energy_manifest.json；并同步写入仓库 tables/。
- 命令：python run_hpat_activity_export.py --output-dir <目录>
  [--config <配置>] [--activity-csv <文件>] [--operator-activity-csv <文件>]
  [--unit-costs-json <文件>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, read_csv, relative, sha256_file, write_csv, write_json
from hpat_eval.activity_trace import HPAT_ACTIVITY_FIELDS, activity_from_operator_rows, proxy_operator_rows
from hpat_eval.energy_model import ENERGY_COMPONENT_FIELDS, PER_LAYER_ENERGY_FIELDS, component_energy_rows, resolve_unit_costs
from hpat_eval.schemas import normalize_hpat_activity_rows


def _load_operator_rows(path: pathlib.Path, config: dict) -> tuple[list[dict], str]:
    """加载算子活动行：优先读指定 CSV，否则用配置生成代理算子行。

    参数：
        path：算子活动 CSV 路径。
        config：实验配置（用于生成代理行）。
    返回：(算子行列表, 来源描述字符串)。
    """
    if path.exists():
        return read_csv(path), "operator activity CSV"
    # 兜底：配置文件驱动生成的代理算子行（backend 标记为 activity-export proxy）
    return proxy_operator_rows(config, backend="activity-export proxy", precision="fp32"), "generated config proxy"


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    activity_csv: pathlib.Path | None,
    operator_activity_csv: pathlib.Path | None,
    unit_costs_json: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    """执行 HPAT 活动导出主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        activity_csv：可选，HPAT 模拟器导出的活动表。
        operator_activity_csv：可选，torch hook 算子活动表。
        unit_costs_json：可选，单位能耗 JSON（覆盖配置里的默认值）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    tables_dir = ensure_dir(output_dir / "tables")
    config = load_json(config_path)
    # 解析单位能耗（unit cost）：传入 JSON 则优先，否则用配置内置值
    unit_cost_payload: dict[str, Any] | None = None
    if unit_costs_json:
        unit_cost_payload = load_json(unit_costs_json)
    unit_costs, unit_cost_meta = resolve_unit_costs(config, unit_cost_payload)
    manifest = base_manifest("hpat_activity_export", "model/config-derived or simulator-exported activity")
    manifest_path = output_dir / "hpat_energy_manifest.json"
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "unit_costs_json": relative(unit_costs_json) if unit_costs_json else "",
            "unit_costs_json_sha256": sha256_file(unit_costs_json) if unit_costs_json else None,
            "unit_costs": unit_cost_meta,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; activity export path is a short parsing/modelled accounting run.",
        }
    )

    # 分支一：提供了模拟器导出的活动表（最高可信路径）
    if activity_csv and activity_csv.exists():
        try:
            # 先做 schema 归一化校验：格式不对会抛 ValueError
            activity_rows = normalize_hpat_activity_rows(read_csv(activity_csv))
        except ValueError as exc:
            # 格式非法 → 记录 blocked 状态并重新抛出，终止本次实验
            manifest.update(
                {
                    "status": "blocked",
                    "evidence_label": "blocked invalid simulator activity export",
                    "activity_source": "simulator_export",
                    "activity_source_path": relative(activity_csv),
                    "activity_source_sha256": sha256_file(activity_csv),
                    "blocked_reason": str(exc),
                    "promotion_note": "Fix the simulator export schema before using it for calibrated energy accounting.",
                }
            )
            write_json(manifest_path, manifest)
            raise
        # 用模拟器导出做"标定能耗"要求单位能耗参数及其出处元数据完整
        if not unit_cost_meta["values_complete"] or not unit_cost_meta["metadata_complete"]:
            missing_parts = []
            if not unit_cost_meta["values_complete"]:
                missing_parts.append("missing explicit unit-cost keys: " + ", ".join(unit_cost_meta["missing_explicit_keys"]))
            if not unit_cost_meta["metadata_complete"]:
                formatted = [
                    f"{key}({','.join(fields)})"
                    for key, fields in unit_cost_meta["missing_metadata"].items()
                ]
                missing_parts.append("missing unit-cost metadata: " + "; ".join(formatted))
            manifest.update(
                {
                    "status": "blocked",
                    "evidence_label": "blocked incomplete unit-cost parameters or provenance metadata",
                    "activity_source": "simulator_export",
                    "activity_source_path": relative(activity_csv),
                    "activity_source_sha256": sha256_file(activity_csv),
                    "blocked_reason": " | ".join(missing_parts),
                    "promotion_note": "Provide every required unit-cost value, unit, source, technology assumption, precision, and scope note before calibrated energy accounting.",
                }
            )
            write_json(manifest_path, manifest)
            raise ValueError(manifest["blocked_reason"])
        # 校验通过：保留一份模拟器活动原始副本，后续计算"不需要归一化"
        raw_activity_out = raw_dir / "hpat_simulator_activity.csv"
        write_csv(raw_activity_out, activity_rows, HPAT_ACTIVITY_FIELDS)
        source_label = "modelled from simulator activity counts and explicit unit costs"
        normalize = False
        activity_source = "simulator_export"
        evidence_tier = "local/modelled simulator-activity energy"
        unit_cost_status = "complete_metadata_claim_eligible"
        manifest.update(
            {
                "status": "ok",
                "evidence_label": source_label,
                "activity_source": activity_source,
                "activity_source_path": relative(activity_csv),
                "activity_source_sha256": sha256_file(activity_csv),
                "raw_activity_copy": relative(raw_activity_out),
                "promotion_note": (
                    "Energy may be described as modelled from simulator activity counts and explicit unit costs. "
                    "It is still not silicon-measured energy."
                ),
            }
        )
    # 分支二：无模拟器导出 → 用 torch hook 算子活动或配置代理
    else:
        op_csv = operator_activity_csv or output_dir / "tables" / "mobilevit_operator_activity.csv"
        op_rows, source = _load_operator_rows(op_csv, config)
        # 把算子活动换算成 HPAT 层活动
        activity_rows = activity_from_operator_rows(op_rows, config)
        source_label = (
            "local torch/timm hook trace-derived HPAT activity proxy"
            if any(row.get("trace_source") == "torch_hooks" for row in op_rows)
            else "model/config-derived activity proxy"
        )
        normalize = True  # 代理路径需要把能耗归一化到配置的总预算
        activity_source = "proxy_hook_trace"
        evidence_tier = "local/modelled proxy"
        unit_cost_status = "not_claim_calibrated"
        manifest.update(
            {
                "status": "proxy",
                "evidence_label": source_label + "; normalized/modelled",
                "activity_source": activity_source,
                "activity_proxy_source": source,
                "blocked_reason": "No HPAT simulator/exported activity CSV was provided.",
                "promotion_note": "Energy remains normalized/modelled until HPAT simulator activity counts are supplied.",
            }
        )

    # 统一核算：按组件（光源/O-E/DAC…）和按层分别算能耗
    component_rows, per_layer_rows = component_energy_rows(
        activity_rows,
        config,
        normalize_to_envelope=normalize,
        source_label=source_label,
        unit_costs=unit_costs,
        activity_source=activity_source,
        unit_cost_status=unit_cost_status,
        evidence_tier=evidence_tier,
    )
    activity_out = output_dir / "hpat_activity_by_layer.csv"
    component_out = output_dir / "hpat_energy_by_component.csv"
    layer_energy_out = output_dir / "hpat_energy_by_layer.csv"
    tables_component_out = tables_dir / "hpat_energy_by_component.csv"
    tables_layer_energy_out = tables_dir / "hpat_energy_by_layer.csv"
    project_component_out = REPO_ROOT / "tables" / "hpat_energy_by_component.csv"
    project_layer_energy_out = REPO_ROOT / "tables" / "hpat_energy_by_layer.csv"
    write_csv(activity_out, activity_rows, HPAT_ACTIVITY_FIELDS)
    write_csv(component_out, component_rows, ENERGY_COMPONENT_FIELDS)
    write_csv(layer_energy_out, per_layer_rows, PER_LAYER_ENERGY_FIELDS)
    write_csv(tables_component_out, component_rows, ENERGY_COMPONENT_FIELDS)
    write_csv(tables_layer_energy_out, per_layer_rows, PER_LAYER_ENERGY_FIELDS)
    write_csv(project_component_out, component_rows, ENERGY_COMPONENT_FIELDS)
    write_csv(project_layer_energy_out, per_layer_rows, PER_LAYER_ENERGY_FIELDS)
    calibrated_out = None
    # 模拟器导出路径（未归一化）额外产出一份"标定"组件能耗表
    if not normalize:
        calibrated_out = REPO_ROOT / "tables" / "hpat_energy_by_component_calibrated.csv"
        write_csv(calibrated_out, component_rows, ENERGY_COMPONENT_FIELDS)
    output_list = [
        relative(activity_out),
        relative(component_out),
        relative(layer_energy_out),
        relative(tables_component_out),
        relative(tables_layer_energy_out),
        relative(project_component_out),
        relative(project_layer_energy_out),
    ]
    if calibrated_out:
        output_list.append(relative(calibrated_out))
    manifest.update(
        {
            "normalize_to_energy_envelope": normalize,
            "outputs": output_list,
            "activity_row_count": len(activity_rows),
            "component_row_count": len(component_rows),
            "unit_cost_status": unit_cost_status,
        }
    )
    write_json(manifest_path, manifest)
    return {
        "activity_csv": activity_out,
        "component_csv": component_out,
        "project_component_csv": project_component_out,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数（含三个可选输入）并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--activity-csv", default="")
    parser.add_argument("--operator-activity-csv", default="")
    parser.add_argument("--unit-costs-json", default="")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.activity_csv) if args.activity_csv else None,
        pathlib.Path(args.operator_activity_csv) if args.operator_activity_csv else None,
        pathlib.Path(args.unit_costs_json) if args.unit_costs_json else None,
    )
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
