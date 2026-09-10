"""能耗盈亏平衡分析（run_energy_break_even.py）。

实验目的：回答"光子处理器（HPAT）相对于电子基线，能耗上到底划不划算"。
通过改变关键能耗假设（如每组件能耗取不同值、与参考基准对比），找出
HPAT 能耗"打平"电子方案的临界区间（盈亏平衡区域），并排出哪些假设对
结果影响最敏感。本脚本只做确定性计算，不做随机实验。

- 输入：tables/hpat_energy_by_component*.csv（各组件能耗表）；可选
  --fig6-context-csv 提供论文 Fig.6 的参考上下文表。
- 产出（--output-dir 下）：
  * tables/energy_break_even_sweep.csv（盈亏平衡扫描结果）；
  * tables/energy_sensitivity_rank.csv（能耗假设敏感性排名）；
  * tables/energy_assumption_ledger.csv（能耗假设台账/来源清单）；
  * energy_break_even_manifest.json（运行清单）。
  同时会在开启 HPAT_WRITE_PROJECT_TABLES=1 时把结果提升到仓库 tables/ 目录。
- 命令：python run_energy_break_even.py --output-dir <目录> [--config <配置>]
  [--component-csv <文件>] [--fig6-context-csv <文件>]
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
from hpat_eval.e_local import (
    BREAK_EVEN_EVIDENCE_LABEL,
    E_LOCAL_CLAIM_BOUNDARY,
    ENERGY_ASSUMPTION_LEDGER_FIELDS,
    ENERGY_BREAK_EVEN_FIELDS,
    ENERGY_SENSITIVITY_RANK_FIELDS,
    energy_assumption_ledger_rows,
    energy_break_even_rows,
    energy_sensitivity_rank_rows,
)


def _first_existing(paths: list[pathlib.Path]) -> pathlib.Path:
    """从候选路径列表中返回第一个"存在且非空"的路径。

    若全部不存在，则返回列表最后一项（后续会因文件缺失而走阻塞分支）。
    用途：能耗表可能由不同上游脚本产生（trace-driven 或普通版），按优先级挑选。
    """
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return paths[-1]


def run(
    output_dir: pathlib.Path,
    config_path: pathlib.Path,
    component_csv: pathlib.Path | None,
    fig6_context_csv: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    """执行盈亏平衡分析主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径（本函数只用它记录哈希，不依赖其内容）。
        component_csv：显式指定的组件能耗 CSV；为 None 时自动挑选默认文件。
        fig6_context_csv：可选的 Fig.6 参考上下文表，用于填充对比基准。
    返回：各产出文件的路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    _ = load_json(config_path)
    # 优先使用调用方显式传入的能耗表，否则自动找默认表
    source = component_csv or _first_existing(
        [
            REPO_ROOT / "tables" / "hpat_energy_by_component_trace_driven.csv",
            REPO_ROOT / "tables" / "hpat_energy_by_component.csv",
        ]
    )
    fig6_source = fig6_context_csv
    manifest_path = output_dir / "energy_break_even_manifest.json"
    manifest = base_manifest("energy_break_even", BREAK_EVEN_EVIDENCE_LABEL)
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "component_source": relative(source),
            "component_source_sha256": sha256_file(source),
            "fig6_context_source": relative(fig6_source) if fig6_source else "",
            "fig6_context_source_sha256": sha256_file(fig6_source) if fig6_source else None,
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short deterministic sensitivity sweep.",
        }
    )
    # 依赖的上游能耗表不存在 → 记录"阻塞"状态并报错，提醒先跑上游实验
    if not source.exists() or source.stat().st_size == 0:
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "No HPAT component energy CSV exists for break-even analysis.",
                "promotion_note": "Run trace-driven activity or HPAT energy accounting first.",
            }
        )
        write_json(manifest_path, manifest)
        raise ValueError(manifest["blocked_reason"])
    if fig6_source is not None and (not fig6_source.exists() or fig6_source.stat().st_size == 0):
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": "The explicitly supplied Fig. 6 context table is missing or empty.",
                "promotion_note": "Supply a valid bounded reference-context table or omit the option.",
            }
        )
        write_json(manifest_path, manifest)
        raise ValueError(manifest["blocked_reason"])

    component_rows = read_csv(source)
    fig6_rows = read_csv(fig6_source) if fig6_source else []
    # 三组核心计算都委托给 hpat_eval.e_local 库：
    # 盈亏平衡扫描 / 敏感性排名 / 假设台账（记录每个能耗假设的来源）
    break_even = energy_break_even_rows(component_rows, fig6_rows)
    sensitivity = energy_sensitivity_rank_rows(component_rows)
    ledger = energy_assumption_ledger_rows()

    # 结果同时写到"本次运行目录"和"仓库公共 tables 目录"
    # （写仓库公共目录受 project_writes_enabled 保护，见 _common.py）
    run_break_even = tables_dir / "energy_break_even_sweep.csv"
    run_sensitivity = tables_dir / "energy_sensitivity_rank.csv"
    run_ledger = tables_dir / "energy_assumption_ledger.csv"
    project_break_even = REPO_ROOT / "tables" / "energy_break_even_sweep.csv"
    project_sensitivity = REPO_ROOT / "tables" / "energy_sensitivity_rank.csv"
    project_ledger = REPO_ROOT / "tables" / "energy_assumption_ledger.csv"
    write_project = project_writes_enabled()
    write_csv(run_break_even, break_even, ENERGY_BREAK_EVEN_FIELDS)
    write_csv(project_break_even, break_even, ENERGY_BREAK_EVEN_FIELDS)
    write_csv(run_sensitivity, sensitivity, ENERGY_SENSITIVITY_RANK_FIELDS)
    write_csv(project_sensitivity, sensitivity, ENERGY_SENSITIVITY_RANK_FIELDS)
    write_csv(run_ledger, ledger, ENERGY_ASSUMPTION_LEDGER_FIELDS)
    write_csv(project_ledger, ledger, ENERGY_ASSUMPTION_LEDGER_FIELDS)

    manifest.update(
        {
            "status": "ready_with_limitations",
            "outputs": [
                relative(run_break_even),
                relative(run_sensitivity),
                relative(run_ledger),
            ]
            + (
                # 只有确实写了公共目录时才把公共路径也计入产出
                [relative(project_break_even), relative(project_sensitivity), relative(project_ledger)]
                if write_project
                else []
            ),
            "project_write_performed": write_project,
            "reference_context_available": bool(fig6_rows),
            "break_even_row_count": len(break_even),
            "sensitivity_row_count": len(sensitivity),
            "assumption_row_count": len(ledger),
            "promotion_note": (
                "Supports modelled assumption-region and sensitivity claims only. "
                "Reference comparisons are populated only when an explicit bounded context table is supplied."
            ),
        }
    )
    write_json(manifest_path, manifest)
    return {
        "break_even_csv": run_break_even,
        "project_break_even_csv": project_break_even,
        "sensitivity_csv": run_sensitivity,
        "project_sensitivity_csv": project_sensitivity,
        "assumption_ledger": run_ledger,
        "project_assumption_ledger": project_ledger,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数（含可选的 --component-csv / --fig6-context-csv）。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--component-csv", default="")
    parser.add_argument("--fig6-context-csv", default="")
    args = parser.parse_args()
    outputs = run(
        pathlib.Path(args.output_dir),
        pathlib.Path(args.config),
        pathlib.Path(args.component_csv) if args.component_csv else None,
        pathlib.Path(args.fig6_context_csv) if args.fig6_context_csv else None,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
