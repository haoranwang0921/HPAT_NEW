"""算子 profiler 闭合台账（run_operator_profiler_closure.py）。

实验目的：用 PyTorch profiler 对 MobileViT（合成输入）做一次 CPU 前向，
统计各类算子的调用次数与 CPU 自耗时，整理成"函数级覆盖台账"，
用于解释 HPAT 映射闭合边界与电子剩余部分。注意：profiler 的 CPU 时间
只是覆盖率说明，绝不能当作 HPAT 延迟/硅片时序证据。

- 输入：hpat_experiment_config.json（模型变体配置）；需 torch/timm。
- 产出（--output-dir 下）：tables/operator_fx_profiler_closure.csv/.md，
  并同步写入仓库 tables/；operator_fx_profiler_closure_manifest.json。
- 命令：python run_operator_profiler_closure.py --output-dir <目录>
  [--config <配置>] [--model-variant all|<变体>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from collections import defaultdict
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json, write_text
from hpat_eval.mobilevit_loader import variants_from_config


# 免责声明：profiler 闭合只是合成输入下的函数级覆盖台账，
# 不是实测 HPAT 运行时、不是边缘部署、也不是硅片验证。
CLAIM_BOUNDARY = (
    "Profiler closure is a synthetic-input function-level coverage ledger. "
    "It is not measured HPAT runtime, not edge deployment, and not silicon validation."
)

FIELDS = [
    "variant",
    "timm_model",
    "input_shape",
    "profiler_device",
    "op_group",
    "event_count",
    "total_call_count",
    "self_cpu_time_total_us",
    "coverage_role",
    "claim_use",
    "example_ops",
    "evidence_label",
    "claim_boundary",
]


def _available(name: str) -> bool:
    """判断某 Python 包是否可导入。"""
    return importlib.util.find_spec(name) is not None


def _classify_op(key: str) -> tuple[str, str, str]:
    """把 profiler 算子名分类成 (算子组, 覆盖角色, 主张用途)。

    规则：Linear/矩阵乘 → PDPU 候选；卷积/softmax/归一化/激活/形状/逐元素
    → 电子剩余或前端；其它 → 未映射剩余。
    """
    lower = key.lower()
    if any(token in lower for token in ["linear", "addmm", "matmul", "mm", "bmm"]):
        return (
            "linear_or_matrix_multiply",
            "pdpu_candidate_or_supporting_linear_function_visible",
            "coverage_boundary_only_not_measured_hpat_execution",
        )
    if "conv" in lower:
        return ("convolution", "electronic_remainder_or_frontend", "coverage_boundary_only")
    if "softmax" in lower:
        return ("softmax_attention_normalization", "electronic_remainder", "coverage_boundary_only")
    if "norm" in lower or "batch_norm" in lower or "layer_norm" in lower:
        return ("normalization", "electronic_remainder", "coverage_boundary_only")
    if any(token in lower for token in ["silu", "relu", "gelu", "sigmoid", "activation"]):
        return ("activation", "electronic_remainder", "coverage_boundary_only")
    if any(token in lower for token in ["view", "reshape", "permute", "transpose", "contiguous", "cat", "slice"]):
        return ("shape_or_data_movement", "electronic_or_control_remainder", "coverage_boundary_only")
    if any(token in lower for token in ["add", "mul", "sub", "div", "where"]):
        return ("elementwise", "electronic_or_control_remainder", "coverage_boundary_only")
    return ("other_profiler_visible", "unmapped_profiler_visible_remainder", "coverage_boundary_only")


def _profile_variant(torch: Any, timm: Any, variant: dict[str, Any]) -> list[dict[str, Any]]:
    """对单个模型变体做合成输入 profiler，返回按算子组聚合的行。

    参数：
        torch / timm：对应库模块。
        variant：模型变体配置（含 timm_model 与分辨率）。
    返回：该变体的 profiler 覆盖行列表。
    """
    resolution = int(variant["input_resolution"])
    model = timm.create_model(variant["timm_model"], pretrained=False).eval().to("cpu")
    inputs = torch.randn(1, 3, resolution, resolution)
    # 先 warm-up 一次，再正式 profiling（避免首次初始化开销计入统计）
    with torch.no_grad():
        _ = model(inputs)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False) as prof:
            _ = model(inputs)
    # 按算子组聚合事件数、调用次数、CPU 自耗时与示例算子名
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": 0, "calls": 0, "time": 0.0, "examples": []})
    for event in prof.key_averages():
        op_group, coverage_role, claim_use = _classify_op(event.key)
        bucket = grouped[op_group]
        bucket["events"] += 1
        bucket["calls"] += int(getattr(event, "count", 0))
        bucket["time"] += float(getattr(event, "self_cpu_time_total", 0.0))
        if len(bucket["examples"]) < 4:
            bucket["examples"].append(event.key)
        bucket["coverage_role"] = coverage_role
        bucket["claim_use"] = claim_use
    rows: list[dict[str, Any]] = []
    for op_group, bucket in sorted(grouped.items()):
        rows.append(
            {
                "variant": variant["variant"],
                "timm_model": variant["timm_model"],
                "input_shape": f"1x3x{resolution}x{resolution}",
                "profiler_device": "cpu_synthetic_input",
                "op_group": op_group,
                "event_count": bucket["events"],
                "total_call_count": bucket["calls"],
                "self_cpu_time_total_us": f"{bucket['time']:.4f}",
                "coverage_role": bucket["coverage_role"],
                "claim_use": bucket["claim_use"],
                "example_ops": "; ".join(bucket["examples"]),
                "evidence_label": "PyTorch profiler synthetic-input operator closure; coverage ledger only",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def _markdown(rows: list[dict[str, Any]]) -> str:
    """把 profiler 覆盖台账渲染成 Markdown 表格。"""
    lines = [
        "# Operator FX/Profiler Closure",
        "",
        CLAIM_BOUNDARY,
        "",
        "| Variant | Op group | Calls | Coverage role | Claim use |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {op_group} | {total_call_count} | {coverage_role} | {claim_use} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Use in paper: cite this ledger only to explain HPAT mapping closure and electronic remainder boundaries. "
            "Do not cite profiler CPU time as HPAT latency, edge latency, or silicon timing.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: pathlib.Path, config_path: pathlib.Path, model_variant: str) -> dict[str, pathlib.Path]:
    """执行 profiler 闭合台账主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
        model_variant：要跑的模型变体（"all" 表示全部）。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 依赖检查：缺 torch/timm 直接报错（本脚本无法降级）
    if not (_available("torch") and _available("timm")):
        missing = [name for name in ["torch", "timm"] if not _available(name)]
        raise RuntimeError(f"Missing Python packages for operator profiler closure: {', '.join(missing)}")
    import timm  # type: ignore
    import torch  # type: ignore

    config = load_json(config_path)
    variants = [row for row in variants_from_config(config) if model_variant == "all" or row["variant"] == model_variant]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.extend(_profile_variant(torch, timm, variant))

    csv_path = tables_dir / "operator_fx_profiler_closure.csv"
    project_csv = REPO_ROOT / "tables" / "operator_fx_profiler_closure.csv"
    md_path = tables_dir / "operator_fx_profiler_closure.md"
    project_md = REPO_ROOT / "tables" / "operator_fx_profiler_closure.md"
    write_csv(csv_path, rows, FIELDS)
    write_csv(project_csv, rows, FIELDS)
    text = _markdown(rows)
    write_text(md_path, text)
    write_text(project_md, text)

    manifest_path = output_dir / "operator_fx_profiler_closure_manifest.json"
    manifest = base_manifest("operator_fx_profiler_closure", "synthetic-input operator profiler closure ledger")
    manifest.update(
        {
            "status": "ready_with_limitations",
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(csv_path), relative(project_csv), relative(md_path), relative(project_md)],
            "row_count": len(rows),
            "claim_boundary": CLAIM_BOUNDARY,
            # 免责说明：只是覆盖台账，CPU profiler 时间不能当 HPAT 运行时证据
            "promotion_note": "Coverage ledger only; CPU profiler timing must not be used as HPAT runtime evidence.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; one synthetic profiler pass per model is short.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": csv_path, "project_csv": project_csv, "md": md_path, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--model-variant", default="all")
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config), args.model_variant)
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
