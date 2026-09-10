"""算子映射闭合台账（run_operator_mapping_closure.py）。

实验目的：从算子活动表出发，把 MobileViT 的各类算子整理成"映射闭合台账"，
明确每个算子组的覆盖角色和"能/不能主张什么"：
  - Linear hook 组（QKV 投影、注意力输出、FFN、分类头）：HPAT 映射扰动边界；
  - 注意力 score/value、逐点卷积：架构兼容映射场景（扩展场景）；
  - 空间/深度卷积：只算"全卷积 MAC 上限"（分析上限，未实现降维方案）；
  - 其余：电子剩余部分（Amdahl 上界的扣减项）。
输出 CSV 台账 + Markdown 说明，供论文论证"哪些算子真的被 HPAT 覆盖"。

- 输入：tables/mobilevit_operator_activity.csv（算子活动表）。
- 产出（--output-dir 下）：tables/operator_mapping_closure.csv/.md，
  并同步写入仓库 tables/；operator_mapping_closure_manifest.json。
- 命令：python run_operator_mapping_closure.py --output-dir <目录>
  [--operator-activity-csv <文件>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any

from _common import (
    REPO_ROOT,
    base_manifest,
    ensure_dir,
    project_writes_enabled,
    read_csv,
    relative,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)
from hpat_eval.e_local import E_LOCAL_CLAIM_BOUNDARY


OPERATOR_MAPPING_CLOSURE_FIELDS = [
    "variant",
    "op_group",
    "capture_mode",
    "capture_status",
    "total_trace_rows",
    "pdpu_candidate_rows",
    "linear_hook_rows",
    "matched_layer_count",
    "representative_layers",
    "coverage_role",
    "safe_use",
    "must_not_imply",
    "evidence_label",
    "claim_boundary",
]

# 属于"Linear hook 映射边界"的算子组（可用 torch hook 真实捕获）
LINEAR_HOOK_GROUPS = {
    "qkv_projection",
    "attention_output_projection",
    "ffn_linear",
    "classifier_linear",
}

# 注意力 score/value 与逐点卷积：靠模块形状推算的"扩展场景"组
ATTENTION_EXTENSION_GROUPS = {"attention_score", "attention_value"}
CONV_EXTENSION_GROUPS = {"pointwise_conv"}
# 空间/深度卷积：只做"全卷积 MAC 上限"分析（未实现真正的光子降维映射）
CONV_CEILING_GROUPS = {"spatial_conv", "depthwise_conv"}


def _join_layers(rows: list[dict[str, str]], limit: int = 6) -> str:
    """把层名去重排序后拼接成可读字符串，超过 limit 个用 +N more 截断。"""
    names = [row.get("layer_name", "") for row in rows if row.get("layer_name")]
    unique = sorted(dict.fromkeys(names))
    suffix = "" if len(unique) <= limit else f"; +{len(unique) - limit} more"
    return "; ".join(unique[:limit]) + suffix


def _row_for_group(variant: str, op_group: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """为某个 (模型, 算子组) 生成一行"映射闭合"记录。

    参数：
        variant：模型变体名。
        op_group：算子组名。
        rows：该组对应的算子活动行。
    返回：含捕获方式、状态、覆盖角色、安全用法与禁止主张的字典。
    """
    # 从行中筛出 PDPU 候选（能被光子点积单元处理的算子）与 Linear 模块行
    pdpu_rows = [row for row in rows if row.get("pdpu_candidate") in {"yes", "partial"}]
    linear_rows = [row for row in pdpu_rows if row.get("module_type") == "Linear" and row.get("layer_name")]
    # 按算子组类型给出捕获方式、状态、安全用法等说明
    if op_group in LINEAR_HOOK_GROUPS:
        capture_mode = "torch.nn.Linear forward hook"
        capture_status = "hpat_mapping_hook_captured" if linear_rows else "expected_linear_group_not_observed"
        safe_use = "Use as HPAT-mapped fixed-subset perturbation boundary when paired with the recorded hook manifest."
        must_not_imply = "Does not imply silicon execution, layout closure, edge/mobile deployment, or coverage of non-Linear function ops."
        coverage_role = "PDPU-candidate Linear module coverage"
    elif op_group in ATTENTION_EXTENSION_GROUPS:
        capture_mode = "attention module shape-derived function-op estimate"
        capture_status = "extension_shape_accounted" if rows else "expected_attention_group_not_observed"
        safe_use = "Use in the architecture-compatible mapping scenario and Amdahl design bound only."
        must_not_imply = "Must not be promoted to measured HPAT execution without a matching mapped implementation and timing/energy path."
        coverage_role = "attention extension scenario"
    elif op_group in CONV_EXTENSION_GROUPS:
        capture_mode = "torch.nn.Conv2d forward hook with 1x1 kernel classification"
        capture_status = "extension_module_captured" if rows else "expected_pointwise_group_not_observed"
        safe_use = "Use in the architecture-compatible mapping scenario after keeping conversion and feature traffic visible."
        must_not_imply = "Trace visibility is not measured PDPU execution or proof that lowering is energy-positive."
        coverage_role = "pointwise-convolution extension scenario"
    elif op_group in CONV_CEILING_GROUPS:
        capture_mode = "Conv2d shape/MAC trace with unimplemented analytical lowering"
        capture_status = "mac_ceiling_only" if rows else "expected_convolution_group_not_observed"
        safe_use = "Use only for the all-convolution MAC-coverage ceiling."
        must_not_imply = "Must not enter headline Amdahl results until direct/im2col lowering, buffering, conversion, and scheduling are costed."
        coverage_role = "analytical MAC ceiling"
    else:
        capture_mode = "electronic remainder trace group"
        capture_status = "electronic_remainder_or_non_pdpu_trace"
        safe_use = "Use to bound electronic remainder and Amdahl-style limits."
        must_not_imply = "Must not imply that the electronic remainder is accelerated by HPAT."
        coverage_role = "remainder accounting"
    return {
        "variant": variant,
        "op_group": op_group,
        "capture_mode": capture_mode,
        "capture_status": capture_status,
        "total_trace_rows": len(rows),
        "pdpu_candidate_rows": len(pdpu_rows),
        "linear_hook_rows": len(linear_rows),
        "matched_layer_count": len({row.get("layer_name", "") for row in linear_rows if row.get("layer_name")}),
        "representative_layers": _join_layers(linear_rows or rows),
        "coverage_role": coverage_role,
        "safe_use": safe_use,
        "must_not_imply": must_not_imply,
        "evidence_label": "local torch/timm operator mapping closure; hook/proxy boundary evidence",
        "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
    }


def _markdown(rows: list[dict[str, Any]], source_csv: pathlib.Path) -> str:
    """把映射闭合台账渲染成 Markdown 表格。"""
    lines = [
        "# HPAT Operator Mapping Closure",
        "",
        "This ledger separates the conservative Linear-hook perturbation boundary, the pointwise/attention extension scenario, and the all-convolution MAC ceiling.",
        "",
        f"- Source CSV: `{relative(source_csv)}`",
        "- Safe claim: HPAT fixed-subset robustness remains tied to the recorded Linear hook boundary; extension scenarios are mapping/Amdahl design bounds only.",
        "- Forbidden claim: the closure does not establish measured silicon execution, calibrated physical design, or real edge/mobile deployment.",
        "",
        "| Variant | Group | Status | Hook Rows | Safe Use |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {op_group} | {capture_status} | {linear_hook_rows} | {safe_use} |".format(**row)
        )
    lines.append("")
    lines.append("Attention score/value MACs are inferred from recorded module shapes; convolution ceiling rows remain unimplemented analytical lowering targets.")
    lines.append("")
    return "\n".join(lines)


def run(output_dir: pathlib.Path, operator_activity_csv: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行映射闭合台账主流程。

    参数：
        output_dir：结果输出目录。
        operator_activity_csv：算子活动表路径。
    返回：产出文件路径字典。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    # 按 (模型, 算子组) 对活动行分组
    source_rows = read_csv(operator_activity_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row.get("model", ""), row.get("op_group", ""))].append(row)
    variants = sorted({row.get("model", "") for row in source_rows if row.get("model")})
    op_groups = sorted({row.get("op_group", "") for row in source_rows if row.get("op_group")})
    # 组并集：数据里出现的组 + 所有扩展/上限组（保证每类都有账可查）
    all_groups = sorted(set(op_groups) | ATTENTION_EXTENSION_GROUPS | CONV_EXTENSION_GROUPS | CONV_CEILING_GROUPS)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for op_group in all_groups:
            rows.append(_row_for_group(variant, op_group, grouped.get((variant, op_group), [])))

    csv_path = tables_dir / "operator_mapping_closure.csv"
    project_csv = REPO_ROOT / "tables" / "operator_mapping_closure.csv"
    md_path = tables_dir / "operator_mapping_closure.md"
    project_md = REPO_ROOT / "tables" / "operator_mapping_closure.md"
    write_csv(csv_path, rows, OPERATOR_MAPPING_CLOSURE_FIELDS)
    md_text = _markdown(rows, operator_activity_csv)
    write_text(md_path, md_text)
    write_project = project_writes_enabled()
    if write_project:
        write_csv(project_csv, rows, OPERATOR_MAPPING_CLOSURE_FIELDS)
        write_text(project_md, md_text)

    output_paths = [csv_path, md_path]
    if write_project:
        output_paths.extend([project_csv, project_md])

    manifest_path = output_dir / "operator_mapping_closure_manifest.json"
    manifest = base_manifest("operator_mapping_closure", "local operator mapping closure ledger")
    manifest.update(
        {
            "status": "ready_with_limitations",
            "operator_activity_csv": relative(operator_activity_csv),
            "operator_activity_csv_sha256": sha256_file(operator_activity_csv),
            "outputs": [relative(path) for path in output_paths],
            "project_write_performed": write_project,
            "row_count": len(rows),
            "claim_boundary": E_LOCAL_CLAIM_BOUNDARY,
            # 免责说明：本台账只解释 hook 覆盖与电子/分析剩余，
            # 不要把注意力 score/value 等非 Linear 行当作实测 HPAT 执行
            "promotion_note": (
                "Use this ledger to explain hook coverage and electronic/analytical remainders. "
                "Do not promote non-Linear attention score/value rows to measured HPAT execution."
            ),
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short ledger derivation.",
        }
    )
    write_json(manifest_path, manifest)
    result = {"csv": csv_path, "markdown": md_path, "manifest": manifest_path}
    if write_project:
        result.update({"project_csv": project_csv, "project_markdown": project_md})
    return result


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--operator-activity-csv", default=str(REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"))
    args = parser.parse_args()
    _ = args.config
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.operator_activity_csv))
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
