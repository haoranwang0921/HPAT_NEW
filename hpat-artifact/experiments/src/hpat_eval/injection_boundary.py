"""误差注入边界（injection boundary）：决定哪些层可以被"映射进光子端"。

背景：HPAT 是光子+电子混合架构，矩阵乘法等算子可以映射到光子的
PDPU（光点积单元，Photonic Dot-Product Unit）执行，但归一化、
softmax、残差、控制逻辑等算子只能留在电子端。所谓"边界"，就是
从算子清单（每行描述一个算子的归属候选）出发，划定：

  1) 候选层：理论上适合放上光子端的算子（如线性、注意力矩阵乘）；
  2) 匹配层：候选层中，模型里"真的存在"的模块（available_module_names）；
  3) 缺失层：候选但模型里找不到的模块（可能是命名差异或模型不含）；
  4) 覆盖率：匹配层数 / 候选层数（0~1，越高表示映射越完整）。

返回值是"边界报告"，供后续误差注入时知道：哪些层可以注入光子端误差、
哪些必须留在电子端。不同边界函数代表论文里的不同映射假设（保守 /
冒烟 / 最大 / 合理）。

易混淆点：本文件只回答"哪些层能上光子端"；具体注入什么误差、
怎么注入，见 nonidealities.py 与 joint_sim（本项目另一库）。
"""

from __future__ import annotations

from typing import Any


# 边界报告版本号：不同映射假设对应不同版本，方便下游追踪用了哪个边界
BOUNDARY_VERSION = "hpat-mapping-v1"        # HPAT 标准映射边界
MAXIMAL_BOUNDARY_VERSION = "maximal-all-mac-v1"  # 把所有 MAC（乘加）都映射的最大边界

# 理论上可以放到 PDPU 光子端执行的算子组（按模型结构分组）
PDPU_CANDIDATE_GROUPS = {
    "pointwise_projection",          # 逐点（1×1）投影
    "qkv_projection",                # Q/K/V 三个投影
    "attention_score",               # 注意力分数（QK^T）
    "attention_value",               # 注意力与 V 相乘（Attention@V）
    "attention_output_projection",   # 注意力输出投影
    "ffn_linear",                    # 前馈网络（FFN）的线性层
    "classifier_linear",             # 分类头线性层
}


def boundary_from_operator_rows(
    rows: list[dict[str, Any]],
    *,
    model_variant: str,
    available_module_names: set[str],
) -> dict[str, Any]:
    """从算子行清单生成"HPAT 标准映射"边界报告。

    :param rows: 算子清单行（每行有 model / pdpu_candidate / op_group /
                 execution_domain / layer_name 等字段）。
    :param model_variant: 目标模型变体名（只统计该模型的算子）。
    :param available_module_names: 模型中实际加载出来的模块名集合
        （来自 torch 模型遍历，如 "transformer.0.attention.q"）。
    :return: 边界报告字典（含覆盖率、匹配/缺失层名单等）。
    """
    # 候选层：PDPU 候选为 yes/partial 且算子组在候选名单里、且层名非空
    candidate_rows = [
        row
        for row in rows
        if row.get("model") == model_variant
        and row.get("pdpu_candidate") in {"yes", "partial"}
        and row.get("op_group") in PDPU_CANDIDATE_GROUPS
        and row.get("layer_name")
    ]
    # 电子端层：明确标记为"电子剩余"的层（留在电子端执行）
    electronic_rows = [
        row
        for row in rows
        if row.get("model") == model_variant
        and row.get("execution_domain") == "electronic remainder"
        and row.get("layer_name")
    ]
    allowlist = sorted({str(row["layer_name"]) for row in candidate_rows})  # 去重排序
    matched = sorted(name for name in allowlist if name in available_module_names)  # 模型里真有的
    missing = sorted(name for name in allowlist if name not in available_module_names)  # 模型里没有的
    candidate_groups = sorted({str(row.get("op_group", "")) for row in candidate_rows if row.get("op_group")})
    missing_groups = sorted(PDPU_CANDIDATE_GROUPS - set(candidate_groups))  # 候选名单里没有出现的组
    coverage = len(matched) / max(len(allowlist), 1)  # 覆盖率（分母至少为 1，防除零）
    return {
        "boundary_version": BOUNDARY_VERSION,
        "model_variant": model_variant,
        "mode": "hpat-mapping",
        "candidate_layer_count": len(allowlist),
        "matched_layer_count": len(matched),
        "missing_layer_count": len(missing),
        "coverage": coverage,
        "matched_layers": matched,
        "missing_layers": missing,
        "candidate_groups": candidate_groups,
        "missing_mapped_groups": missing_groups,
        "excluded_electronic_groups": sorted({str(row.get("op_group", "")) for row in electronic_rows if row.get("op_group")}),
    }


def all_linear_smoke_boundary(model_variant: str, available_module_names: set[str], linear_module_names: set[str]) -> dict[str, Any]:
    """冒烟测试（smoke test）边界：把所有 torch.nn.Linear 模块都当候选。

    用途：用于快速跑通整个映射+注入流程，不追求架构真实性，只验证
    管线本身没 bug。因此 candidate_layer_count 直接等于模型里 Linear
    模块的数量。

    :param model_variant: 模型变体名。
    :param available_module_names: 模型实际模块名集合。
    :param linear_module_names: 模型里所有 Linear 模块名集合。
    :return: 冒烟测试边界报告。
    """
    matched = sorted(name for name in linear_module_names if name in available_module_names)
    return {
        "boundary_version": "all-linear-smoke-v1",
        "model_variant": model_variant,
        "mode": "all-linear-smoke",
        "candidate_layer_count": len(matched),
        "matched_layer_count": len(matched),
        "missing_layer_count": 0,
        "coverage": 1.0 if matched else 0.0,  # 冒烟模式全部匹配，覆盖率为 1（空则 0）
        "matched_layers": matched,
        "missing_layers": [],
        "candidate_groups": ["all torch.nn.Linear modules"],
        "missing_mapped_groups": [],
        "excluded_electronic_groups": [],
    }


def maximal_all_mac_boundary(
    rows: list[dict[str, Any]],
    *,
    model_variant: str,
    available_module_names: set[str],
) -> dict[str, Any]:
    """Diagnostic output-perturbation boundary for the maximal mapped-MAC design.

    Leaf Linear/Conv2d modules are included directly. Function-level attention
    score/value rows are represented by one hook on their enclosing Attention
    module because QK^T and Attention-by-V are not standalone torch modules.
    """
    # （英文原注释）面向"最大映射"设计的诊断用输出扰动边界。
    # 通俗解释：这是"把一切能算 MAC 的层都塞进光子端"的极端边界，
    # 目的不是架构合理性，而是观测误差放大的上界。
    # 细节：叶级 Linear/Conv2d 模块直接纳入；注意力的 QK^T 和 Attention@V
    # 不是独立的 torch 模块，所以在算子行里按"所属 Attention 模块"挂一个钩子表示。
    maximal_groups = PDPU_CANDIDATE_GROUPS | {"pointwise_conv", "spatial_conv", "depthwise_conv"}  # 追加三类卷积
    candidates: set[str] = set()
    candidate_groups: set[str] = set()
    for row in rows:
        if row.get("model") != model_variant or row.get("op_group") not in maximal_groups:
            continue
        name = str(row.get("layer_name") or "")
        if not name:
            continue
        candidate_groups.add(str(row.get("op_group")))
        if row.get("op_group") in {"attention_score", "attention_value"}:
            # 注意力分数/值算子没有独立模块名，向上取到所属 Attention 模块名
            name = name.rsplit(".", 1)[0]
        candidates.add(name)
    matched = sorted(candidates & available_module_names)
    missing = sorted(candidates - available_module_names)
    return {
        "boundary_version": MAXIMAL_BOUNDARY_VERSION,
        "model_variant": model_variant,
        "mode": "maximal-all-mac",
        "candidate_layer_count": len(candidates),
        "matched_layer_count": len(matched),
        "missing_layer_count": len(missing),
        "coverage": len(matched) / max(len(candidates), 1),
        "matched_layers": matched,
        "missing_layers": missing,
        "candidate_groups": sorted(candidate_groups),
        "missing_mapped_groups": sorted(maximal_groups - candidate_groups),
        # 明确排除的电子端组：归一化/softmax/激活、控制/残差、unfold/fold 缓冲
        "excluded_electronic_groups": ["normalization_softmax_activation", "control_residual", "unfold_fold_buffer"],
        "claim_boundary": "Diagnostic module-output proxy for maximal mapping; not internal optical-operator or hardware accuracy evidence.",
    }


def reasonable_max_boundary(
    rows: list[dict[str, Any]],
    *,
    model_variant: str,
    available_module_names: set[str],
) -> dict[str, Any]:
    """Architecture-compatible high boundary: Linear, 1x1 Conv, and attention matmuls."""
    # （英文原注释）与架构兼容的高边界：线性层、1×1 卷积和注意力矩阵乘。
    # 通俗解释：这是论文认为"架构上合理"的上界——排除空间卷积/深度卷积
    # 等不适合光计算的算子，只保留矩阵乘法类算子作为候选。
    groups = PDPU_CANDIDATE_GROUPS | {"pointwise_conv"}  # 在标准候选基础上只追加 1×1 卷积
    candidates: set[str] = set()
    candidate_groups: set[str] = set()
    for row in rows:
        if row.get("model") != model_variant or row.get("op_group") not in groups:
            continue
        name = str(row.get("layer_name") or "")
        if not name:
            continue
        candidate_groups.add(str(row.get("op_group")))
        if row.get("op_group") in {"attention_score", "attention_value"}:
            name = name.rsplit(".", 1)[0]  # 同上：注意力算子映射到所属 Attention 模块
        candidates.add(name)
    matched = sorted(candidates & available_module_names)
    missing = sorted(candidates - available_module_names)
    return {
        "boundary_version": "reasonable-max-v1",
        "model_variant": model_variant,
        "mode": "reasonable-max",
        "candidate_layer_count": len(candidates),
        "matched_layer_count": len(matched),
        "missing_layer_count": len(missing),
        "coverage": len(matched) / max(len(candidates), 1),
        "matched_layers": matched,
        "missing_layers": missing,
        "candidate_groups": sorted(candidate_groups),
        "missing_mapped_groups": sorted(groups - candidate_groups),
        # 空间/深度卷积留在电子端（光计算更擅长稠密矩阵乘）
        "excluded_electronic_groups": ["spatial_conv", "depthwise_conv", "normalization_softmax_activation", "control_residual", "unfold_fold_buffer"],
        "claim_boundary": "Architecture-compatible high-boundary module-output diagnostic; not hardware accuracy evidence.",
    }
