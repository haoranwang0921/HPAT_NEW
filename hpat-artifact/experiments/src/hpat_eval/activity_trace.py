"""算子活动轨迹（activity trace）生成：把模型执行过程拆成一行行"算子活动"。

背景：为了评估 HPAT 光子芯片跑一个模型要花多少能耗/时间，需要知道
模型里每个算子（矩阵乘、卷积、softmax、残差……）到底做了多少"活动"——
用了多少次 DAC/ADC/PD/TIA 采样、激活了多少个 MRR 微环、搬了多少字节。
本文件负责生成这张"活动轨迹表"，提供两条路径：

1) proxy_operator_rows：纯配置推导（不用装 torch），按 MobileViT 的
   典型结构手写算子清单，适合快速扫参；
2) trace_timm_operator_rows：真的加载 timm 模型、挂 forward hook，
   实际跑一次前向，抓取每个模块的真实输入/输出形状与参数量来估算 MAC。

之后由 activity_from_operator_rows 把算子行换算成 HPAT 活动行
（字段见 HPAT_ACTIVITY_FIELDS，可被 schemas 校验后进入能耗建模）。

易混淆点：
- "算子行"（operator row）描述计算形状；"活动行"（activity row）
  描述光子芯片硬件活动量。前者是模型的视角，后者是硬件的视角。
- hook 抓的是"模块级"数据；注意力 QK^T 与 Attention@V 不是独立模块，
  只能在 Attention 模块内部按形状推算（见 hook 里的 attention 分支）。
"""

from __future__ import annotations

import math
from typing import Any

from .mobilevit_loader import primary_token_count, select_device, sync_device, variants_from_config


# 各精度下每个元素占的字节数（未知精度默认按 fp32 处理）
BYTES_PER_ELEM = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "int8": 1}


def _bytes(precision: str) -> int:
    """查精度名 → 每元素字节数（未知精度退回 4 字节）。

    :param precision: 精度字符串（如 "fp32"）。
    :return: 每元素字节数。
    """
    return BYTES_PER_ELEM.get(precision, 4)


def _shape(value: Any) -> str:
    """把 tensor/tuple/list 的形状格式化成 "1x3x192x192" 字符串。

    取第一个元素递归（适用于 hook 收到 tuple 输入的情况）。

    :param value: 带 shape 属性的对象或可迭代容器。
    :return: 形状字符串；无法识别时返回 ""。
    """
    if hasattr(value, "shape"):
        return "x".join(str(int(dim)) for dim in value.shape)
    if isinstance(value, (tuple, list)) and value:
        return _shape(value[0])
    return ""


def _numel(shape: str) -> int:
    """把 "1x3x192x192" 形状字符串折算成元素总数。

    :param shape: 形状字符串。
    :return: 元素个数（空字符串返回 0）。
    """
    total = 1
    for part in shape.split("x"):
        if part:
            total *= int(part)
    return total if shape else 0


def _domain(op_group: str) -> tuple[str, str]:
    """按算子组判断它的"执行域"归属（光子端/扩展候选/电子端/混合）。

    返回 (执行域标签, PDPU 候选标记)：
    - "PDPU-candidate linear/MVM" / "yes"：可直接映射到光点积单元；
    - "PDPU-extension candidate" / "extension"：扩展映射候选（卷积/注意力）；
    - "electronic remainder" / "no"：只能留在电子端（softmax、归一化等）；
    - "hybrid support" / "partial"：混合支撑（控制、残差等）。

    :param op_group: 算子组名。
    :return: (执行域, PDPU 候选标记) 二元组。
    """
    pdpu = {
        "pointwise_projection",      # 逐点投影（1×1 稠密）
        "qkv_projection",            # Q/K/V 投影
        "attention_output_projection",  # 注意力输出投影
        "ffn_linear",                # 前馈线性层
        "classifier_linear",         # 分类线性层
    }
    if op_group in pdpu:
        return "PDPU-candidate linear/MVM", "yes"
    if op_group in {"pointwise_conv", "attention_score", "attention_value", "depthwise_conv", "spatial_conv"}:
        return "PDPU-extension candidate", "extension"
    if op_group in {"unfold_fold_buffer", "normalization_softmax_activation", "control_residual"}:
        return "electronic remainder", "no"
    return "hybrid support", "partial"


def _row(
    *,
    variant: dict[str, Any],
    input_shape: str,
    backend: str,
    precision: str,
    op_group: str,
    op_type: str,
    stage: str,
    tokens: int,
    channels: int,
    macs: int,
    input_bytes: int,
    output_bytes: int,
    weight_bytes: int,
    output_elements: int | None = None,
    kernel_elements: int = 1,
    groups: int = 1,
    in_channels_actual: int = 0,
    out_channels_actual: int = 0,
    nonlinear_ops: int = 0,
    layer_name: str = "",
    module_type: str = "",
    trace_source: str = "config_proxy",
    evidence_label: str = "model/config-derived proxy; not torch traced",
) -> dict[str, Any]:
    """构造一行"算子行"（统一字段格式，供下游分析表使用）。

    算子行描述一次计算操作的形状特征：输入/输出/权重字节数、
    估计 MAC 数、执行域归属等。所有字段以关键字参数传入。

    :param variant: 模型变体规格字典。
    :param input_shape: 模型输入形状字符串。
    :param backend: 计算后端标签（proxy / mps / cuda / cpu）。
    :param precision: 数值精度。
    :param op_group: 算子组名（决定执行域归属）。
    :param op_type: 算子类型描述（如 "QK^T"）。
    :param stage: 模型阶段（stem / attention / ffn / classifier 等）。
    :param tokens: token 数量。
    :param channels: 通道数。
    :param macs: 估计的乘加次数。
    :param input_bytes: 输入字节数。
    :param output_bytes: 输出字节数。
    :param weight_bytes: 权重字节数。
    :param output_elements: 输出元素个数（缺省按 output_bytes/每元素字节 推算）。
    :param kernel_elements: 卷积核元素数。
    :param groups: 卷积分组数。
    :param in_channels_actual: 实际输入通道数。
    :param out_channels_actual: 实际输出通道数。
    :param nonlinear_ops: 非线性算子执行次数。
    :param layer_name: 层名（如 "blocks.0.attention.qkv"）。
    :param module_type: torch 模块类型名。
    :param trace_source: 轨迹来源标签。
    :param evidence_label: 证据标签。
    :return: 一行算子数据字典。
    """
    domain, pdpu = _domain(op_group)
    return {
        "model": variant["variant"],
        "model_name": variant.get("timm_model", ""),
        "input_shape": input_shape,
        "backend": backend,
        "precision": precision,
        "stage": stage,
        "op_group": op_group,
        "op_type": op_type,
        "execution_domain": domain,   # 执行域（光子/电子/混合）
        "pdpu_candidate": pdpu,       # PDPU 候选标记
        "tokens": tokens,
        "channels": channels,
        "embedding_dim": variant["embedding_dim"],
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "weight_bytes": weight_bytes,
        "output_elements": int(output_elements if output_elements is not None else output_bytes // max(_bytes(precision), 1)),
        "kernel_elements": kernel_elements,
        "groups": groups,
        "in_channels_actual": in_channels_actual,
        "out_channels_actual": out_channels_actual,
        "estimated_macs": macs,
        "nonlinear_ops": nonlinear_ops,
        "estimated_latency_share": "",  # 延迟占比（后续 _fill_shares 填充）
        "estimated_energy_share": "",   # 能耗占比（后续 _fill_shares 填充）
        "layer_name": layer_name,
        "module_type": module_type,
        "trace_source": trace_source,
        "evidence_label": evidence_label,
    }


def proxy_operator_rows(config: dict[str, Any], backend: str = "proxy", precision: str = "fp32") -> list[dict[str, Any]]:
    """路径一：从配置直接推导算子行（不装 torch，纯公式推算）。

    按 MobileViT 的标准结构手写一份"虚拟算子清单"：stem 卷积、
    1×1 投影、tokenization、QKV 投影、注意力分数/值、输出投影、
    FFN 扩展/收缩、LayerNorm/残差、分类头，每个算子给出输入输出
    形状推算出的字节量与 MAC 数。所有量都由 embedding_dim d、
    token 数 n、分辨率 r 的公式给出。

    :param config: 实验配置。
    :param backend: 后端标签。
    :param precision: 精度（决定每元素字节数）。
    :return: 算子行列表（已填充延迟/能耗占比）。
    """
    rows: list[dict[str, Any]] = []
    elem_bytes = _bytes(precision)
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])        # 嵌入维度
        n = primary_token_count(variant, config) # token 数
        r = int(variant["input_resolution"])     # 输入分辨率（边长）
        input_shape = f"1x3x{r}x{r}"             # 图像输入形状：批1 × 3 通道 × r × r
        stem_channels = max(16, d // 4)          # stem 卷积输出通道数（经验公式）
        hidden = 2 * d                           # FFN 隐藏层宽度（MobileViT 惯例扩两倍）
        token_bytes = n * d * elem_bytes         # 一份 token 序列的字节数
        image_bytes = 3 * r * r * elem_bytes     # 整张输入图像的字节数
        rows.extend(
            [
                _row(  # 1) stem 3×3 卷积：把 3 通道图像变成 stem_channels 通道
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="stem",
                    op_group="spatial_conv",
                    op_type="Conv2d 3x3",
                    tokens=0,
                    channels=stem_channels,
                    macs=r * r * 3 * stem_channels * 9,  # 每个像素 × 输入3通道 × 输出通道 × 3×3核
                    input_bytes=image_bytes,
                    output_bytes=r * r * stem_channels * elem_bytes,
                    weight_bytes=3 * stem_channels * 9 * elem_bytes,
                    nonlinear_ops=r * r * stem_channels,
                ),
                _row(  # 2) 1×1 投影：局部表征阶段的稠密投影
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="local representation",
                    op_group="pointwise_projection",
                    op_type="1x1 projection / dense",
                    tokens=n,
                    channels=d,
                    macs=n * stem_channels * d,
                    input_bytes=token_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=stem_channels * d * elem_bytes,
                ),
                _row(  # 3) tokenization：unfold/reshape 整理 token（纯搬运，无 MAC）
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="tokenization",
                    op_group="unfold_fold_buffer",
                    op_type="unfold/reshape/buffer",
                    tokens=n,
                    channels=d,
                    macs=0,
                    input_bytes=token_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=0,
                ),
                _row(  # 4) QKV 投影：3 个稠密层，每个 n×d×d 次乘加
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="attention",
                    op_group="qkv_projection",
                    op_type="3 dense Q/K/V projections",
                    tokens=n,
                    channels=d,
                    macs=3 * n * d * d,
                    input_bytes=token_bytes,
                    output_bytes=3 * token_bytes,
                    weight_bytes=3 * d * d * elem_bytes,
                ),
                _row(  # 5) 注意力分数 QK^T：n×n×d
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="attention",
                    op_group="attention_score",
                    op_type="QK^T",
                    tokens=n,
                    channels=d,
                    macs=n * n * d,
                    input_bytes=2 * token_bytes,
                    output_bytes=n * n * elem_bytes,
                    weight_bytes=0,
                ),
                _row(  # 6) softmax：无乘加，只有非线性（n×n 次）
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="attention",
                    op_group="normalization_softmax_activation",
                    op_type="softmax / scaling",
                    tokens=n,
                    channels=d,
                    macs=0,
                    input_bytes=n * n * elem_bytes,
                    output_bytes=n * n * elem_bytes,
                    weight_bytes=0,
                    nonlinear_ops=n * n,
                ),
                _row(  # 7) 注意力加权和 Attention@V：n×n×d
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="attention",
                    op_group="attention_value",
                    op_type="attention x V",
                    tokens=n,
                    channels=d,
                    macs=n * n * d,
                    input_bytes=n * n * elem_bytes + token_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=0,
                ),
                _row(  # 8) 注意力输出投影：n×d×d
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="attention",
                    op_group="attention_output_projection",
                    op_type="dense output projection",
                    tokens=n,
                    channels=d,
                    macs=n * d * d,
                    input_bytes=token_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=d * d * elem_bytes,
                ),
                _row(  # 9) FFN 扩展+收缩：两次 d↔hidden 的稠密变换
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="ffn",
                    op_group="ffn_linear",
                    op_type="dense expand/project",
                    tokens=n,
                    channels=hidden,
                    macs=2 * n * d * hidden,
                    input_bytes=token_bytes + n * hidden * elem_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=2 * d * hidden * elem_bytes,
                ),
                _row(  # 10) FFN 里的激活函数（GELU/ReLU）：无乘加
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="ffn",
                    op_group="normalization_softmax_activation",
                    op_type="GELU/ReLU",
                    tokens=n,
                    channels=hidden,
                    macs=0,
                    input_bytes=n * hidden * elem_bytes,
                    output_bytes=n * hidden * elem_bytes,
                    weight_bytes=0,
                    nonlinear_ops=n * hidden,
                ),
                _row(  # 11) LayerNorm/残差/控制：无乘加，有少量非线性
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="fusion",
                    op_group="control_residual",
                    op_type="LayerNorm/residual/control",
                    tokens=n,
                    channels=d,
                    macs=0,
                    input_bytes=2 * token_bytes,
                    output_bytes=token_bytes,
                    weight_bytes=0,
                    nonlinear_ops=4 * n * d,
                ),
                _row(  # 12) 分类头：1000 类 × d 维
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="classifier",
                    op_group="classifier_linear",
                    op_type="final dense",
                    tokens=1,
                    channels=d,
                    macs=1000 * d,
                    input_bytes=d * elem_bytes,
                    output_bytes=1000 * elem_bytes,
                    weight_bytes=1000 * d * elem_bytes,
                ),
            ]
        )
    _fill_shares(rows)  # 补算各算子的延迟/能耗占比
    return rows


def _fill_shares(rows: list[dict[str, Any]]) -> None:
    """给每行算子补算"延迟占比/能耗占比"（按 MAC 数占模型总 MAC 的比例）。

    没有 MAC 的电子端算子（如 softmax）不全是零开销：给它们一个
    固定的 8% 基础权重（remainder_weight），近似表示这类算子在真实
    执行中也要花时间/能耗。

    :param rows: 算子行列表（就地修改，不返回新列表）。
    """
    totals: dict[str, float] = {}
    # 先按模型累计总 MAC
    for row in rows:
        totals[row["model"]] = totals.get(row["model"], 0.0) + float(row.get("estimated_macs") or 0.0)
    for row in rows:
        total = totals.get(row["model"], 0.0) or 1.0  # 防除零
        macs = float(row.get("estimated_macs") or 0.0)
        # 电子端且无 MAC 的算子给 8% 固定权重，其余按 MAC 占比
        remainder_weight = 0.08 if row["execution_domain"] == "electronic remainder" and macs == 0 else 0.0
        row["estimated_latency_share"] = f"{100.0 * (macs / total + remainder_weight):.4f}"
        row["estimated_energy_share"] = f"{100.0 * (macs / total + remainder_weight):.4f}"


def _module_group(module: Any, layer_name: str) -> tuple[str, str]:
    """根据 torch 模块的类型与层名，归类算子组并给出类型描述。

    判定顺序：卷积（1×1→pointwise；分组=输入通道→depthwise；否则
    spatial）→ 线性（按名字里的 qkv/attn/head/mlp 关键字细分）→
    归一化/激活 → 其余归 control_residual。

    :param module: torch 模块对象。
    :param layer_name: 层名字符串。
    :return: (op_group, op_type) 二元组。
    """
    module_type = module.__class__.__name__
    lower = f"{layer_name} {module_type}".lower()  # 层名+类型拼接，用于关键字匹配
    if "conv" in module_type.lower():
        kernel = getattr(module, "kernel_size", (1, 1))
        if not isinstance(kernel, tuple):
            kernel = (int(kernel), int(kernel))
        kernel = tuple(int(v) for v in kernel)
        groups = int(getattr(module, "groups", 1))
        in_channels = int(getattr(module, "in_channels", 1))
        out_channels = int(getattr(module, "out_channels", 1))
        kernel_label = "x".join(str(v) for v in kernel)
        if kernel == (1, 1) and groups == 1:
            return "pointwise_conv", f"Conv2d {kernel_label}"  # 1×1 标准卷积
        if groups == in_channels and out_channels % max(in_channels, 1) == 0:
            return "depthwise_conv", f"Depthwise Conv2d {kernel_label}"  # 深度卷积
        return "spatial_conv", f"Conv2d {kernel_label}"  # 空间卷积
    if "linear" in module_type.lower():
        # 线性层按层名关键字细分归属
        if "qkv" in lower:
            return "qkv_projection", "Linear"
        if "attn" in lower:
            return "attention_output_projection", "Linear"
        if "head" in lower or "classifier" in lower:
            return "classifier_linear", "Linear"
        if "mlp" in lower or "ffn" in lower:
            return "ffn_linear", "Linear"
        return "pointwise_projection", "Linear"
    if any(token in lower for token in ["softmax", "norm", "gelu", "relu", "act"]):
        return "normalization_softmax_activation", module_type  # 归一化/激活
    return "control_residual", module_type  # 兜底：控制/残差


def trace_timm_operator_rows(
    config: dict[str, Any], requested_device: str = "auto", precision: str = "fp32"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """路径二：真实加载 timm 模型、跑一次前向，用 hook 抓算子行。

    步骤：
    1) 选设备（auto/mps/cuda/cpu），加载 MobileViT 模型（pretrained=False）；
    2) 给所有非空模块注册 forward hook：模块前向完成后回调，
       用输入/输出形状推算出 MAC、字节量并生成算子行；
    3) 特殊处理 Attention 模块：QK^T 与 Attention@V 不是独立模块，
       按 批×头数×token×head_dim 的形状推算成两行；
    4) 跑一次随机输入的前向，卸载 hook，返回算子行与运行元信息。

    :param config: 实验配置。
    :param requested_device: 请求的设备（默认 auto）。
    :param precision: 数值精度。
    :return: (算子行列表, 元信息字典)。元信息含后端、版本、失败记录。
    """
    import torch  # type: ignore
    import timm  # type: ignore

    device, backend, reason = select_device(torch, requested_device)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []  # 记录哪些模型加载失败（如缺网络权重）
    elem_bytes = _bytes(precision)
    for variant in variants_from_config(config):
        model_name = variant.get("timm_model")
        resolution = int(variant["input_resolution"])
        input_shape = f"1x3x{resolution}x{resolution}"
        try:
            model = timm.create_model(model_name, pretrained=False).eval().to(device)
        except Exception as exc:
            failures.append({"model": variant["variant"], "reason": str(exc)})
            continue  # 加载失败则跳过该变体，不中断整体
        hooks = []

        def hook(name: str):
            """返回一个绑定层名的 forward hook 回调（闭包捕获 name）。"""
            def inner(module: Any, inputs: Any, output: Any) -> None:
                module_type = module.__class__.__name__
                is_attention = module_type.lower() == "attention"
                # 跳过含子模块的容器（只在叶模块上记录），Attention 除外（需特殊推算）
                if any(module.children()) and not is_attention:
                    return
                if is_attention:
                    # ---- Attention 模块特殊处理：按形状推算 QK^T 和 Attention@V ----
                    in_shape = _shape(inputs)
                    dims = [int(part) for part in in_shape.split("x") if part]
                    if len(dims) != 3:  # 期望 (batch, tokens, channels)
                        return
                    batch, tokens, channels = dims
                    heads = int(getattr(module, "num_heads", 1))
                    head_dim = int(getattr(module, "head_dim", max(channels // max(heads, 1), 1)))
                    attn_dim = heads * head_dim
                    attention_macs = batch * heads * tokens * tokens * head_dim  # QK^T 或 AV 的乘加
                    score_bytes = batch * heads * tokens * tokens * elem_bytes   # 注意力分数矩阵字节
                    activation_bytes = batch * tokens * attn_dim * elem_bytes    # 激活值字节
                    for op_group, op_type, input_bytes, output_bytes in [
                        ("attention_score", "QK^T", 2 * activation_bytes, score_bytes),
                        ("attention_value", "Attention x V", score_bytes + activation_bytes, activation_bytes),
                    ]:
                        rows.append(
                            _row(
                                variant=variant,
                                input_shape=input_shape,
                                backend=backend,
                                precision=precision,
                                stage=name.split(".")[0] if name else "model",
                                op_group=op_group,
                                op_type=op_type,
                                tokens=tokens,
                                channels=channels,
                                macs=attention_macs,
                                input_bytes=input_bytes,
                                output_bytes=output_bytes,
                                weight_bytes=0,
                                output_elements=output_bytes // elem_bytes,
                                layer_name=f"{name}.{op_group}",
                                module_type="Attention(function-level inferred)",
                                trace_source="torch_hooks_shape_inferred",
                                evidence_label=(
                                    "local attention-module shape-derived function-op estimate; "
                                    "extension scenario only, not measured HPAT execution"
                                ),
                            )
                        )
                    return
                # ---- 普通叶模块：按实际形状与参数推算 MAC/字节量 ----
                op_group, op_type = _module_group(module, name)
                in_shape = _shape(inputs)
                out_shape = _shape(output)
                out_elems = _numel(out_shape)
                in_elems = _numel(in_shape)
                params = sum(int(p.numel()) for p in module.parameters(recurse=False)) if hasattr(module, "parameters") else 0
                macs = 0
                if module_type.lower().startswith("linear"):
                    # 线性层：输出元素数 × 输入特征维
                    macs = max(out_elems * int(getattr(module, "in_features", 1)), 0)
                elif module_type.lower().startswith("conv"):
                    # 卷积：输出元素 × (输入通道/分组) × 核元素数
                    kernel = getattr(module, "kernel_size", (1, 1))
                    kernel_elems = int(math.prod(kernel if isinstance(kernel, tuple) else (kernel, kernel)))
                    in_channels = int(getattr(module, "in_channels", 1))
                    groups = int(getattr(module, "groups", 1))
                    macs = out_elems * max(in_channels // groups, 1) * kernel_elems
                else:
                    kernel_elems = 1
                    in_channels = 0
                    groups = 1
                nonlinear = out_elems if op_group == "normalization_softmax_activation" else 0
                rows.append(
                    _row(
                        variant=variant,
                        input_shape=input_shape,
                        backend=backend,
                        precision=precision,
                        stage=name.split(".")[0] if name else "model",
                        op_group=op_group,
                        op_type=op_type,
                        tokens=primary_token_count(variant, config),
                        channels=int(variant["embedding_dim"]),
                        macs=int(macs),
                        input_bytes=in_elems * elem_bytes,
                        output_bytes=out_elems * elem_bytes,
                        weight_bytes=params * elem_bytes,
                        output_elements=out_elems,
                        kernel_elements=kernel_elems if module_type.lower().startswith("conv") else 1,
                        groups=groups if module_type.lower().startswith("conv") else 1,
                        in_channels_actual=in_channels if module_type.lower().startswith("conv") else int(getattr(module, "in_features", 0)),
                        out_channels_actual=int(getattr(module, "out_channels", getattr(module, "out_features", 0))),
                        nonlinear_ops=nonlinear,
                        layer_name=name,
                        module_type=module_type,
                        trace_source="torch_hooks",
                        evidence_label="local torch/timm hook trace; not edge evidence",
                    )
                )

            return inner

        # 给所有有名字的模块注册 forward hook
        for name, module in model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            x = torch.randn(1, 3, resolution, resolution, device=device)  # 随机输入
            _ = model(x)                       # 跑一次前向触发所有 hook
            sync_device(torch, device)         # 同步设备，确保 hook 全部执行完
        for h in hooks:
            h.remove()  # 卸载 hook，避免下次重复计数
    _fill_shares(rows)  # 补算延迟/能耗占比
    meta = {
        "backend": backend,
        "device_reason": reason,          # 设备选择原因（写进清单）
        "torch_version": getattr(torch, "__version__", "unknown"),
        "timm_version": getattr(timm, "__version__", "unknown"),
        "failures": failures,
    }
    return rows, meta


def activity_from_operator_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    mapping_scenario: str = "linear_only",
) -> list[dict[str, Any]]:
    """把"算子行"换算成"HPAT 活动行"（硬件活动量视角）。

    换算规则（每个算子若被映射进光子端 pdpu=True，否则活动量全 0）：
    - DAC 采样数：默认取输出元素数；卷积按"直接滑窗展开"（不建 im2col
      大矩阵）近似为 ceil(MAC×分组/输出通道)；注意力按"QKV 操作数只流
      一遍光复用"取 输入元素数 与 输出元素数 的较大值；
    - ADC/PD/TIA 采样数：取输出元素数（读出多少就采多少）；
    - 活跃 MRR 数：按权重比特数折算（每 bit 一个微环）；
    - 光路活动时长：MAC / 每纳秒操作数（仅光子端算子）；
    - 缓冲/总线字节、非线性算子数、校准事件数等照搬。

    mapping_scenario 决定哪些扩展算子也被算进光子端：
    linear_only（只线性层）/ reasonable_max（+1×1 卷积与注意力）/
    maximal_all_mac（+所有卷积）。

    :param rows: 算子行列表（proxy 或 hook 路径产生）。
    :param config: 实验配置。
    :param mapping_scenario: 映射场景。
    :return: HPAT 活动行列表（字段见 HPAT_ACTIVITY_FIELDS）。
    """
    bit_width = int(config.get("qkv", {}).get("default_bit_width", 8))
    activity_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        # 证据标签按轨迹来源区分：hook 抓的 / 配置推导的
        row_source = (
            "local torch/timm hook trace-derived HPAT activity proxy"
            if row.get("trace_source") == "torch_hooks"
            else "model/config-derived activity proxy"
        )
        base_pdpu = row.get("pdpu_candidate") in {"yes", "partial"}  # 基础候选
        # 扩展组（卷积/注意力）是否被纳入，取决于映射场景
        extension_group = row.get("op_group") in {
            "pointwise_conv",
            "attention_score",
            "attention_value",
            "depthwise_conv",
            "spatial_conv",
        }
        reasonable_extension = row.get("op_group") in {"pointwise_conv", "attention_score", "attention_value"}
        pdpu = base_pdpu or (mapping_scenario == "reasonable_max" and reasonable_extension) or (mapping_scenario == "maximal_all_mac" and extension_group)
        macs = int(float(row.get("estimated_macs") or 0))
        output_bytes = int(float(row.get("output_bytes") or 0))
        input_bytes = int(float(row.get("input_bytes") or 0))
        weight_bytes = int(float(row.get("weight_bytes") or 0))
        output_elements = int(float(row.get("output_elements") or output_bytes // max(_bytes(str(row.get("precision") or "fp32")), 1)))
        samples = max(output_elements, 0)  # ADC/PD/TIA 的基准采样数 = 输出元素数
        dac_samples = samples             # DAC 默认同样取输出元素数
        if pdpu and row.get("op_group") in {"depthwise_conv", "spatial_conv", "pointwise_conv"}:
            groups = max(int(float(row.get("groups") or 1)), 1)
            out_channels = max(int(float(row.get("out_channels_actual") or 1)), 1)
            # Direct sliding-window lowering: stream each receptive-field value once
            # per output-channel group; do not materialize a full im2col matrix.
            # （英文原注释）直接滑窗降级：每个感受野值按输出通道组只流一遍，
            # 不物化完整的 im2col 大矩阵（省内存、省采样）。
            dac_samples = max(int(math.ceil(macs * groups / out_channels)), samples)
        elif pdpu and row.get("op_group") in {"attention_score", "attention_value"}:
            # Strongest reuse-compatible proxy: dynamic Q/K/V operands are streamed
            # once and reused optically; ADC/PD/TIA counts remain output-element based.
            # （英文原注释）最强复用兼容近似：动态 Q/K/V 操作数只流一遍并光复用；
            # ADC/PD/TIA 计数仍按输出元素数。
            elem_bytes = max(_bytes(str(row.get("precision") or "fp32")), 1)
            dac_samples = max(input_bytes // elem_bytes, samples)
        # 光路活动时长：仅光子端算子；ops_per_ns 是硬件吞吐假设（默认每纳秒 25 万次）
        active_ns = macs / float(config.get("activity_proxy", {}).get("ops_per_ns", 250000.0)) if pdpu else 0.0
        # 活跃 MRR 数：按权重比特数折算（bit_width 位宽下，每个权重比特占一个微环）
        active_rings = max(weight_bytes * 8 // max(bit_width, 1), 0) if pdpu else 0
        # 编程 MRR 比例：默认 15% 的活跃微环需要重新编程
        program_fraction = float(config.get("activity_proxy", {}).get("mrr_program_fraction", 0.15))
        activity_rows.append(
            {
                "layer_id": f"{row['model']}_{index:03d}_{row['op_group']}",  # 全局唯一层 ID
                "op_type": row["op_type"],
                "model_variant": row["model"],
                "n_dac_samples": dac_samples if pdpu else 0,  # 非光子端算子活动量全 0
                "n_adc_samples": samples if pdpu else 0,
                "n_pd_samples": samples if pdpu else 0,
                "n_tia_samples": samples if pdpu else 0,
                "n_mrr_active": active_rings,
                "n_mrr_program": int(active_rings * program_fraction) if pdpu else 0,
                "active_optical_time_ns": f"{active_ns:.6f}",
                "buffer_read_bytes": input_bytes + weight_bytes,
                "buffer_write_bytes": output_bytes,
                "bus_bytes": input_bytes + output_bytes + weight_bytes,
                "nonlinear_ops": int(float(row.get("nonlinear_ops") or 0)),
                "calibration_events": 1 if pdpu and active_rings else 0,  # 有光计算就记 1 次校准
                "thermal_tracking_time_ns": f"{active_ns:.6f}" if pdpu else "0.000000",
                "source_operator_group": row["op_group"],
                "evidence_label": (
                    row_source
                    + (
                        "; maximal all-MAC direct-streaming mapping scenario"
                        if mapping_scenario == "maximal_all_mac"
                        else ("; reasonable-max Linear/1x1/attention mapping scenario" if mapping_scenario == "reasonable_max" else "; linear-only mapping scenario")
                    )
                    + "; simulator export required for claim promotion"
                ),
            }
        )
    return activity_rows


# 算子行表的列名清单（顺序即 CSV 表头顺序）
OPERATOR_ACTIVITY_FIELDS = [
    "model",
    "model_name",
    "input_shape",
    "backend",
    "precision",
    "stage",
    "op_group",
    "op_type",
    "execution_domain",
    "pdpu_candidate",
    "tokens",
    "channels",
    "embedding_dim",
    "input_bytes",
    "output_bytes",
    "weight_bytes",
    "output_elements",
    "kernel_elements",
    "groups",
    "in_channels_actual",
    "out_channels_actual",
    "estimated_macs",
    "nonlinear_ops",
    "estimated_latency_share",
    "estimated_energy_share",
    "layer_name",
    "module_type",
    "trace_source",
    "evidence_label",
]


# HPAT 活动行表的列名清单（可被 schemas 校验后进入能耗建模）
HPAT_ACTIVITY_FIELDS = [
    "layer_id",
    "op_type",
    "model_variant",
    "n_dac_samples",
    "n_adc_samples",
    "n_pd_samples",
    "n_tia_samples",
    "n_mrr_active",
    "n_mrr_program",
    "active_optical_time_ns",
    "buffer_read_bytes",
    "buffer_write_bytes",
    "bus_bytes",
    "nonlinear_ops",
    "calibration_events",
    "thermal_tracking_time_ns",
    "source_operator_group",
    "evidence_label",
]
