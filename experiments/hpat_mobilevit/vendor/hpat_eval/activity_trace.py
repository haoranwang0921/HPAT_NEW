from __future__ import annotations

import math
from typing import Any

from .mobilevit_loader import primary_token_count, select_device, sync_device, variants_from_config


BYTES_PER_ELEM = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "int8": 1}


def _bytes(precision: str) -> int:
    return BYTES_PER_ELEM.get(precision, 4)


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return "x".join(str(int(dim)) for dim in value.shape)
    if isinstance(value, (tuple, list)) and value:
        return _shape(value[0])
    return ""


def _numel(shape: str) -> int:
    total = 1
    for part in shape.split("x"):
        if part:
            total *= int(part)
    return total if shape else 0


def _domain(op_group: str) -> tuple[str, str]:
    pdpu = {
        "pointwise_projection",
        "qkv_projection",
        "attention_output_projection",
        "ffn_linear",
        "classifier_linear",
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
        "execution_domain": domain,
        "pdpu_candidate": pdpu,
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
        "estimated_latency_share": "",
        "estimated_energy_share": "",
        "layer_name": layer_name,
        "module_type": module_type,
        "trace_source": trace_source,
        "evidence_label": evidence_label,
    }


def proxy_operator_rows(config: dict[str, Any], backend: str = "proxy", precision: str = "fp32") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elem_bytes = _bytes(precision)
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])
        n = primary_token_count(variant, config)
        r = int(variant["input_resolution"])
        input_shape = f"1x3x{r}x{r}"
        stem_channels = max(16, d // 4)
        hidden = 2 * d
        token_bytes = n * d * elem_bytes
        image_bytes = 3 * r * r * elem_bytes
        rows.extend(
            [
                _row(
                    variant=variant,
                    input_shape=input_shape,
                    backend=backend,
                    precision=precision,
                    stage="stem",
                    op_group="spatial_conv",
                    op_type="Conv2d 3x3",
                    tokens=0,
                    channels=stem_channels,
                    macs=r * r * 3 * stem_channels * 9,
                    input_bytes=image_bytes,
                    output_bytes=r * r * stem_channels * elem_bytes,
                    weight_bytes=3 * stem_channels * 9 * elem_bytes,
                    nonlinear_ops=r * r * stem_channels,
                ),
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
                _row(
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
    _fill_shares(rows)
    return rows


def _fill_shares(rows: list[dict[str, Any]]) -> None:
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["model"]] = totals.get(row["model"], 0.0) + float(row.get("estimated_macs") or 0.0)
    for row in rows:
        total = totals.get(row["model"], 0.0) or 1.0
        macs = float(row.get("estimated_macs") or 0.0)
        remainder_weight = 0.08 if row["execution_domain"] == "electronic remainder" and macs == 0 else 0.0
        row["estimated_latency_share"] = f"{100.0 * (macs / total + remainder_weight):.4f}"
        row["estimated_energy_share"] = f"{100.0 * (macs / total + remainder_weight):.4f}"


def _module_group(module: Any, layer_name: str) -> tuple[str, str]:
    module_type = module.__class__.__name__
    lower = f"{layer_name} {module_type}".lower()
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
            return "pointwise_conv", f"Conv2d {kernel_label}"
        if groups == in_channels and out_channels % max(in_channels, 1) == 0:
            return "depthwise_conv", f"Depthwise Conv2d {kernel_label}"
        return "spatial_conv", f"Conv2d {kernel_label}"
    if "linear" in module_type.lower():
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
        return "normalization_softmax_activation", module_type
    return "control_residual", module_type


def trace_timm_operator_rows(
    config: dict[str, Any], requested_device: str = "auto", precision: str = "fp32"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch  # type: ignore
    import timm  # type: ignore

    device, backend, reason = select_device(torch, requested_device)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    elem_bytes = _bytes(precision)
    for variant in variants_from_config(config):
        model_name = variant.get("timm_model")
        resolution = int(variant["input_resolution"])
        input_shape = f"1x3x{resolution}x{resolution}"
        try:
            model = timm.create_model(model_name, pretrained=False).eval().to(device)
        except Exception as exc:
            failures.append({"model": variant["variant"], "reason": str(exc)})
            continue
        hooks = []

        def hook(name: str):
            def inner(module: Any, inputs: Any, output: Any) -> None:
                module_type = module.__class__.__name__
                is_attention = module_type.lower() == "attention"
                if any(module.children()) and not is_attention:
                    return
                if is_attention:
                    in_shape = _shape(inputs)
                    dims = [int(part) for part in in_shape.split("x") if part]
                    if len(dims) != 3:
                        return
                    batch, tokens, channels = dims
                    heads = int(getattr(module, "num_heads", 1))
                    head_dim = int(getattr(module, "head_dim", max(channels // max(heads, 1), 1)))
                    attn_dim = heads * head_dim
                    attention_macs = batch * heads * tokens * tokens * head_dim
                    score_bytes = batch * heads * tokens * tokens * elem_bytes
                    activation_bytes = batch * tokens * attn_dim * elem_bytes
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
                op_group, op_type = _module_group(module, name)
                in_shape = _shape(inputs)
                out_shape = _shape(output)
                out_elems = _numel(out_shape)
                in_elems = _numel(in_shape)
                params = sum(int(p.numel()) for p in module.parameters(recurse=False)) if hasattr(module, "parameters") else 0
                macs = 0
                if module_type.lower().startswith("linear"):
                    macs = max(out_elems * int(getattr(module, "in_features", 1)), 0)
                elif module_type.lower().startswith("conv"):
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

        for name, module in model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(hook(name)))
        with torch.no_grad():
            x = torch.randn(1, 3, resolution, resolution, device=device)
            _ = model(x)
            sync_device(torch, device)
        for h in hooks:
            h.remove()
    _fill_shares(rows)
    meta = {
        "backend": backend,
        "device_reason": reason,
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
    bit_width = int(config.get("qkv", {}).get("default_bit_width", 8))
    activity_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_source = (
            "local torch/timm hook trace-derived HPAT activity proxy"
            if row.get("trace_source") == "torch_hooks"
            else "model/config-derived activity proxy"
        )
        base_pdpu = row.get("pdpu_candidate") in {"yes", "partial"}
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
        samples = max(output_elements, 0)
        dac_samples = samples
        if pdpu and row.get("op_group") in {"depthwise_conv", "spatial_conv", "pointwise_conv"}:
            groups = max(int(float(row.get("groups") or 1)), 1)
            out_channels = max(int(float(row.get("out_channels_actual") or 1)), 1)
            # Direct sliding-window lowering: stream each receptive-field value once
            # per output-channel group; do not materialize a full im2col matrix.
            dac_samples = max(int(math.ceil(macs * groups / out_channels)), samples)
        elif pdpu and row.get("op_group") in {"attention_score", "attention_value"}:
            # Strongest reuse-compatible proxy: dynamic Q/K/V operands are streamed
            # once and reused optically; ADC/PD/TIA counts remain output-element based.
            elem_bytes = max(_bytes(str(row.get("precision") or "fp32")), 1)
            dac_samples = max(input_bytes // elem_bytes, samples)
        active_ns = macs / float(config.get("activity_proxy", {}).get("ops_per_ns", 250000.0)) if pdpu else 0.0
        active_rings = max(weight_bytes * 8 // max(bit_width, 1), 0) if pdpu else 0
        program_fraction = float(config.get("activity_proxy", {}).get("mrr_program_fraction", 0.15))
        activity_rows.append(
            {
                "layer_id": f"{row['model']}_{index:03d}_{row['op_group']}",
                "op_type": row["op_type"],
                "model_variant": row["model"],
                "n_dac_samples": dac_samples if pdpu else 0,
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
                "calibration_events": 1 if pdpu and active_rings else 0,
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
