"""Capture actual ATen execution and producer dependencies for MobileViT.

Reuses HPAT's variant loader and module classifier. No old latency/energy
proxies enter this trace. Captured FP32 execution and simulated bit widths
are explicitly distinct. Unknown operators remain visible and must be
handled (or explicitly excluded) by the cost backend, never silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .vendor.hpat_eval.mobilevit_loader import variants_from_config
from .vendor.hpat_eval.activity_trace import _module_group
from joint_sim.trace_io import load_manifest_jsonl, sha256_file

ROOT = Path(__file__).resolve().parent
SOURCE_ATTR = "_hpat_joint_producer"


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors(v)


def tensor_digest(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


class ProducerTrace(TorchDispatchMode):
    def __init__(self, model, bits=8):
        super().__init__()
        self.model = model
        self.bits = bits
        self.records = []
        self.stack = []
        self.handles = []
        self.attention_calls = defaultdict(int)
        self.weights = {
            name: tensor_digest(module.weight)
            for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        }

    def __enter__(self):
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_pre_hook(
                lambda m, args, n=name: self.stack.append((n, m))))
            self.handles.append(module.register_forward_hook(
                lambda m, args, out: self.stack.pop() and None))
        return super().__enter__()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            for handle in self.handles:
                handle.remove()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = list(tensors((args, kwargs)))
        deps = sorted({getattr(t, SOURCE_ATTR) for t in inputs
                       if hasattr(t, SOURCE_ATTR)})
        output = func(*args, **kwargs)
        outputs = list(tensors(output))
        if not outputs:
            raise ValueError(f"Tensorless operator needs explicit treatment: {func}")
        name, module = self.stack[-1] if self.stack else ("", self.model)
        raw = str(func)
        short = raw.split(".")[1]
        group, _ = _module_group(module, name)
        elems = sum(t.numel() for t in outputs)
        record = {
            "trace_version": "hpat-aten-v1", "op_id": f"op_{len(self.records):05d}",
            "order": len(self.records), "module_path": name,
            "module_type": type(module).__name__, "aten_op": raw,
            "op_type": short, "op_role": group, "phase": "inference",
            "block_kind": "mobilevit", "layer_index": None, "rollout_step": 0,
            "call_index": 0, "input_shapes": [list(t.shape) for t in inputs],
            "output_shape": list(outputs[0].shape), "output_elements": elems,
            "M": max(elems, 1), "K": 1, "N": 1, "batch_repetitions": 1,
            "dtype": str(outputs[0].dtype), "input_bits": self.bits,
            "weight_bits": self.bits, "output_bits": self.bits,
            "weight_id": None, "weight_static": False, "macs": 0,
            "input_bytes": math.ceil(sum(t.numel() for t in inputs) * self.bits / 8),
            "output_bytes": math.ceil(elems * self.bits / 8), "weight_bytes": 0,
            "dependencies": deps, "captured_precision": "fp32",
        }
        if short in {"mm", "addmm", "bmm"}:
            a, b = (args[1], args[2]) if short == "addmm" else args[:2]
            M, K, N = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
            if K != b.shape[-2]:
                raise ValueError(f"Incompatible GEMM dimensions: {raw}")
            repetitions = math.prod(a.shape[:-2])
            record.update(M=M, K=K, N=N, batch_repetitions=repetitions,
                          op_type="MatMul", macs=M*K*N*repetitions)
            if isinstance(module, torch.nn.Linear):
                record.update(op_type="Linear", weight_static=True,
                              weight_id=f"{name}:{self.weights[name]}",
                              weight_bytes=math.ceil(K*N*self.bits/8),
                              bias_elements=N if module.bias is not None else 0)
            elif short == "bmm" and "attn" in name:
                i = self.attention_calls[name]
                record["op_role"] = "attention_score" if i % 2 == 0 else "attention_value"
                self.attention_calls[name] += 1
        elif short in {"convolution", "_convolution"}:
            a, w = args[:2]
            groups = int(args[8])
            if bool(args[6]):
                raise ValueError("Transposed convolution is outside MobileViT support")
            K = int(math.prod(w.shape[1:]))
            N = int(w.shape[0]) // groups
            M = int(outputs[0].numel()) // int(w.shape[0])
            record.update(op_type="Conv2d", M=M, K=K, N=N,
                          batch_repetitions=groups, groups=groups,
                          kernel_size=list(w.shape[2:]), stride=list(args[3]),
                          padding=list(args[4]), dilation=list(args[5]),
                          macs=M*K*N*groups, weight_static=True,
                          weight_id=f"{name}:{self.weights[name]}",
                          weight_bytes=math.ceil(w.numel()*self.bits/8))
        self.records.append(record)
        for tensor in outputs:
            setattr(tensor, SOURCE_ATTR, record["op_id"])
        return output


def compare_reference(records, rows):
    actual = defaultdict(int)
    expected = defaultdict(int)
    for r in records:
        if r["macs"]:
            path = r["module_path"]
            if r["op_role"] in {"attention_score", "attention_value"}:
                path += "." + r["op_role"]
            actual[path] += r["macs"]
    for r in rows:
        if int(float(r["estimated_macs"])):
            expected[r["layer_name"]] += int(float(r["estimated_macs"]))
    differences = [dict(layer=k, captured=actual[k], reference=expected[k])
                   for k in sorted(actual.keys() | expected.keys())
                   if actual[k] != expected[k]]
    return {"captured_macs": sum(actual.values()), "reference_macs": sum(expected.values()),
            "matching": not differences, "differences": differences,
            "reference_is_not_a_timing_measurement": True}


def export_variant(variant, output_dir, bits=8, seed=20260706, reference_rows=()):
    import timm
    torch.manual_seed(seed)
    model = timm.create_model(variant["timm_model"], pretrained=False).eval()
    decomposed = []
    for name, module in model.named_modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
            decomposed.append(name)
    r = int(variant["input_resolution"])
    x = torch.randn(1, 3, r, r)
    with torch.no_grad():
        expected = model(x)
        tracer = ProducerTrace(model, bits)
        with tracer:
            actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    records = tracer.records
    counts = Counter(r["aten_op"] for r in records)
    matching_rows = [r for r in reference_rows if r["model"] == variant["variant"]]
    manifest = {
        "model": variant["variant"], "timm_model": variant["timm_model"],
        "input_shape": [1, 3, r, r], "total_operators": len(records),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trace_model": "actual_aten_producer_dag", "captured_precision": "fp32",
        "simulated_bits": bits, "seed": seed, "pretrained": False,
        "purpose": "shape/cost trace; no accuracy or nonideality claim",
        "decomposed_attention_modules": decomposed,
        "capture_output_max_abs_error": float((actual-expected).abs().max()),
        "python": platform.python_version(), "torch": torch.__version__,
        "timm": timm.__version__, "aten_operator_counts": dict(counts),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p)
                          for p in [ROOT/"reference/mobilevit_operator_activity.csv",
                                    ROOT/"reference/paper_v1.json",
                                    ROOT/"vendor/hpat_eval/mobilevit_loader.py",
                                    ROOT/"vendor/hpat_eval/activity_trace.py", Path(__file__)]},
        "reference_comparison": compare_reference(records, matching_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "operator_trace.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in [manifest, *records]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    load_manifest_jsonl(path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["all", "xxs", "xs", "s"], default="all")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.bits <= 0:
        parser.error("--bits must be positive")
    torch.set_num_threads(args.threads)
    config = json.loads((ROOT/"reference/paper_v1.json").read_text(encoding="utf-8"))
    with (ROOT/"reference/mobilevit_operator_activity.csv").open(encoding="utf-8") as f:
        reference_rows = list(csv.DictReader(f))
    for variant in variants_from_config(config):
        key = variant["timm_model"].removeprefix("mobilevit_")
        if args.variant != "all" and args.variant != key:
            continue
        if args.resolution:
            variant["input_resolution"] = args.resolution
        manifest = export_variant(variant, args.output / key, args.bits,
                                  reference_rows=reference_rows)
        print(json.dumps({k: manifest[k] for k in ["model", "total_operators",
              "parameter_count", "reference_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
