"""LLMCompass 电子基线 A/B.

A = 与 HPAT P0 同规格的电子基线（4 核 / 10 GHz / 32x32 INT8 / 25 MiB / 64 GB/s）
B = GA100（A100 级）作为外部参照

对同一份 MobileViT 形状 trace 的矩阵类算子（Linear / MatMul / groups==1 的 Conv2d，
trace 已给出 im2col 形式的 M/K/N）逐个调用 LLMCompass 的脉动阵列时序模型，汇总后
与当前手写 D0/HPAT 结果对照。

设计约束：
  * 只读 trace，不修改 HPAT 既有任何路径；
  * 结果按 (config, M, K, N, b, mode) 缓存，避免重复调用 SCALE-Sim；
  * 覆盖透明的三类：simulated / degenerate(N==1，depthwise 类) / failed。

用法（仓库根目录）：
    python -B -m experiments.hpat_mobilevit.run_llmcompass_baseline --config a --models xxs
    python -B -m experiments.hpat_mobilevit.run_llmcompass_baseline --config b --models xxs xs s
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

from joint_sim.trace_io import load_manifest_jsonl, sha256_file

REPO = pathlib.Path(__file__).resolve().parents[2]
LLMC = REPO / "LLMCompass"
TRACE_ROOT = REPO / "results/hpat_mobilevit/trace_v1"
OUT_ROOT = REPO / "results/hpat_mobilevit/llmcompass_baseline_v1"
CONFIGS = {
    "a": REPO / "experiments/hpat_mobilevit/llmcompass_configs/matched_a.json",
    "a2": REPO / "experiments/hpat_mobilevit/llmcompass_configs/matched_a_1p4ghz_16x16.json",
    "a3": REPO / "experiments/hpat_mobilevit/llmcompass_configs/a3_edge_1p4ghz_16x16.json",
    "b": LLMC / "configs/GA100.json",
}
GEMM_OP_TYPES = {"Linear", "MatMul", "Conv2d"}
DEGENERATE_LATENCY_S = 1.0  # 超过 1 s 视为退化值（N==1 类形状返回 1e9 量级）


def gemm_shape(rec):
    """返回 (M, K, N, batch) 或 None（不在矩阵类算子范围）。"""
    if not rec.get("macs"):
        return None
    if rec.get("op_type") not in GEMM_OP_TYPES:
        return None
    if rec.get("op_type") == "Conv2d" and rec.get("groups") not in (1, None):
        return None          # depthwise：LLMCompass 无模型，显式列为覆盖外
    M, K, N = rec.get("M"), rec.get("K"), rec.get("N")
    if not all(isinstance(v, int) and v > 0 for v in (M, K, N)):
        return None
    return M, K, N, int(rec.get("batch_repetitions") or 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--models", nargs="*", default=["xxs"])
    ap.add_argument("--mode", default="heuristic-GPU")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--force", action="store_true", help="忽略形状缓存重算")
    args = ap.parse_args()

    # 必须在 chdir(LLMCompass) 之前解析为绝对路径，否则缓存/输出会写到错误位置
    out = (pathlib.Path(args.out_root) if args.out_root
           else OUT_ROOT / f"{args.config}_llmcompass").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "shape_cache.json"
    cache = {} if args.force or not cache_path.exists() else json.loads(cache_path.read_text(encoding="utf-8"))

    # LLMCompass 用相对路径访问 ./systolic_array_model/...，必须在其根目录下运行
    sys.path.insert(0, str(LLMC))
    os.chdir(LLMC)
    (LLMC / "systolic_array_model/temp").mkdir(parents=True, exist_ok=True)

    from design_space_exploration.dse import read_architecture_template, template_to_system
    from software_model.matmul import Matmul
    from software_model.utils import Tensor, data_type_dict

    cfg_path = CONFIGS[args.config]
    arch = json.loads(cfg_path.read_text(encoding="utf-8"))
    system = template_to_system(arch)
    device = system.device
    dt = data_type_dict["int8"]
    print(json.dumps({"config": args.config, "name": arch.get("name"),
                      "cores": device.compute_module.core_count,
                      "freq_GHz": device.compute_module.clock_freq / 1e9,
                      "dram_GBps": device.io_module.bandwidth / 1e9}), flush=True)

    rows = []
    for model in args.models:
        manifest, records = load_manifest_jsonl(TRACE_ROOT / model / "operator_trace.jsonl")
        agg = defaultdict(float)
        per_op = []
        tick = time.monotonic()
        for rec in records:
            shape = gemm_shape(rec)
            if shape is None:
                if rec.get("macs"):
                    agg["out_of_scope_macs"] += rec["macs"]
                    agg["out_of_scope_ops"] += 1
                continue
            M, K, N, b = shape
            key = f"{args.config}|{M}|{K}|{N}|{b}|{args.mode}"
            if key in cache:
                entry = cache[key]
            else:
                entry = {"status": "failed", "latency_s": None}
                try:
                    op = Matmul(dt)
                    op(Tensor([b, M, K], dt), Tensor([K, N], dt))
                    op.compile_and_simulate(device, args.mode)
                    lat = op.latency
                    if lat is None or lat > DEGENERATE_LATENCY_S:
                        entry = {"status": "degenerate", "latency_s": None}
                    else:
                        entry = {"status": "simulated", "latency_s": lat,
                                 "energy_j": getattr(op, "energy", None)}
                except Exception as exc:                                  # noqa: BLE001
                    entry = {"status": "failed", "latency_s": None,
                             "error": f"{type(exc).__name__}: {exc}"[:200]}
                cache[key] = entry
                cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
            macs = M * K * N * b
            agg[f"{entry['status']}_macs"] += macs
            agg[f"{entry['status']}_ops"] += 1
            if entry.get("latency_s"):
                agg["simulated_latency_s"] += entry["latency_s"]
                if entry.get("energy_j"):
                    agg["simulated_energy_j"] += entry["energy_j"]
            per_op.append(dict(model=model, op_id=rec["op_id"], module_path=rec["module_path"],
                               op_type=rec["op_type"], op_role=rec.get("op_role"),
                               M=M, K=K, N=N, batch=b, macs=macs,
                               status=entry["status"], latency_s=entry.get("latency_s"),
                               error=entry.get("error")))
        wall = time.monotonic() - tick
        total_macs = sum(agg[k] for k in agg if k.endswith("_macs"))
        rows.append(dict(model=model,
                         simulated_ops=int(agg["simulated_ops"]), simulated_macs=int(agg["simulated_macs"]),
                         degenerate_ops=int(agg["degenerate_ops"]), degenerate_macs=int(agg["degenerate_macs"]),
                         failed_ops=int(agg["failed_ops"]), failed_macs=int(agg["failed_macs"]),
                         out_of_scope_ops=int(agg["out_of_scope_ops"]),
                         out_of_scope_macs=int(agg["out_of_scope_macs"]),
                         total_macs=int(total_macs),
                         simulated_latency_ms=agg["simulated_latency_s"] * 1e3,
                         simulated_energy_mj=(agg["simulated_energy_j"] * 1e3
                                              if agg["simulated_energy_j"] else None),
                         coverage_pct=100.0 * agg["simulated_macs"] / total_macs if total_macs else 0.0,
                         wall_s=round(wall, 1)))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        with (out / f"per_op_{model}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_op[0].keys()))
            w.writeheader()
            w.writerows(per_op)

    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "provenance.json").write_text(json.dumps(dict(
        config_name=arch.get("name"), config_path=str(cfg_path.relative_to(REPO)),
        config_sha256=sha256_file(cfg_path), mode=args.mode,
        scalesim_version=__import__("scalesim").__version__
        if hasattr(__import__("scalesim"), "__version__") else "unknown",
        python=sys.version.split()[0], llmcompass_root=str(LLMC),
        trace_manifest={m: json.loads((TRACE_ROOT / m / "operator_trace.jsonl")
                                      .read_text(encoding="utf-8").splitlines()[0])["total_operators"]
                        for m in args.models},
        scope="matrix-class operators only (Linear/MatMul/Conv2d groups==1); "
              "vector/normalisation ops and depthwise conv are out of scope by design",
        note="baseline A is derived from GA100.json with frequency/core/array/memory overridden "
             "to match the frozen HPAT P0 point; it is not a measured device",
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
