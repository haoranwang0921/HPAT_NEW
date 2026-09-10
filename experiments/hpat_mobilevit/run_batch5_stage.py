"""Batch-5: photonic three-stage timing — serialized (legacy) vs stage-parallel.

Runs the frozen P0 point (32x32 / 10 GHz / 25 MiB, DRAM 64 GB/s) on the
MobileViT-XXS/XS/S shape traces, for D0 and the H1/H3 photonic mappings:

  b5_serial  photonic_stage_pipeline=False -> must reproduce the frozen P0 ledger
  b5_pipe    photonic_stage_pipeline=True  -> DAC / optical / ADC pools overlap

The switch only changes the achievable overlap of the three stage latencies.
Per-stage dynamic energy is untouched; duration-proportional static terms
follow the (shorter) frame duration.

Run from the repo root:
    python -B -m experiments.hpat_mobilevit.run_batch5_stage
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from joint_sim.trace_io import load_manifest_jsonl, sha256_file
from .backends import PhysicalCoreCosts
from .mobile_profile import profile_costs
from .plan_sweep import P0, audit_events
from .run import REPO, run_case, write_csv, write_json
from .verify import verify_timeline

TRACE_ROOT = REPO / "results/hpat_mobilevit/trace_v1"
OUT_ROOT = REPO / "results/hpat_mobilevit/experiments_stagepipeline_v1"

MODELS = ["xxs", "xs", "s"]
MODES = {"D0": "digital", "H1": "linear", "H3": "linear_pointwise_attention"}
VARIANTS = {
    "b5_serial": {"photonic_stage_pipeline": False},
    "b5_pipe": {"photonic_stage_pipeline": True},
    # 全部优化开关同时打开 = 当前实现在 P0 主点上的"最好状态"
    "b5_all": {"photonic_stage_pipeline": True, "accumulator_residency": True,
               "dma_batch_fetch": True, "activity_gated_static": True,
               "reduce_compute_overlap": True},
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(VARIANTS), default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--models", nargs="*", choices=MODELS, default=MODELS)
    args = ap.parse_args()

    names = args.only if args.only else list(VARIANTS)
    out_root = Path(args.out_root) if args.out_root else OUT_ROOT
    base = json.loads((P0 / "config.json").read_text(encoding="utf-8"))
    traces = {v: load_manifest_jsonl(TRACE_ROOT / v / "operator_trace.jsonl")
              for v in args.models}
    out_root.mkdir(parents=True, exist_ok=True)

    for name in names:
        overrides = VARIANTS[name]
        out = out_root / name
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"{out} not empty; pass --out-root elsewhere")
        out.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(base)
        config["model_assumptions"].update(overrides)
        config["stage_pipeline_variant"] = {"name": name, "overrides": overrides}
        config["experiment_evidence"] = {
            "pretrained": False, "parameter_tag": "Nominal",
            "feasibility_tag": "Aggressive whole-array write assumption",
            "scope": "XXS/XS/S shape/cost traces only; no accuracy or device measurement",
            "stage_timing_policy": (
                "DAC/optical/ADC modelled as one serialized core timeline"
                if not overrides["photonic_stage_pipeline"] else
                "DAC/optical/ADC modelled as three device pools streaming successive "
                "vectors concurrently; completion = max(N*t_e, t_e+N*t_c, t_e+t_c+N*t_v) "
                "with N=2*M signed passes"),
        }
        write_json(out / "config.json", config)
        costs = profile_costs(PhysicalCoreCosts(config), config)
        write_json(out / "backend.json", costs.summary())

        rows, checks = [], []
        tick = time.monotonic()
        for model, (manifest, records) in traces.items():
            for label, mode in MODES.items():
                run_id = f"{model}__{label}__main__nominal"
                dest = out / run_id
                part = run_case(config, costs, mode, manifest, records, dest, run_id, timeline=True)
                rows.extend([dict(profile=name, mapping=label,
                                  dense_int8_tops=config["user_confirmed"].get("dense_int8_tops"),
                                  **r) for r in part])
                summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
                summary["effective_config"] = config
                events = verify_timeline(dest / "events.jsonl.gz", summary)
                checks.append(dict(run_id=run_id, valid=True, events=events))
                warm = part[-1]
                print(json.dumps(dict(variant=name, run_id=run_id, warm_ms=warm["latency_ms"],
                                      warm_mj=warm["energy_mj"])), flush=True)
        write_csv(out / "aggregate.csv", rows)
        write_json(out / "verification.json", dict(valid=all(c["valid"] for c in checks), cases=checks))
        write_json(out / "completion.json", dict(
            status="complete", variant=name, overrides=overrides,
            run_count=len(rows) // 2, frames=len(rows),
            wall_time_s=time.monotonic() - tick,
            aggregate_sha256=sha256_file(out / "aggregate.csv")))
        print(json.dumps(dict(variant=name, status="complete",
                              wall_time_s=time.monotonic() - tick)), flush=True)


if __name__ == "__main__":
    main()
