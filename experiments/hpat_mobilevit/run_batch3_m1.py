"""Batch-3: M1 reduce-under-compute pipelining sweep.

All variants run on the rolled-back mobile_reference profile at 10 TOPS:
  b3_none  all switches off  -> must reproduce the historical ledger bit-exactly
  b3_m1    reduce_compute_overlap=True (M1 only)
  b3_all   reduce_compute_overlap + electronic_io_compute_overlap + activity_gated_static

Run from repo root:  python -B -m experiments.hpat_mobilevit.run_batch3_m1
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from joint_sim.trace_io import load_manifest_jsonl, sha256_file
from .backends import PhysicalCoreCosts
from .mobile_profile import resolve_profile, profile_costs
from .run import ROOT, REPO, run_case, write_csv, write_json
from .run_mobile import MODES

SPEC_PATH = ROOT / "mobile_a16_alignment.json"
BASE_PATH = SPEC_PATH.parent / json.loads(SPEC_PATH.read_text(encoding="utf-8"))["base_config"]
TRACE_ROOT = REPO / "results/hpat_mobilevit/trace_v1"
OUT_ROOT = REPO / "results/hpat_mobilevit/experiments_batch3"

VARIANTS = {
    "b3_none": {},
    "b3_m1": {"reduce_compute_overlap": True},
    "b3_m2": {"accumulator_residency": True},
    "b3_m1m2": {"reduce_compute_overlap": True, "accumulator_residency": True},
    "b3_all": {"reduce_compute_overlap": True,
               "electronic_io_compute_overlap": True,
               "activity_gated_static": True},
    "b3_full": {"reduce_compute_overlap": True,
                "accumulator_residency": True,
                "electronic_io_compute_overlap": True,
                "activity_gated_static": True},
    "b4_dma": {"dma_batch_fetch": True},
    "b4_dma_full": {"dma_batch_fetch": True,
                    "reduce_compute_overlap": True,
                    "accumulator_residency": True,
                    "electronic_io_compute_overlap": True,
                    "activity_gated_static": True},
    "b3_a16_full": {"_tops": 0.15,
                    "external_memory_bandwidth_bytes_per_s": 12e9,
                    "scratchpad_bandwidth_bytes_per_s": 50e9,
                    "electronic_static_power_w": 3.5,
                    "external_memory_energy_per_byte_j": 3.5e-10,
                    "reduce_compute_overlap": True,
                    "accumulator_residency": True,
                    "electronic_io_compute_overlap": True,
                    "activity_gated_static": True},
    "b3_a16_off": {"_tops": 0.15,
                   "external_memory_bandwidth_bytes_per_s": 12e9,
                   "scratchpad_bandwidth_bytes_per_s": 50e9,
                   "electronic_static_power_w": 3.5,
                   "external_memory_energy_per_byte_j": 3.5e-10},
    "b3_a16_on": {"_tops": 0.15,
                  "external_memory_bandwidth_bytes_per_s": 12e9,
                  "scratchpad_bandwidth_bytes_per_s": 50e9,
                  "electronic_static_power_w": 3.5,
                  "external_memory_energy_per_byte_j": 3.5e-10,
                  "reduce_compute_overlap": True},
    "b3_a16_m2": {"_tops": 0.15,
                  "external_memory_bandwidth_bytes_per_s": 12e9,
                  "scratchpad_bandwidth_bytes_per_s": 50e9,
                  "electronic_static_power_w": 3.5,
                  "external_memory_energy_per_byte_j": 3.5e-10,
                  "accumulator_residency": True},
    "b3_a16_m1m2": {"_tops": 0.15,
                    "external_memory_bandwidth_bytes_per_s": 12e9,
                    "scratchpad_bandwidth_bytes_per_s": 50e9,
                    "electronic_static_power_w": 3.5,
                    "external_memory_energy_per_byte_j": 3.5e-10,
                    "reduce_compute_overlap": True,
                    "accumulator_residency": True},
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(VARIANTS), default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    names = args.only if args.only else list(VARIANTS)
    out_root = Path(args.out_root) if args.out_root else OUT_ROOT
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    traces = {v: load_manifest_jsonl(TRACE_ROOT / v / "operator_trace.jsonl")
              for v in ["xxs", "xs", "s"]}
    out_root.mkdir(parents=True, exist_ok=True)

    for name in names:
        overrides = VARIANTS[name]
        tops = overrides.get("_tops", 10)
        ma_overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        out = out_root / name
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(
                f"{out} not empty; pass --out-root to write reruns elsewhere")
        out.mkdir(parents=True, exist_ok=True)
        variant_spec = copy.deepcopy(spec)
        variant_spec["batch3_variant"] = {"name": name, "tops": tops, "overrides": ma_overrides}
        variant_spec["user_confirmed"]["main_dense_int8_tops"] = tops
        for key, value in ma_overrides.items():
            variant_spec["mobile_assumptions"][key] = value
        write_json(out / "specification.json", variant_spec)

        cfg = resolve_profile(base, variant_spec, tops, "mobile_reference")
        costs = PhysicalCoreCosts(base)
        effective = profile_costs(costs, cfg)
        write_json(out / "backend.json", effective.summary())

        rows = []
        tick = time.monotonic()
        for variant_key, (manifest, records) in traces.items():
            for mode in MODES:
                run_id = f"{variant_key}__{mode}__main__nominal"
                part = run_case(cfg, effective, mode, manifest, records,
                                out / run_id, run_id, timeline=False)
                rows.extend([dict(profile=name, dense_int8_tops=tops, **r) for r in part])
                warm = part[-1]
                print(json.dumps(dict(variant=name, run_id=run_id,
                                      warm_ms=warm["latency_ms"], warm_mj=warm["energy_mj"])),
                      flush=True)
        write_csv(out / "aggregate.csv", rows)
        write_json(out / "completion.json", dict(
            status="complete", variant=name, overrides=overrides,
            run_count=len(rows) // 3, frames=len(rows),
            wall_time_s=time.monotonic() - tick,
            aggregate_sha256=sha256_file(out / "aggregate.csv")))
        print(json.dumps(dict(variant=name, status="complete",
                              wall_time_s=time.monotonic() - tick)), flush=True)


if __name__ == "__main__":
    main()
