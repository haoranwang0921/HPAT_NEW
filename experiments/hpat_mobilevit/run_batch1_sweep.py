"""Batch-1 parameter sweep on the rolled-back mobile_reference profile.

Variants (all on mobile_reference 10 TOPS, 10ps write ring):
  b1_vpf050    vector_peak_fraction 0.1 -> 0.5
  b1_vpf100    vector_peak_fraction 0.1 -> 1.0
  b1_mem_boost vpf 1.0 + LPDDR 64->128 GB/s + scratchpad 256->512 GB/s

Baseline for comparison: results/hpat_mobilevit/experiments_mobile_v1/mobile_reference_10tops.
Run from repo root:  python -B -m experiments.hpat_mobilevit.run_batch1_sweep
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
OUT_ROOT = REPO / "results/hpat_mobilevit/experiments_batch1"

VARIANTS = {
    "b1_vpf050": {"vector_peak_fraction": 0.5},
    "b1_vpf100": {"vector_peak_fraction": 1.0},
    "b1_mem_boost": {
        "vector_peak_fraction": 1.0,
        "external_memory_bandwidth_bytes_per_s": 128e9,
        "scratchpad_bandwidth_bytes_per_s": 512e9,
    },
}


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    traces = {v: load_manifest_jsonl(TRACE_ROOT / v / "operator_trace.jsonl")
              for v in ["xxs", "xs", "s"]}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, overrides in VARIANTS.items():
        out = OUT_ROOT / name
        if out.exists():
            raise FileExistsError(f"{out} already exists; remove it before rerunning")
        out.mkdir()
        variant_spec = copy.deepcopy(spec)
        variant_spec["batch1_variant"] = {"name": name, "overrides": overrides}
        for key, value in overrides.items():
            variant_spec["mobile_assumptions"][key] = value
        write_json(out / "specification.json", variant_spec)

        cfg = resolve_profile(base, variant_spec, 10, "mobile_reference")
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
                rows.extend([dict(profile=name, dense_int8_tops=10, **r) for r in part])
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
