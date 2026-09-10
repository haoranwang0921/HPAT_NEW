"""Batch-2: bandwidth-floor + activity-gated static power sweep.

All variants run on the rolled-back mobile_reference profile at 10 TOPS:
  b2_none   both switches off  -> must reproduce the historical ledger bit-exactly
  b2_overlap electronic_io_compute_overlap=True (input fetch || compute)
  b2_gated  activity_gated_static=True (static power on busy windows, laser gating)
  b2_both   both on

Run from repo root:  python -B -m experiments.hpat_mobilevit.run_batch2_gated
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
OUT_ROOT = REPO / "results/hpat_mobilevit/experiments_batch2"

VARIANTS = {
    "b2_none": {},
    "b2_overlap": {"electronic_io_compute_overlap": True},
    "b2_gated": {"activity_gated_static": True},
    "b2_both": {"electronic_io_compute_overlap": True, "activity_gated_static": True},
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(VARIANTS), default=None,
                    help="Run a subset of variants (overwrite existing outputs)")
    ap.add_argument("--out-root", default=None,
                    help="Alternative output root (e.g. for fixed reruns)")
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
        out = out_root / name
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(
                f"{out} not empty; pass --out-root to write fixed reruns elsewhere")
        out.mkdir(parents=True, exist_ok=True)
        variant_spec = copy.deepcopy(spec)
        variant_spec["batch2_variant"] = {"name": name, "overrides": overrides}
        for key, value in overrides.items():
            variant_spec["mobile_assumptions"][key] = value
        write_json(out / "specification.json", variant_spec)

        tops = variant_spec["user_confirmed"]["main_dense_int8_tops"]
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
