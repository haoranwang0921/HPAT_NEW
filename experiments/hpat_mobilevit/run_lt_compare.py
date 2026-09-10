"""Run a versioned, trace-backed D0/HPAT/LT smoke comparison with full timelines."""
import argparse
import csv
import gzip
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .run import ROOT, REPO, run_case, write_json, write_csv
from .backends import PhysicalCoreCosts
from .lt_backend import LTCoreCosts
from .lt_streaming import LTStreamSimulation
from .streaming import TileStreamSimulation
from .mobile_profile import resolve_profile, profile_costs
from .verify import verify_timeline
from joint_sim.trace_io import load_manifest_jsonl, sha256_file


def verify_lt(path, summary):
    ends, stages = defaultdict(float), {}
    counts, energy = defaultdict(Counter), defaultdict(lambda: defaultdict(float))
    macs, frame_end, accum = Counter(), defaultdict(float), defaultdict(float)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        e = json.loads(line)
        r, kind, f = e["resource"], e["event_type"], e["frame"]
        assert e["start_s"]+1e-14 >= ends[r]
        assert math.isclose(e["end_s"], e["start_s"]+e["duration_s"], abs_tol=1e-14)
        ends[r] = e["end_s"]
        assert kind != "programming"
        if kind in ("dac_encode", "optical_compute", "adc_convert"):
            expected = {"dac_encode": None, "optical_compute": "dac_encode", "adc_convert": "optical_compute"}[kind]
            assert stages.get(r) == expected
            stages[r] = None if kind == "adc_convert" else kind
        key = (f, e["op_id"], e.get("accumulator"))
        if e.get("purpose") == "partial_sum_read":
            assert e["start_s"]+1e-14 >= accum[key]
        elif e.get("purpose") == "partial_sum_write":
            accum[key] = e["end_s"]
        macs[f] += e.get("macs", 0)
        counts[f][kind] += 1
        energy[f][kind] += e["energy_j"]
        frame_end[f] = max(frame_end[f], e["end_s"])
    prior = 0
    for f in summary["frames"]:
        i = f["frame"]
        assert counts[i] == Counter(f["event_counts"])
        assert math.isclose(frame_end[i]-prior, f["latency_s"], rel_tol=1e-9, abs_tol=1e-14)
        prior = frame_end[i]
        for kind, value in energy[i].items():
            assert math.isclose(value, f["energy_breakdown_j"][kind], rel_tol=1e-8, abs_tol=1e-14)
        assert macs[i] == summary["expected_photonic_macs"]
    return sum(sum(c.values()) for c in counts.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["xxs", "xs", "s"], default="xxs")
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--modes", nargs="+", choices=["linear", "linear_pointwise", "linear_pointwise_attention"],
                   default=["linear", "linear_pointwise_attention"])
    args = p.parse_args()
    if args.frames < 1:
        p.error("frames must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT/"config_10ps_fixed_write_energy.json").read_text())
    spec = json.loads((ROOT/"mobile_reference.json").read_text())
    cfg = resolve_profile(base, spec, 10, "mobile_reference")
    cfg["model_assumptions"]["inferences_per_run"] = args.frames
    for key in ("dma_batch_fetch", "electronic_io_compute_overlap", "activity_gated_static",
                "accumulator_residency", "reduce_compute_overlap"):
        cfg["model_assumptions"][key] = False
    trace = REPO/"results/hpat_mobilevit/trace_v1"/args.variant/"operator_trace.jsonl"
    manifest, records = load_manifest_jsonl(trace)
    assert manifest["reference_comparison"]["matching"]
    files = list(ROOT.rglob("*.py"))+list((REPO/"joint_sim").rglob("*.py"))
    files += list((REPO/"SimPhony/onnarchsim").rglob("*.py"))+list((REPO/"SimPhony/configs").rglob("*.yml"))
    files += [REPO/"LLMCompass/hardware_model/energy_model.py"]
    source_hashes = {str(f.relative_to(REPO)): sha256_file(f) for f in sorted(files)}
    write_json(args.output/"config.json", cfg)
    write_json(args.output/"provenance.json", dict(source_sha256=source_hashes,
        trace_path=str(trace), trace_sha256=sha256_file(trace), python=platform.python_version(),
        lt_official_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=REPO/"external/Lightening-Transformer", text=True).strip(),
        command="python -m experiments.hpat_mobilevit.run_lt_compare "+" ".join(__import__("sys").argv[1:]),
        scope="Controlled 4-core 8-bit comparison; same memory/electronic cost model, not equal area or official LT reproduction"))
    all_rows, checks = [], []
    from .streaming import eligible
    for arch, cls, simcls in [("HPAT", PhysicalCoreCosts, TileStreamSimulation), ("LT", LTCoreCosts, LTStreamSimulation)]:
        costs = profile_costs(cls(cfg), cfg)
        write_json(args.output/f"backend_{arch}.json", costs.summary())
        modes = (["digital"] if arch == "HPAT" else []) + list(dict.fromkeys(args.modes))
        for mode in modes:
            run_id = f"{args.variant}_{'D0' if mode == 'digital' else arch}_{mode}"
            out = args.output/run_id
            rows = run_case(cfg, costs, mode, manifest, records, out, run_id, timeline=True, simulation_class=simcls)
            summary = json.loads((out/"summary.json").read_text())
            if arch == "LT":
                summary["expected_photonic_macs"] = sum(r["macs"] for r in records if eligible(r, mode))
                write_json(out/"summary.json", summary)
                n = verify_lt(out/"events.jsonl.gz", summary)
            else:
                n = verify_timeline(out/"events.jsonl.gz", summary)
            for f in summary["frames"]:
                assert math.isclose(sum(f["energy_breakdown_j"].values()), f["energy_j"], rel_tol=1e-10)
                assert math.isclose(f["energy_breakdown_j"]["electronic_static"],
                                    costs.electronic_energy.static_energy(f["latency_s"]), rel_tol=1e-9)
                if mode != "digital":
                    for k, power in costs.static_power.items():
                        assert math.isclose(f["energy_breakdown_j"][k], power*f["latency_s"], rel_tol=1e-9, abs_tol=1e-14)
            for r in rows:
                r.update(architecture="D0" if mode == "digital" else arch,
                         average_power_w=r["energy_mj"]/r["latency_ms"])
            all_rows.extend(rows)
            checks.append(dict(run_id=run_id, events=n, valid=True, timeline_sha256=sha256_file(out/"events.jsonl.gz")))
            print(json.dumps(rows[-1]), flush=True)
        write_json(args.output/f"kernel_costs_{arch}.json", list(costs.cache.values()))
    write_csv(args.output/"aggregate.csv", all_rows)
    # Fail if a source changed during a run; never silently validate mixed code.
    for f, digest in source_hashes.items():
        assert sha256_file(REPO/f) == digest, f
    digest = sha256_file(args.output/"aggregate.csv")
    write_json(args.output/"verification.json", dict(valid=True, cases=checks, aggregate_sha256=digest,
        scope="Resource exclusion, LT stage order/MAC coverage, partial-sum ordering and energy closure; not numerical accuracy"))
    write_json(args.output/"completion.json", dict(status="complete", run_count=len(checks), frames=len(all_rows), aggregate_sha256=digest))


if __name__ == "__main__":
    main()
