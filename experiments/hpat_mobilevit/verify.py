"""Independent checks of serialized experiment reports and physical timelines."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from joint_sim.trace_io import sha256_file
from .run import write_json


def near(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-14):
        raise AssertionError((a, b))


def verify_timeline(path, summary):
    resource_end, core_end, core_state = defaultdict(float), defaultdict(float), {}
    accumulated = defaultdict(float)
    counts = defaultdict(Counter)
    energies = defaultdict(lambda: defaultdict(float))
    frame_end = defaultdict(float)
    events = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            events += 1
            frame, kind = r["frame"], r["event_type"]
            start, end, resource = r["start_s"], r["end_s"], r["resource"]
            near(start+r["duration_s"], end)
            if r["duration_s"] > 0 and start+1e-14 < resource_end[resource]:
                raise AssertionError(("overlap", resource, r))
            resource_end[resource] = max(resource_end[resource], end)
            if kind == "programming":
                core = r["core"]
                if start+1e-14 < core_end[core]:
                    raise AssertionError("Programming overwrites a busy physical core")
                core_state[core] = r["weight_block"]
                core_end[core] = end
            elif kind in {"dac_encode", "optical_compute", "adc_convert"}:
                core = r["core"]
                if core_state.get(core) != r["weight_block"] or start+1e-14 < core_end[core]:
                    raise AssertionError("Photonic access uses an absent or not-ready weight")
                core_end[core] = end
            if r.get("purpose") in {"partial_sum_read", "partial_sum_write"}:
                key = (frame, r["op_id"], r["accumulator"])
                if r["purpose"] == "partial_sum_read" and start+1e-14 < accumulated[key]:
                    raise AssertionError("Read-before-write hazard in partial sums")
                if r["purpose"] == "partial_sum_write":
                    accumulated[key] = end
            counts[frame][kind] += 1
            energies[frame][kind] += r["energy_j"]
            frame_end[frame] = max(frame_end[frame], end)
    prior = 0.0
    for frame in summary["frames"]:
        i = frame["frame"]
        near(frame_end[i]-prior, frame["latency_s"])
        prior = frame_end[i]
        if counts[i] != Counter(frame["event_counts"]):
            raise AssertionError("Serialized event counts do not match summary")
        for kind, energy in energies[i].items():
            near(energy, frame["energy_breakdown_j"][kind])
    if events != summary["total_events"]:
        raise AssertionError("Timeline total differs from report")
    return events


def verify(results, *, full_suite=True):
    complete = json.loads((results/"completion.json").read_text(encoding="utf-8"))
    if complete["status"] != "complete" or sha256_file(results/"aggregate.csv") != complete["aggregate_sha256"]:
        raise AssertionError("Missing or corrupted completion/aggregate")
    with (results/"aggregate.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    provenance = json.loads((results/"provenance.json").read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[2]
    for relative, digest in provenance["source_sha256"].items():
        if sha256_file(repo/relative) != digest:
            raise AssertionError(f"Source changed since execution: {relative}")
    cfg = json.loads((results/"config.json").read_text(encoding="utf-8"))
    expected_runs = 3*4*(1+sum(len(v) for v in cfg["sweeps"].values()))
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    if full_suite and len(by_run) != expected_runs:
        raise AssertionError(("incomplete suite", len(by_run), expected_runs))
    backend = json.loads((results/"backend.json").read_text(encoding="utf-8"))
    total_events = 0
    full_timelines = 0
    for run_id, csv_rows in by_run.items():
        directory = results/run_id
        s = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if len(csv_rows) != len(s["frames"]):
            raise AssertionError("Missing frame in aggregate")
        for r, frame in zip(csv_rows, s["frames"]):
            near(r["latency_ms"], frame["latency_s"]*1e3)
            near(r["energy_mj"], frame["energy_j"]*1e3)
            near(sum(frame["energy_breakdown_j"].values()), frame["energy_j"])
            near(frame["energy_breakdown_j"]["electronic_static"],
                 backend["electronic_energy"]["static_power_w"]*frame["latency_s"])
            if s["mode"] != "digital":
                for kind, power in backend["static_power_w"].items():
                    near(frame["energy_breakdown_j"][kind], power*frame["latency_s"])
            elif "laser_static" in frame["energy_breakdown_j"]:
                raise AssertionError("Digital-only baseline includes absent photonics")
        with (directory/"layers.csv").open(encoding="utf-8") as f:
            layers = list(csv.DictReader(f))
        for frame in s["frames"]:
            part = [r for r in layers if int(r["frame"]) == frame["frame"]]
            if len(part) != s["trace_manifest"]["total_operators"]:
                raise AssertionError("Missing operator cost")
            near(sum(float(r["latency_s"]) for r in part), frame["latency_s"])
            if sum(int(r["macs"]) for r in part) != s["trace_manifest"]["reference_comparison"]["captured_macs"]:
                raise AssertionError("Missing MACs in simulated workload")
        if s["scratchpad_peak_bytes"] > s["effective_config"]["model_assumptions"]["activation_scratchpad_bytes"]:
            raise AssertionError("Scratchpad overflow")
        if s["sram"]["occupied"] > s["sram"]["capacity_tiles"]:
            raise AssertionError("Weight SRAM overflow")
        if (directory/"events.jsonl.gz").exists():
            total_events += verify_timeline(directory/"events.jsonl.gz", s)
            full_timelines += 1
    report = dict(valid=True, runs=len(by_run), frames=len(rows), full_timelines=full_timelines,
                  independently_checked_events=total_events,
                  checks=["source fingerprints", "complete suite", "all operator MACs", "energy closure",
                          "static energy integration", "layer latency closure", "capacity",
                          "serialized resource/weight/partial-sum event dependencies"],
                  scope="Implementation and accounting checks, not device calibration")
    write_json(results/"verification.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--partial-suite", action="store_true")
    args = p.parse_args()
    print(json.dumps(verify(args.results, full_suite=not args.partial_suite), indent=2))
