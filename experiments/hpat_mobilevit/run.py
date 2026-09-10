"""Run reproducible MobileViT joint-simulator experiments and parameter sweeps."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from joint_sim.trace_io import load_manifest_jsonl, sha256_file
from .backends import PhysicalCoreCosts
from .streaming import TileStreamSimulation, MODES

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(config, costs, mode, manifest, records, output, run_id, axis="main", value="nominal", timeline=False,
             simulation_class=TileStreamSimulation):
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(output/"events.jsonl.gz", "wt", encoding="utf-8") if timeline else _null_file() as f:
        sink = (lambda row: f.write(json.dumps(row, ensure_ascii=False)+"\n")) if timeline else None
        sim = simulation_class(config, costs, mode, event_sink=sink)
        result = sim.run(records, config["model_assumptions"]["inferences_per_run"])
    result.update(run_id=run_id, model=manifest["model"], axis=axis, sweep_value=value,
                  effective_config=config, trace_manifest=manifest)
    write_json(output/"summary.json", result)
    write_csv(output/"layers.csv", sim.layer_rows)
    rows = []
    for frame in result["frames"]:
        rows.append(dict(run_id=run_id, model=manifest["model"], mode=mode, axis=axis,
                         sweep_value=value, frame=frame["frame"], state=frame["state"],
                         latency_ms=frame["latency_s"]*1e3, energy_mj=frame["energy_j"]*1e3,
                         sram_hit_rate=frame["sram_hit_rate"],
                         programming_events=frame["event_counts"].get("programming", 0)))
    write_csv(output/"frames.csv", rows)
    write_csv(output/"energy.csv", [dict(frame=f["frame"], component=k, energy_j=v)
                                   for f in result["frames"] for k, v in f["energy_breakdown_j"].items()])
    return rows


class _null_file:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"config.json")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", choices=["main", "sweep", "all"], default="main")
    p.add_argument("--variants", nargs="+", default=["xxs", "xs", "s"])
    p.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["digital", "linear", "linear_pointwise", "linear_pointwise_attention"])
    p.add_argument("--timeline", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"config.json", cfg)
    # All source and device/architecture files affecting this integration are
    # pinned. Hashes describe the current clone including our local adapters.
    source_paths = list(ROOT.rglob("*.py")) + list((REPO/"joint_sim").rglob("*.py"))
    source_paths += list((REPO/"SimPhony/onnarchsim").rglob("*.py"))
    source_paths += list((REPO/"SimPhony/configs").rglob("*.yml"))
    source_paths += [REPO/"LLMCompass/hardware_model/energy_model.py"]
    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in sorted(source_paths)}
    costs = PhysicalCoreCosts(cfg)
    write_json(args.output/"backend.json", costs.summary())
    write_json(args.output/"provenance.json", dict(
        upstream_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        python=platform.python_version(), config_sha256=sha256_file(args.config), source_sha256=source_hashes,
        command="python -m experiments.hpat_mobilevit.run " + " ".join(__import__("sys").argv[1:]),
        evidence_scope=cfg["scope"]))
    cases = []
    if args.suite in {"main", "all"}:
        cases.append(("main", "nominal", cfg))
    if args.suite in {"sweep", "all"}:
        for axis, values in cfg["sweeps"].items():
            for value in values:
                changed = copy.deepcopy(cfg)
                section = "user_confirmed" if axis == "program_response_time_s" else "model_assumptions"
                changed[section][axis] = value
                cases.append((axis, value, changed))
    aggregate = []
    started = time.monotonic()
    for variant in args.variants:
        path = args.traces/variant/"operator_trace.jsonl"
        manifest, records = load_manifest_jsonl(path)
        if not manifest["reference_comparison"]["matching"]:
            raise ValueError("MAC disagreement with reused HPAT reference must be reviewed before experiments")
        for axis, value, effective in cases:
            for mode in args.modes:
                run_id = f"{variant}__{mode}__{axis}__{value}"
                rows = run_case(effective, costs, mode, manifest, records, args.output/run_id,
                                run_id, axis, value, timeline=args.timeline and axis == "main")
                aggregate.extend(rows)
                print(json.dumps(dict(run_id=run_id, cold_ms=rows[0]["latency_ms"],
                                      warm_ms=rows[-1]["latency_ms"], warm_mj=rows[-1]["energy_mj"])), flush=True)
                write_csv(args.output/"aggregate.csv", aggregate)
    write_json(args.output/"kernel_costs.json", list(costs.cache.values()))
    write_json(args.output/"completion.json", dict(status="complete", run_count=len(aggregate)//cfg["model_assumptions"]["inferences_per_run"],
        wall_time_s=time.monotonic()-started, aggregate_sha256=sha256_file(args.output/"aggregate.csv"),
        traces_sha256={v: sha256_file(args.traces/v/"operator_trace.jsonl") for v in args.variants}))


if __name__ == "__main__":
    main()
