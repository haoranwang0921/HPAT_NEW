"""Run matched mobile/reference profiles without changing historical experiments."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from joint_sim.trace_io import load_manifest_jsonl, sha256_file
from .backends import PhysicalCoreCosts
from .mobile_profile import resolve_profile, profile_costs
from .run import ROOT, REPO, run_case, write_csv, write_json

MODES = ["digital", "linear", "linear_pointwise", "linear_pointwise_attention"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--specification", type=Path, default=ROOT/"mobile_reference.json")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.specification.read_text(encoding="utf-8"))
    base_path = args.specification.parent/spec["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    peaks = [spec["user_confirmed"]["main_dense_int8_tops"], *spec["user_confirmed"]["comparison_dense_int8_tops"]]
    if len(set(peaks)) != len(peaks):
        raise ValueError("Duplicate peak settings")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"specification.json", spec)
    write_json(args.output/"base_config.json", base)
    source_paths = list(ROOT.rglob("*.py")) + list((REPO/"joint_sim").rglob("*.py"))
    source_paths += list((REPO/"SimPhony/onnarchsim").rglob("*.py"))
    source_paths += list((REPO/"SimPhony/configs").rglob("*.yml"))
    source_paths += [REPO/"LLMCompass/hardware_model/energy_model.py"]
    sources = {str(path.relative_to(REPO)):sha256_file(path) for path in sorted(source_paths)}
    costs = PhysicalCoreCosts(base)
    traces = {v:load_manifest_jsonl(args.traces/v/"operator_trace.jsonl") for v in ["xxs", "xs", "s"]}
    trace_hashes = {v:sha256_file(args.traces/v/"operator_trace.jsonl") for v in traces}
    all_rows, profile_names = [], []
    started = time.monotonic()
    for kind in spec["profiles"]:
        for tops in peaks:
            profile = f"{kind}_{tops:g}tops"
            profile_names.append(profile)
            cfg = resolve_profile(base, spec, tops, kind)
            effective_costs = profile_costs(costs, cfg)
            directory = args.output/profile
            directory.mkdir()
            write_json(directory/"config.json", cfg)
            backend = effective_costs.summary()
            backend["electronic_throughput"] = {k:v for k,v in cfg["model_assumptions"].items()
                                                if k.startswith("electronic_")}
            write_json(directory/"backend.json", backend)
            write_json(directory/"provenance.json", dict(upstream_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), python=platform.python_version(),
                config_sha256=sha256_file(directory/"config.json"), source_sha256=sources,
                specification_sha256=sha256_file(args.specification), base_config_sha256=sha256_file(base_path),
                command="python -m experiments.hpat_mobilevit.run_mobile "+" ".join(__import__("sys").argv[1:]),
                evidence_scope=cfg["scope"]))
            rows, tick = [], time.monotonic()
            for variant, (manifest, records) in traces.items():
                if not manifest["reference_comparison"]["matching"]:
                    raise ValueError("Trace MACs do not match the HPAT reference")
                for mode in MODES:
                    run_id = f"{variant}__{mode}__main__nominal"
                    part = run_case(cfg, effective_costs, mode, manifest, records, directory/run_id,
                                    run_id, timeline=kind == "mobile_reference" and tops == peaks[0])
                    rows.extend(part)
                    all_rows.extend([dict(profile=profile, profile_kind=kind, dense_int8_tops=tops, **r) for r in part])
                    print(json.dumps(dict(profile=profile, run_id=run_id,
                        warm_ms=part[-1]["latency_ms"], warm_mj=part[-1]["energy_mj"])), flush=True)
                    write_csv(directory/"aggregate.csv", rows)
                    write_csv(args.output/"aggregate.csv", all_rows)
            write_json(directory/"kernel_costs.json", list(costs.cache.values()))
            write_json(directory/"completion.json", dict(status="complete", run_count=len(rows)//3,
                wall_time_s=time.monotonic()-tick, aggregate_sha256=sha256_file(directory/"aggregate.csv"),
                traces_sha256=trace_hashes))
    write_json(args.output/"completion.json", dict(status="complete", profiles=profile_names,
        run_count=len(all_rows)//3, frames=len(all_rows), wall_time_s=time.monotonic()-started,
        aggregate_sha256=sha256_file(args.output/"aggregate.csv")))


if __name__ == "__main__":
    main()
