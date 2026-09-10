"""Verify all mobile profiles, matching configs and independent event checks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from joint_sim.electronic_backend import ElectronicBackend
from joint_sim.trace_io import sha256_file
from dataclasses import asdict
from .mobile_profile import resolve_profile
from .run import write_json
from .verify import verify, near


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    root = args.results
    complete = json.loads((root/"completion.json").read_text())
    if complete["status"] != "complete" or sha256_file(root/"aggregate.csv") != complete["aggregate_sha256"]:
        raise AssertionError("Missing/corrupt mobile completion")
    spec = json.loads((root/"specification.json").read_text())
    base = json.loads((root/"base_config.json").read_text())
    peaks = [spec["user_confirmed"]["main_dense_int8_tops"], *spec["user_confirmed"]["comparison_dense_int8_tops"]]
    expected = {f"{kind}_{tops:g}tops":(kind, tops) for kind in spec["profiles"] for tops in peaks}
    assert len(complete["profiles"]) == len(expected) and set(complete["profiles"]) == set(expected)
    nominal_energy = asdict(ElectronicBackend().energy_model)
    with (root/"aggregate.csv").open() as f:
        top_rows = list(csv.DictReader(f))
    assert len({(r["profile"],r["run_id"],r["frame"]) for r in top_rows}) == len(top_rows)
    reports = []
    for name, (kind, tops) in expected.items():
        directory = root/name
        cfg = json.loads((directory/"config.json").read_text())
        assert cfg == resolve_profile(base, spec, tops, kind), "Effective profile differs from specification"
        backend = json.loads((directory/"backend.json").read_text())
        energy = {**nominal_energy, **cfg["model_assumptions"]["electronic_energy_overrides"]}
        assert backend["electronic_energy"] == energy
        near(cfg["model_assumptions"]["electronic_peak_flops"], tops*1e12*spec["compute_utilization"])
        for summary_path in directory.glob("*__main__nominal/summary.json"):
            summary = json.loads(summary_path.read_text())
            assert summary["effective_config"] == cfg, "D0 and hybrid parameters differ"
        with (directory/"aggregate.csv").open() as f:
            rows = list(csv.DictReader(f))
        selected = [{k:v for k,v in r.items() if k not in {"profile", "profile_kind", "dense_int8_tops"}}
                    for r in top_rows if r["profile"] == name]
        assert selected == rows, "Top-level data differs from profile"
        report = verify(directory)
        reports.append(dict(profile=name, **report))
        print(json.dumps(dict(profile=name, valid=True)), flush=True)
    assert sum(r["frames"] for r in reports) == len(top_rows) == complete["frames"]
    assert sum(r["runs"] for r in reports) == complete["run_count"]
    result = dict(valid=True, profiles=reports, frames=len(top_rows), runs=complete["run_count"],
        aggregate_sha256=sha256_file(root/"aggregate.csv"),
        matched_parameters=True, source_of_truth="specification.json plus base_config.json",
        scope="Accounting/implementation validation only, not phone calibration")
    write_json(root/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
