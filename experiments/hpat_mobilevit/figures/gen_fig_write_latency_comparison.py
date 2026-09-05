"""Compare verified 1 us and 10 ps main results at unchanged per-write energy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .gen_fig_evaluation import MODE_ORDER, MODELS, SHORT, read_rows, unique, styling, save
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(before, after):
    old, new = read_rows(before), read_rows(after)
    old_cfg = json.loads((before/"config.json").read_text(encoding="utf-8"))
    new_cfg = json.loads((after/"config.json").read_text(encoding="utf-8"))
    assert old_cfg["user_confirmed"]["program_response_time_s"] == 1e-6
    assert new_cfg["user_confirmed"]["program_response_time_s"] == 10e-12
    for section in ["user_confirmed", "model_assumptions"]:
        for key, value in old_cfg[section].items():
            if key in {"date", "program_response_time_s", "program_energy_multiplier"}:
                continue
            assert new_cfg[section][key] == value, ("Unmatched configuration", section, key)
    rows = []
    for model in MODELS:
        for mode in MODE_ORDER:
            for frame in range(3):
                a = unique(old, model=model, mode=mode, axis="main", frame=frame)
                b = unique(new, model=model, mode=mode, axis="main", frame=frame)
                assert a["programming_events"] == b["programming_events"]
                summaries = [json.loads((root/r["run_id"]/"summary.json").read_text(encoding="utf-8"))
                             for root, r in [(before, a), (after, b)]]
                manifests = [s["trace_manifest"] for s in summaries]
                assert manifests[0] == manifests[1], "Workload changed"
                energies = [s["frames"][frame]["energy_breakdown_j"].get("programming", 0.0)
                            for s in summaries]
                assert math.isclose(*energies, rel_tol=1e-11, abs_tol=1e-18), "Write energy changed"
                if mode == "digital":
                    for key in ["latency_ms", "energy_mj"]:
                        assert a[key] == b[key], "Digital baseline changed"
                rows.append(dict(model=model, mode=mode, frame=frame,
                    old_latency_ms=a["latency_ms"], new_latency_ms=b["latency_ms"],
                    old_energy_mj=a["energy_mj"], new_energy_mj=b["energy_mj"],
                    latency_reduction_percent=100*(1-b["latency_ms"]/a["latency_ms"]),
                    energy_reduction_percent=100*(1-b["energy_mj"]/a["energy_mj"]),
                    programming_events=int(b["programming_events"]),
                    old_programming_energy_j=energies[0], new_programming_energy_j=energies[1]))
    return rows


def plot(rows, output, frame):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.7))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.865, bottom=0.17, hspace=0.4, wspace=0.34)
    colors, hatches = ["#0072B2", "#E69F00"], ["//", ""]
    for col, model in enumerate(MODELS):
        part = [unique(rows, model=model, mode=mode, frame=frame) for mode in MODE_ORDER]
        for line, metric in enumerate(["latency_ms", "energy_mj"]):
            ax = axes[line, col]
            for i, prefix in enumerate(["old", "new"]):
                values = [r[f"{prefix}_{metric}"] for r in part]
                bars = ax.bar(np.arange(4)+(i-0.5)*0.36, values, width=0.33,
                              color=colors[i], hatch=hatches[i], edgecolor="white", linewidth=0.4)
                for bar, value in zip(bars, values):
                    ax.annotate(f"{value:.2f}" if value < 10 else f"{value:.1f}",
                        (bar.get_x()+bar.get_width()/2, value), xytext=(0, 2),
                        textcoords="offset points", ha="center", rotation=90, fontsize=7)
            ax.set_xticks(np.arange(4), SHORT)
            if col == 0:
                ax.set_ylabel("Latency (ms)" if line == 0 else "Energy / image (mJ)")
            ax.set_title(f"({chr(97+line*3+col)}) {model}", loc="left", fontsize=9)
            ax.set_ylim(0, ax.get_ylim()[1]*1.27)
            ax.grid(axis="y")
    fig.legend([Patch(facecolor=c, hatch=h, edgecolor="white") for c, h in zip(colors, hatches)],
               ["Previous: 1 μs / block", "Updated: 10 ps / block"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    state = "Cold, image 1" if frame == 0 else "Warmed, image 3"
    fig.text(0.5, 0.10, "D0 Digital   H1 Linear   H2 +1×1 Conv   H3 +QKᵀ/AV", ha="center", fontsize=8)
    save(fig, output, "fig_write_latency_"+("cold" if frame == 0 else "warm"),
         f"{state}. Per-write energy unchanged. 10 ps is a user-specified whole-array assumption.\n"
         "Four physical 16×16 cores. All other parameters unchanged; no hardware validation implied.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = comparison(args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    styling()
    for frame in [0, 2]:
        plot(rows, args.output, frame)
    meta = dict(before=str(args.before.resolve()), after=str(args.after.resolve()),
        before_aggregate_sha256=digest(args.before/"aggregate.csv"),
        after_aggregate_sha256=digest(args.after/"aggregate.csv"),
        script_sha256=digest(Path(__file__)),
        shared_plot_helpers_sha256=digest(Path(__file__).with_name("gen_fig_evaluation.py")),
        matched_main_frames=len(rows), per_frame_write_energy_preserved=True,
        digital_baseline_unchanged=True, all_other_hardware_parameters_unchanged=True,
        outputs={f.name:digest(f) for f in sorted(args.output.iterdir()) if f.is_file()})
    (args.output/"comparison_verification.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
