"""Data-only plots for verified mobile-class profiles and compute-only controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .gen_fig_evaluation import MODE_ORDER, MODELS, LABELS, COLORS, MARKERS, STYLES, HATCHES, styling, save, unique
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_mobile_rows(root):
    verified = json.loads((root/"verification.json").read_text())
    complete = json.loads((root/"completion.json").read_text())
    actual = digest(root/"aggregate.csv")
    if not verified.get("valid") or actual != verified["aggregate_sha256"] or actual != complete["aggregate_sha256"]:
        raise ValueError("Unverified or changed mobile data")
    with (root/"aggregate.csv").open() as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        key = r["profile"], r["run_id"], r["frame"]
        if key in seen:
            raise ValueError("Duplicate profile/run/frame")
        seen.add(key)
        for field in ["latency_ms", "energy_mj", "dense_int8_tops"]:
            r[field] = float(r[field])
            if not np.isfinite(r[field]) or r[field] <= 0:
                raise ValueError(f"Invalid {field}")
        r["frame"] = int(r["frame"])
    if len(rows) != complete["frames"]:
        raise ValueError("Missing mobile frames")
    return rows


def main_point(rows, spec, output):
    tops = spec["user_confirmed"]["main_dense_int8_tops"]
    profile = f"mobile_reference_{tops:g}tops"
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.17, hspace=0.4, wspace=0.27)
    x = np.arange(3)
    for row, frame in enumerate([0, 2]):
        for col, metric in enumerate(["latency_ms", "energy_mj"]):
            ax = axes[row, col]
            for i, mode in enumerate(MODE_ORDER):
                values = [unique(rows, profile=profile, model=model, mode=mode, frame=frame)[metric] for model in MODELS]
                bars = ax.bar(x+(i-1.5)*0.19, values, width=0.175,
                    color=COLORS[i], hatch=HATCHES[i], edgecolor="white", linewidth=0.4)
                for bar, value in zip(bars, values):
                    ax.annotate(f"{value:.2f}" if value < 10 else f"{value:.1f}",
                        (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", rotation=90, fontsize=7)
            ax.set_xticks(x, ["XXS", "XS", "S"])
            ax.set_ylabel("Latency (ms)" if col == 0 else "Energy / image (mJ)")
            ax.set_title(f"({chr(97+2*row+col)}) "+("Cold, image 1" if frame == 0 else "Warmed, image 3"), loc="left")
            ax.set_ylim(0, ax.get_ylim()[1]*1.25)
            ax.grid(axis="y")
    fig.legend([Patch(facecolor=c, hatch=h, edgecolor="white") for c,h in zip(COLORS,HATCHES)],
               LABELS, loc="upper center", ncol=4, bbox_to_anchor=(0.5,0.99))
    a = spec["mobile_assumptions"]
    save(fig, output, "fig_mobile_main", f"{tops:g} dense INT8 TOPS; {a['external_memory_bandwidth_bytes_per_s']/1e9:g} GB/s external DRAM; "
         f"{a['electronic_static_power_w']:g} W electronic static power.\n"
         "10 ps writes at unchanged event energy. Nominal mobile-class reference, not a measured phone.")


def scale(rows, spec, output, kind):
    peaks = sorted([spec["user_confirmed"]["main_dense_int8_tops"], *spec["user_confirmed"]["comparison_dense_int8_tops"]])
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.17, hspace=0.32, wspace=0.30)
    for col, model in enumerate(MODELS):
        for row, metric in enumerate(["latency_ms", "energy_mj"]):
            ax = axes[row, col]
            for i, mode in enumerate(MODE_ORDER[1:], 1):
                values = []
                for tops in peaks:
                    profile = f"{kind}_{tops:g}tops"
                    numerator = unique(rows, profile=profile, model=model, mode=mode, frame=2)[metric]
                    denominator = unique(rows, profile=profile, model=model, mode="digital", frame=2)[metric]
                    values.append(numerator/denominator)
                ax.plot(peaks, values, color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i])
            ax.axhline(1, color="#666666", linestyle=":", linewidth=1)
            ax.set_xticks(peaks)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y")
            if row == 0:
                ax.set_title(model, fontsize=9)
            else:
                ax.set_xlabel("Dense INT8 peak (TOPS)")
            if col == 0:
                ax.set_ylabel("Latency / D0" if row == 0 else "Energy / D0")
    fig.legend([Line2D([], [], color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i]) for i in range(1,4)],
               LABELS[1:], loc="upper center", ncol=3, bbox_to_anchor=(0.5,0.99))
    context = ("Mobile-class memory and power assumptions. Not a measured phone."
               if kind == "mobile_reference" else "Compute-only control: old 512 GB/s and 15 W retained; not a phone configuration.")
    save(fig, output, "fig_"+kind+"_compute_scale",
         "Warmed image 3. Ratios above 1 are worse; each point uses a matched D0.\n"+context)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = read_mobile_rows(args.results)
    spec = json.loads((args.results/"specification.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    styling()
    main_point(rows, spec, args.output)
    for kind in spec["profiles"]:
        scale(rows, spec, args.output, kind)
    manifest = dict(source_results=str(args.results.resolve()), input_sha256=digest(args.results/"aggregate.csv"),
        specification_sha256=digest(args.results/"specification.json"), script_sha256=digest(Path(__file__)),
        shared_plot_helpers_sha256=digest(Path(__file__).with_name("gen_fig_evaluation.py")),
        outputs={p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        notes="All mapping modes retained; no smoothing or inferred confidence intervals.")
    (args.output/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
