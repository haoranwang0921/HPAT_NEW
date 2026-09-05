"""Data-only publication figures for HPAT's verified joint-simulator outputs.

No hardcoded results, smoothing, inferred error bars, or omitted bad cases.
Exports vector PDF/SVG and 400-dpi PNG. The warmed view is image 3, not a
statistical estimate or an assertion of asymptotic steady-state convergence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODE_ORDER = ["digital", "linear", "linear_pointwise", "linear_pointwise_attention"]
LABELS = ["D0 Digital", "H1 Linear", "H2 +1x1 Conv", "H3 +Attention"]
SHORT = ["D0", "H1", "H2", "H3"]
COLORS = ["#777777", "#0072B2", "#E69F00", "#009E73"]
MARKERS = ["o", "s", "^", "D"]
STYLES = ["-", "-", "--", "-."]
HATCHES = ["", "//", "\\\\", "xx"]
MODELS = ["MobileViT-XXS", "MobileViT-XS", "MobileViT-S"]


def styling():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.axisbelow": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
        "lines.linewidth": 1.45, "lines.markersize": 4, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 400})


def read_rows(results):
    verification = json.loads((results/"verification.json").read_text(encoding="utf-8"))
    if not verification.get("valid"):
        raise ValueError("Only verified results may be plotted")
    completion = json.loads((results/"completion.json").read_text(encoding="utf-8"))
    if hashlib.sha256((results/"aggregate.csv").read_bytes()).hexdigest() != completion["aggregate_sha256"]:
        raise ValueError("The aggregate data changed after the experiment completed")
    with (results/"aggregate.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        key = (r["run_id"], r["frame"])
        if key in seen:
            raise ValueError("Duplicate experiment frame")
        seen.add(key)
        for field in ["latency_ms", "energy_mj", "sram_hit_rate"]:
            r[field] = float(r[field])
            if not np.isfinite(r[field]) or r[field] < 0:
                raise ValueError(f"Invalid {field}")
        r["frame"] = int(r["frame"])
    return rows


def unique(rows, **conditions):
    selected = [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected[0]


def save(fig, output, name, footnote):
    fig.text(0.5, 0.018, footnote, ha="center", va="bottom", fontsize=7.5, color="#444444")
    fig.canvas.draw()
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(output/f"{name}.{ext}", bbox_inches="tight", pad_inches=0.055, facecolor="white")
    plt.close(fig)


def main_comparison(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.875, bottom=0.13, hspace=0.38, wspace=0.27)
    x = np.arange(3)
    for row, frame in enumerate([0, 2]):
        for col, metric in enumerate(["latency_ms", "energy_mj"]):
            ax = axes[row, col]
            for i, mode in enumerate(MODE_ORDER):
                values = [unique(rows, model=m, mode=mode, axis="main", frame=frame)[metric] for m in MODELS]
                bars = ax.bar(x+(i-1.5)*0.19, values, width=0.175, color=COLORS[i],
                              hatch=HATCHES[i], edgecolor="white", linewidth=0.4)
                for bar, value in zip(bars, values):
                    ax.annotate(f"{value:.2f}" if value < 10 else f"{value:.1f}",
                                (bar.get_x()+bar.get_width()/2, value), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=7, rotation=90)
            ax.set_xticks(x, ["XXS", "XS", "S"])
            ax.set_ylabel("Latency (ms)" if col == 0 else "Energy / image (mJ)")
            ax.set_title(f"({chr(97+2*row+col)}) " + ("Cold, image 1" if frame == 0 else "Warmed, image 3"), loc="left")
            ax.set_ylim(0, ax.get_ylim()[1]*1.23)
            ax.grid(axis="y")
    fig.legend([Patch(facecolor=c, hatch=h, edgecolor="white") for c, h in zip(COLORS, HATCHES)],
               LABELS, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.98), columnspacing=1.15)
    save(fig, output, "fig01_latency_energy", "4 physical 16x16 cores; signed row pairing; 1 us writing. Nominal model, not silicon measurements.")


def energy_breakdown(results, rows, output):
    groups = {
        "Electronic static": ["electronic_static"], "HBM": ["hbm"],
        "SRAM + NoC": ["sram_noc"], "Electronic compute": ["electronic_dynamic", "signed_reduce"],
        "Photonic datapath": ["optical_compute"], "Writing": ["programming"],
        "Laser + hold": ["laser_static", "mrr_hold", "photonic_periphery_static"],
    }
    colors = ["#999999", "#0072B2", "#56B4E9", "#CC79A7", "#009E73", "#E69F00", "#D55E00"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.85), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.71, bottom=0.20, wspace=0.12)
    for j, model in enumerate(MODELS):
        data = []
        for mode in MODE_ORDER:
            r = unique(rows, model=model, mode=mode, axis="main", frame=2)
            summary = json.loads((results/r["run_id"]/"summary.json").read_text(encoding="utf-8"))
            data.append(summary["frames"][2]["energy_breakdown_j"])
        base = np.zeros(4)
        for (label, keys), color, hatch in zip(groups.items(), colors, ["", "//", "\\\\", "..", "xx", "--", "++"]):
            values = np.array([100*sum(d.get(k, 0) for k in keys)/sum(d.values()) for d in data])
            axes[j].bar(np.arange(4), values, bottom=base, color=color, edgecolor="white", linewidth=0.45, label=label, hatch=hatch)
            base += values
        np.testing.assert_allclose(base, 100, atol=1e-8)
        axes[j].set_xticks(np.arange(4), SHORT)
        axes[j].set_title(model)
        axes[j].set_ylim(0, 100)
        axes[j].grid(axis="y")
    axes[0].set_ylabel("Energy share (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.52, 1.0), columnspacing=1.0)
    save(fig, output, "fig02_energy_breakdown", "Warmed image 3. Absolute energies are in Fig. 1; all present hardware domains remain powered on.")


def sensitivity(rows, output, axis, xlabel, factor, name, log=False):
    selected = [r for r in rows if r["axis"] == axis and r["frame"] == 2]
    values = sorted({float(r["sweep_value"]) for r in selected})
    if len(values) < 2:
        raise ValueError(f"Incomplete sweep: {axis}")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.65))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.165, wspace=0.27, hspace=0.29)
    for col, model in enumerate(MODELS):
        for row, metric in enumerate(["latency_ms", "energy_mj"]):
            ax = axes[row, col]
            ax.axhline(1, color="#555555", linewidth=0.8, linestyle=":")
            for i, mode in enumerate(MODE_ORDER[1:], 1):
                ratio = []
                for value in values:
                    matched = [r for r in selected if float(r["sweep_value"]) == value]
                    a = unique(matched, model=model, mode=mode)
                    b = unique(matched, model=model, mode="digital")
                    ratio.append(a[metric]/b[metric])
                ax.plot(np.array(values)*factor, ratio, color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i])
            if log:
                ax.set_xscale("log")
            ax.set_xticks(np.array(values)*factor, [f"{v*factor:g}" for v in values])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(model)
            else:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("Latency / digital" if row == 0 else "Energy / digital")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y")
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i]) for i in range(1, 4)]
    fig.legend(handles, LABELS[1:], loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.99))
    save(fig, output, name, "Warmed image 3. Each point uses a matched digital configuration; ratios above 1 are worse.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    styling()
    rows = read_rows(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    main_comparison(rows, args.output)
    energy_breakdown(args.results, rows, args.output)
    specs = [
        ("program_response_time_s", "Write time (us)", 1e6, "fig03_programming_time", True),
        ("sram_capacity_bytes", "Weight SRAM (MiB)", 1/2**20, "fig04_weight_sram", True),
        ("hbm_bandwidth_bytes_per_s", "HBM bandwidth (GB/s)", 1e-9, "fig05_memory_bandwidth", True),
        ("program_parallelism", "Writing lanes", 1, "fig06_programming_parallelism", False),
        ("program_energy_multiplier", "Write-energy multiplier", 1, "fig07_programming_energy", False),
    ]
    for spec in specs:
        sensitivity(rows, args.output, *spec)
    outputs = sorted(args.output.glob("fig*.*"))
    manifest = {"source_results": str(args.results.resolve()), "input_sha256": hashlib.sha256((args.results/"aggregate.csv").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib": matplotlib.__version__, "formats": ["pdf", "svg", "png"],
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                "notes": "Deterministic model sweeps; no confidence intervals or smoothing. All methods retained."}
    (args.output/"figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(outputs), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
