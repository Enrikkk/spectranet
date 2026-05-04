#!/usr/bin/env python3
"""Pareto frontier figure: test L2 vs parameter count (log-x).

Shades the Pareto-frontier rows (excluding Transformer, the full-attention crown).
Annotates the headline canonical-SpectraNet point.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "tables" / "pareto_data.csv"
OUT  = ROOT / "figures" / "pareto.pdf"

rows = []
with CSV.open() as f:
    for r in csv.DictReader(f):
        r["params"]  = int(r["params"])
        r["test_L2"] = float(r["test_L2"])
        rows.append(r)

ours       = [r for r in rows if r["is_ours"] == "Y"]
baselines  = [r for r in rows if r["is_ours"] != "Y"]
transformer = [r for r in baselines if "Transformer" in r["model_short"] and "Galerkin" not in r["model_short"]]
other_base  = [r for r in baselines if r not in transformer]

# Pareto frontier among "ours + baselines without transformer" — sorted by params asc, then keep monotone-decreasing L2
def compute_frontier(rs):
    rs = sorted(rs, key=lambda r: r["params"])
    frontier = []
    best_L2 = float("inf")
    for r in rs:
        if r["test_L2"] < best_L2:
            frontier.append(r)
            best_L2 = r["test_L2"]
    return frontier

# combine ours + non-transformer baselines for the lightweight Pareto frontier
combined_lightweight = [r for r in rows if r not in transformer]
frontier = compute_frontier(combined_lightweight)
frontier_keys = {r["model_short"] for r in frontier}

fig, ax = plt.subplots(figsize=(7.5, 4.8))
# baselines (excluding transformer)
ax.scatter([r["params"] for r in other_base],
           [r["test_L2"] for r in other_base],
           s=42, c="#888", alpha=0.85, edgecolor="black", linewidths=0.5,
           label="Baselines (15 models)", zorder=3)
# ours
ax.scatter([r["params"] for r in ours],
           [r["test_L2"] for r in ours],
           s=70, c="#0072B2", alpha=0.95, edgecolor="black", linewidths=0.7,
           marker="D", label="SpectraNet variants (ours)", zorder=4)
# transformer crown — separate
if transformer:
    ax.scatter([r["params"] for r in transformer],
               [r["test_L2"] for r in transformer],
               s=120, c="#009E73", marker="*", edgecolor="black", linewidths=0.7,
               label="NSL Transformer (full softmax)", zorder=5)

# Pareto frontier line (lightweight, excluding transformer)
fx = [r["params"] for r in frontier]
fy = [r["test_L2"] for r in frontier]
ax.plot(fx, fy, ls="--", color="#0072B2", lw=1.4, alpha=0.65, zorder=2,
        label="Pareto frontier (lightweight)")

# annotate select points
ANNOT = {
    "SpectraNet (ours, headline)":          (12, 8),
    "SpectraNet (ours), w=20 plain":        (12, 6),
    "FNO seed 0":                           (-65, 16),
    "Transformer":                          (-100, -16),
    "U-Net":                                (-50, -18),
    "MWT":                                  (-30, -18),
    "ONO":                                  (12, 4),
}
for r in rows:
    short = r["model_short"]
    for k, (dx, dy) in ANNOT.items():
        if k in short:
            ax.annotate(short, (r["params"], r["test_L2"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

ax.set_xscale("log")
ax.set_xlabel("Parameter count (log scale)")
ax.set_ylabel(r"Test relative $L^2$")
ax.set_title(r"Pareto frontier — NS $\nu{=}10^{-5}$, $64{\times}64$, unified protocol")
ax.set_ylim(0.0, 0.40)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"frontier (lightweight, excl. Transformer): {[r['model_short'] for r in frontier]}")
