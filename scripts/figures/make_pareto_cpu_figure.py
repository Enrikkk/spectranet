#!/usr/bin/env python3
"""CPU-latency Pareto figure: median CPU inference latency at B=1 vs test L2.

Companion to make_pareto_figure.py (the GPU/params Pareto). Reads
TIME_ANALYSIS/results_cpu/leaderboard_speed.csv (median latency) and produces a
log-x scatter highlighting the SpectraNet variants and the Transformer "crown" point.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT.parent / "TIME_ANALYSIS" / "results_cpu" / "leaderboard_speed.csv"
OUT  = ROOT / "figures" / "pareto_cpu.pdf"

OURS_PREFIX = "SpectraNet"
TRANSFORMER_NAMES = {"Transformer"}

rows = []
with SRC.open() as f:
    for r in csv.DictReader(f):
        if int(r["batch_size"]) != 1:
            continue
        rows.append({
            "name":   r["display"],
            "params": int(r["params"]),
            "L2":     float(r["test_l2"]),
            "lat":    float(r["latency_ms_median"]),
        })

ours        = [r for r in rows if r["name"].startswith(OURS_PREFIX)]
transformer = [r for r in rows if r["name"] in TRANSFORMER_NAMES]
others      = [r for r in rows if r not in ours and r not in transformer]

# Pareto frontier: lightweight (no transformer crown), sorted by latency asc, monotone L2.
def compute_frontier(rs):
    rs = sorted(rs, key=lambda r: r["lat"])
    out = []
    best = float("inf")
    for r in rs:
        if r["L2"] < best:
            out.append(r)
            best = r["L2"]
    return out

frontier = compute_frontier(ours + others)

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.scatter([r["lat"] for r in others], [r["L2"] for r in others],
           s=42, c="#888", alpha=0.85, edgecolor="black", linewidths=0.5,
           label="Baselines", zorder=3)
ax.scatter([r["lat"] for r in ours], [r["L2"] for r in ours],
           s=70, c="#0072B2", alpha=0.95, edgecolor="black", linewidths=0.7,
           marker="D", label="SpectraNet variants (ours)", zorder=4)
if transformer:
    ax.scatter([r["lat"] for r in transformer], [r["L2"] for r in transformer],
               s=140, c="#009E73", marker="*", edgecolor="black", linewidths=0.7,
               label="NSL Transformer (full softmax)", zorder=5)

# frontier line
fx = [r["lat"] for r in frontier]; fy = [r["L2"] for r in frontier]
ax.plot(fx, fy, ls="--", color="#0072B2", lw=1.4, alpha=0.7, zorder=2,
        label="Pareto frontier (excl. Transformer)")

ANNOT = {
    "SpectraNet (w=32 (canonical))": (12, -16),
    "SpectraNet (w=20)":             (10, 6),
    "FNO":                            (10, 6),
    "Transformer":                    (-180, 6),
    "U_Net":                          (10, -14),
    "MWT":                            (-12, -18),
    "ONO":                            (10, 6),
}
for r in rows:
    for k, (dx, dy) in ANNOT.items():
        if r["name"] == k or (k == "Transformer" and r["name"] == "Transformer"):
            ax.annotate(r["name"], (r["lat"], r["L2"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
            break

ax.set_xscale("log")
ax.set_xlabel("CPU latency at $B{=}1$ (ms, log scale; i5-1155G7)")
ax.set_ylabel(r"Test relative $L^2$")
ax.set_title(r"CPU deployment Pareto — NS $\nu{=}10^{-5}$, $64{\times}64$")
ax.set_ylim(0.0, 0.40)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"frontier (excl. Transformer): {[r['name'] for r in frontier]}")
