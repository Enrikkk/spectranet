#!/usr/bin/env python3
"""Make the long-horizon stability figure (the new headline visual).

Two panels:
  (a) blow-up fraction vs T for T in {10, 20, 50, 100} — fraction of test
      trajectories that diverge (NaN or exceed the energy threshold) by step T
  (b) energy ‖u_t‖^2 vs T (log-y) — shows FNO blowup from T=20 to T=50

Inputs: results/long_horizon.csv
Output: figures/long_horizon.pdf

CSV note: there are two unlabelled `ar2d` row-blocks. By order of submission in
run_eval_long_horizon.sh, the first block is canonical SpectraNet (headline) and
the second is the SpectraNet w=48 plain ablation. We pick the first block for
the headline plot and include the second as a faded line for completeness.
"""
import csv
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]  # paper/
CSV  = ROOT / "tables" / "long_horizon.csv"
OUT  = ROOT / "figures" / "long_horizon.pdf"

with CSV.open() as f:
    rows = list(csv.DictReader(f))

# group rows: each model produces a contiguous block of 4 rows (T = 10, 20, 50, 100).
# CSV has TWO ar2d blocks (first = w32+resi, second = w48 plain), then fno, then transformer.
def to_float(s):
    return float("nan") if s.strip().lower() == "nan" else float(s)

blocks = []  # list of dicts: {label, color, T, mean_L2, mean_energy, blowup}
cur = None
prev_T = None
ar2d_seen = 0
LABELS = {
    "ar2d_0":     ("SpectraNet (ours)",               "#0072B2", "-"),
    "ar2d_1":     ("SpectraNet w=48 plain (ours)",    "#56B4E9", "--"),
    "fno":         ("FNO (NSL baseline)",             "#D55E00", "-"),
    "transformer": ("Transformer (NSL baseline)",     "#009E73", "-"),
}
for r in rows:
    m = r["model"]
    T = int(r["T"])
    if cur is None or (prev_T is not None and T <= prev_T):
        # new block
        if m == "ar2d":
            key = f"ar2d_{ar2d_seen}"
            ar2d_seen += 1
        else:
            key = m
        cur = {"key": key, "T": [], "mean_L2": [], "energy": [], "blowup": []}
        blocks.append(cur)
    cur["T"].append(T)
    cur["mean_L2"].append(to_float(r["mean_L2"]))
    cur["energy"].append(to_float(r["mean_energy"]))
    cur["blowup"].append(to_float(r["blowup_frac"]))
    prev_T = T

fig, (axL, axE) = plt.subplots(1, 2, figsize=(11, 4.0))

for blk in blocks:
    label, color, ls = LABELS[blk["key"]]
    Ts = blk["T"]
    E  = blk["energy"]
    BL = blk["blowup"]

    # panel (a): blow-up fraction vs T (staircase)
    BL_clean = [v if not math.isnan(v) else 0.0 for v in BL]
    axL.plot(Ts, BL_clean, marker="o", color=color, linestyle=ls,
             label=label, lw=1.8, ms=6, drawstyle="steps-post")

    # panel (b): energy
    Ts_E = list(Ts)
    E_safe = [max(v, 1e-3) if not math.isnan(v) else float("nan") for v in E]
    axE.plot(Ts_E, E_safe, marker="o", color=color, linestyle=ls, label=label, lw=1.6, ms=5)
    # mark blowup with a red x
    for t, e, b in zip(Ts_E, E_safe, BL):
        if b >= 1.0:
            axE.scatter([t], [e], marker="x", color="red", s=120, zorder=5,
                        linewidths=2.5)

axL.set_xlabel("Rollout step T")
axL.set_ylabel("Blow-up fraction (of 200 test trajectories)")
axL.set_title(r"(a) Fraction of test trajectories that diverge by step $T$")
axL.set_xticks([10, 20, 50, 100])
axL.set_ylim(-0.05, 1.10)
axL.grid(alpha=0.3)
axL.legend(fontsize=8, loc="upper left")

axE.set_xlabel("Rollout step T")
axE.set_ylabel(r"mean energy $\langle \omega_T^2 \rangle$")
axE.set_yscale("log")
axE.set_title(r"(b) Vorticity energy vs T (log-y) — FNO diverges at T$\geq$50")
axE.set_xticks([10, 20, 50, 100])
axE.grid(alpha=0.3, which="both")
axE.legend(fontsize=8, loc="upper left")
# add text annotation for the FNO blowup
axE.annotate(r"FNO: 200/200 trajectories blow up",
             xy=(50, 3.6e7), xytext=(35, 1e15),
             fontsize=8, color="#D55E00",
             arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
