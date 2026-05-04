#!/usr/bin/env python3
"""Train/val L2 vs epoch for the four headline-relevant models.

Reads losses CSVs (epoch, train_l2, val_l2) from paper/data/, plots two panels:
  (a) train L2 vs epoch (log-y)
  (b) val L2 vs epoch (log-y)

Output: figures/train_val_curves.pdf
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]   # paper/
DATA = ROOT / "data"
OUT  = ROOT / "figures" / "train_val_curves.pdf"

SERIES = [
    ("ar2d_w32_resi_e6_losses.csv",
        "SpectraNet (canonical headline)",                "#0072B2", "-"),
    ("ar2d_w32_resi_losses.csv",
        "SpectraNet $w{=}32$ + residual (no Semigroup-Cons.\\ Loss)",
                                                          "#56B4E9", "--"),
    ("fno_losses.csv",
        "FNO (NSL baseline)",                             "#D55E00", "-"),
    ("transformer_losses.csv",
        "Transformer (NSL baseline)",                     "#009E73", "-"),
]


def read_curves(path):
    eps, tr, va = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            eps.append(int(row["epoch"]))
            tr.append(float(row["train_l2"]))
            va.append(float(row["val_l2"]))
    return eps, tr, va


fig, (axT, axV) = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)

for fname, label, color, ls in SERIES:
    p = DATA / fname
    if not p.exists():
        print(f"WARN: missing {p}; skipping")
        continue
    eps, tr, va = read_curves(p)
    axT.plot(eps, tr, color=color, linestyle=ls, lw=1.4, label=label, alpha=0.9)
    axV.plot(eps, va, color=color, linestyle=ls, lw=1.4, label=label, alpha=0.9)

for ax, title in [(axT, "(a) Training relative $L^2$"),
                  (axV, "(b) Validation relative $L^2$")]:
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"relative $L^2$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
