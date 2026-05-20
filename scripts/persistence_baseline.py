#!/usr/bin/env python3
"""Trivial persistence baseline — predict ``u_t = u_{t-1}`` for all ``t``.

This is the floor that every learned operator must beat to claim non-trivial
generalization.  ``LpLoss`` of persistence on NS ν = 10⁻⁵ at 64² is roughly
``0.7481`` — about 7× worse than SpectraNet.
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectranet.data import MatReader
from spectranet.losses import LpLoss


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/NavierStokes_V1e-5_N1200_T20.mat")
    p.add_argument("--out_csv", default="./results/persistence.csv")
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--T_out", type=int, default=10)
    p.add_argument("--ntrain", type=int, default=850)
    p.add_argument("--nval", type=int, default=150)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--label", default="persistence")
    args = p.parse_args()

    print(f"Loading {args.data_path} ...")
    u_all = MatReader(args.data_path).read_field("u")
    s = args.ntrain + args.nval
    X = u_all[s : s + args.ntest, :, :, : args.T_in]                      # (N, H, W, T_in)
    Y = u_all[s : s + args.ntest, :, :, args.T_in : args.T_in + args.T_out]  # (N, H, W, T_out)

    # Persistence: predict the last input frame for every output step.
    last = X[..., -1:]                         # (N, H, W, 1)
    pred = last.expand_as(Y).contiguous()      # (N, H, W, T_out)

    loss = LpLoss(size_average=True)
    l2 = loss(pred.reshape(args.ntest, -1), Y.reshape(args.ntest, -1)).item()
    print(f"[persistence] mean relative L² = {l2:.4f} (ntest = {args.ntest})")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "params", "ntest", "test_L2"])
        w.writerow([args.label, 0, args.ntest, f"{l2:.6f}"])
    print(f"Appended → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
