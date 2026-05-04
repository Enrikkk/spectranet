#!/usr/bin/env python3
"""Empirical Lipschitz-constant estimator.

For each of ``--n_samples`` test inputs ``u`` we draw ``--n_perturbations``
random perturbations ``ε`` of ``L²`` norm ``--eps`` and measure
``L̂ = ‖f(u+ε) − f(u)‖ / ‖ε‖`` where ``f`` is either a single model step
(``--rollout_T 1``) or a 10-step free rollout (``--rollout_T 10``).  Results
(max / mean / p95) are appended as a single row to ``--out_csv``.

Sanity check (see paper §5 / Lipschitz appendix):
    L̂(SpectraNet, T=1, residual)   ≈ 1.0
    L̂(SpectraNet, T=10, residual)  ≈ 12 (expected for a residual-target net)
    L̂(FNO, T=1)                    ≈ 1.8
    L̂(Transformer, T=1)            ≈ 0.17
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _eval_common import free_rollout_ar2d, load_spectranet
from spectranet.data import MatReader


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="SpectraNet checkpoint path")
    p.add_argument("--config", default=None,
                   help="Optional config JSON (auto-inferred from ckpt path if omitted)")
    p.add_argument("--model_kind", default="ar2d", choices=["ar2d"],
                   help="Reserved.  See scripts/eval_lipschitz_baseline.py for FNO/Transformer.")
    p.add_argument("--data_path", default="./data/NavierStokes_V1e-5_N1200_T20.mat")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--n_perturbations", type=int, default=10)
    p.add_argument("--rollout_T", type=int, default=1, choices=[1, 10])
    p.add_argument("--out_csv", default="./results/lipschitz.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--ntrain", type=int, default=850)
    p.add_argument("--nval", type=int, default=150)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--S", type=int, default=64)
    p.add_argument("--label", default=None,
                   help="Override the model label written to the CSV (defaults to model_kind)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_spectranet(
        args.ckpt, args.config, S=args.S, T_in=args.T_in, device=str(device),
    )
    n_params = sum(p_.numel() for p_ in model.parameters())
    residual_target = cfg.get("residual_target", False)
    print(f"Model: SpectraNet ({n_params:,} params)  rollout_T={args.rollout_T}  residual={residual_target}")

    print(f"Loading {args.data_path} ...")
    reader = MatReader(args.data_path)
    u_all = reader.read_field("u").numpy()  # (B, S, S, 20)
    x_test_np = u_all[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest, :, :, : args.T_in]

    L_hats = []
    n = min(args.n_samples, x_test_np.shape[0])
    for i in range(n):
        u_in = torch.from_numpy(x_test_np[i][np.newaxis]).to(device)  # (1, S, S, T_in)
        f_u = free_rollout_ar2d(model, u_in, args.rollout_T, residual_target)
        for _ in range(args.n_perturbations):
            eps_vec = torch.randn_like(u_in)
            eps_vec = eps_vec * (args.eps / eps_vec.norm().clamp(min=1e-12))
            f_upe = free_rollout_ar2d(model, u_in + eps_vec, args.rollout_T, residual_target)
            L = (f_u - f_upe).norm().item() / args.eps
            L_hats.append(L)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}  running max={max(L_hats):.4f}  mean={np.mean(L_hats):.4f}")

    L_arr = np.asarray(L_hats)
    L_max = float(L_arr.max())
    L_mean = float(L_arr.mean())
    L_p95 = float(np.percentile(L_arr, 95))
    print(f"\n[lipschitz] T={args.rollout_T}: max={L_max:.4f} mean={L_mean:.4f} p95={L_p95:.4f}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    label = args.label or "spectranet"
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "model", "rollout_T", "params", "n_samples", "n_perturb", "eps",
                "L_hat_max", "L_hat_mean", "L_hat_p95",
            ])
        w.writerow([
            label, args.rollout_T, n_params, n, args.n_perturbations, args.eps,
            f"{L_max:.6f}", f"{L_mean:.6f}", f"{L_p95:.6f}",
        ])
    print(f"Appended → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
