#!/usr/bin/env python3
"""Cross-viscosity zero-shot evaluator.

Evaluates a SpectraNet checkpoint trained on NS ν=10⁻⁵ on the FNO-released
ν=10⁻³ / ν=10⁻⁴ files without retraining.

**Honest framing.**  The cross-viscosity datasets were generated independently
of the headline ν=10⁻⁵ release (different ICs, possibly different forcing),
so this number conflates viscosity shift with IC-distribution shift.  We
treat it as a robustness probe in the appendix, not as a headline claim.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as scio
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _eval_common import load_spectranet


def load_cross_viscosity_data(path: str, H: int, W: int) -> np.ndarray:
    """Load NS ν cross-viscosity .mat files and normalise to (N, H, W, T).

    The ν=1e-5 file is scipy v5 with `u` of shape ``(N, H, W, T)``.
    The ν=1e-3 / ν=1e-4 files are HDF5 v7.3 with `u` reading as
    ``(T, H, W, N)`` (MATLAB column-major to NumPy row-major reversal).
    """
    print(f"Loading {path} ...")
    try:
        mat = scio.loadmat(path)
        u = np.asarray(mat["u"]).astype(np.float32)
        print(f"  scipy raw u shape: {u.shape}")
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as f:
            u = np.asarray(f["u"]).astype(np.float32)
        print(f"  h5py raw u shape: {u.shape}")
    if u.shape[1:3] == (H, W) and u.shape[0] != H:
        if u.shape[0] < u.shape[-1]:                # (T, H, W, N) layout
            u = np.transpose(u, (3, 1, 2, 0))
            print(f"  transposed to (N, H, W, T): {u.shape}")
    elif u.shape[2:] == (H, W):
        u = np.transpose(u, (0, 2, 3, 1))
        print(f"  transposed to (N, H, W, T): {u.shape}")
    return u


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--viscosity", required=True, help='Display label, e.g. "1e-3"')
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--out_csv", default="./results/cross_viscosity.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--T_out", type=int, default=10)
    p.add_argument("--S", type=int, default=64)
    p.add_argument("--label", default="spectranet")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    H = W = args.S

    u_all = load_cross_viscosity_data(args.data_path, H, W)
    N_total, _, _, T_total = u_all.shape
    if T_total < args.T_in + args.T_out:
        sys.exit(f"  data has only T={T_total} frames; need ≥ {args.T_in + args.T_out}")

    rng = np.random.default_rng(args.sample_seed)
    idx = rng.choice(N_total, size=min(args.n_samples, N_total), replace=False)
    idx.sort()
    n_eval = len(idx)
    x_test_np = u_all[idx, :, :, : args.T_in]
    y_test_np = u_all[idx, :, :, args.T_in : args.T_in + args.T_out]
    print(f"Sampled {n_eval} trajectories from N_total={N_total}")

    model, cfg = load_spectranet(
        args.ckpt, args.config, S=args.S, T_in=args.T_in, device=str(device),
    )
    residual = cfg.get("residual_target", False)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model: {args.label} ({n_params:,} params)  ν = {args.viscosity}")

    @torch.no_grad()
    def rollout_one(x_np: np.ndarray) -> np.ndarray:
        cur = torch.from_numpy(x_np[np.newaxis]).to(device)
        frames = []
        for _ in range(args.T_out):
            last = cur[..., -1:]
            raw = model(cur)
            y = (raw + last) if residual else raw
            frames.append(y.squeeze().cpu().numpy())
            cur = torch.cat([cur[..., 1:], y], dim=-1)
        return np.stack(frames, axis=-1)

    joint_l2s = []
    for i in range(n_eval):
        pred = rollout_one(x_test_np[i])
        gt = y_test_np[i]
        joint_l2s.append(float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), 1e-12)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_eval}  running mean L² = {np.mean(joint_l2s):.4f}")

    mean_l2 = float(np.mean(joint_l2s))
    std_l2 = float(np.std(joint_l2s, ddof=1)) if len(joint_l2s) > 1 else 0.0
    print(f"\n[ν = {args.viscosity}] {args.label}  N = {n_eval}  mean L² = {mean_l2:.4f}  std = {std_l2:.4f}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "ckpt_basename", "viscosity", "n_samples", "mean_L2", "std_L2", "params"])
        w.writerow([args.label, os.path.basename(args.ckpt), args.viscosity,
                    n_eval, f"{mean_l2:.6f}", f"{std_l2:.6f}", n_params])
    print(f"Appended → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
