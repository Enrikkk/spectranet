#!/usr/bin/env python3
"""Long-horizon rollout evaluator.

Rolls out each test sample for ``--T_max`` steps from the 10-frame input
window and reports per-step statistics:

  * for ``T <= T_out`` (10 by default): relative L² vs ground truth + energy;
  * for ``T > T_out``                  : energy, max vorticity, blowup flag
                                          (any ``|u| > 100`` or NaN).

Headline finding (paper Figure 3a): canonical FNO has ``blowup_frac = 1.00``
between ``T = 20`` and ``T = 50`` (every test trajectory diverges); SpectraNet
stays at ``0.00`` for all 100 steps.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _eval_common import load_spectranet
from spectranet.data import MatReader


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data_path", default="./data/NavierStokes_V1e-5_N1200_T20.mat")
    p.add_argument("--T_max", type=int, default=100)
    p.add_argument("--T_eval", type=str, default="10,20,50,100",
                   help="Comma-separated horizons to log")
    p.add_argument("--out_csv", default="./results/long_horizon.csv")
    p.add_argument("--out_fig", default="./figures/long_horizon.pdf")
    p.add_argument("--device", default="cuda")
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--T_out", type=int, default=10)
    p.add_argument("--ntrain", type=int, default=850)
    p.add_argument("--nval", type=int, default=150)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--S", type=int, default=64)
    p.add_argument("--label", default="spectranet",
                   help="Label written to the CSV / figure legend")
    args = p.parse_args()

    T_eval = [int(t) for t in args.T_eval.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    H = W = args.S

    model, cfg = load_spectranet(
        args.ckpt, args.config, S=args.S, T_in=args.T_in, device=str(device),
    )
    n_params = sum(p_.numel() for p_ in model.parameters())
    residual = cfg.get("residual_target", False)
    print(f"Model: {args.label} ({n_params:,} params)  T_max={args.T_max}  T_eval={T_eval}")

    print(f"Loading {args.data_path} ...")
    u_all = MatReader(args.data_path).read_field("u").numpy()
    x_test_np = u_all[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest, :, :, : args.T_in]
    y_test_np = u_all[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest, :, :, args.T_in : args.T_in + args.T_out]
    y_test = torch.from_numpy(y_test_np)

    @torch.no_grad()
    def rollout_one(x_np: np.ndarray):
        cur = torch.from_numpy(x_np[np.newaxis]).to(device)  # (1, H, W, T_in)
        frames = []
        for _ in range(args.T_max):
            last = cur[..., -1:]
            raw = model(cur)
            y = (raw + last) if residual else raw
            frames.append(y.squeeze().cpu().numpy())
            cur = torch.cat([cur[..., 1:], y], dim=-1)
        return frames

    acc_l2 = {t: [] for t in T_eval}
    acc_energy = {t: [] for t in T_eval}
    acc_maxvort = {t: [] for t in T_eval}
    acc_blowup = {t: [] for t in T_eval}
    joint_l2s = []

    ntest_actual = x_test_np.shape[0]
    for i in range(ntest_actual):
        frames = rollout_one(x_test_np[i])
        all_arr = np.stack(frames)
        blowup_mask = np.isnan(all_arr) | (np.abs(all_arr) > 100)

        pred_full = np.stack(frames[: args.T_out], axis=-1)  # (H, W, T_out)
        gt_full = y_test_np[i]
        diff = np.linalg.norm(pred_full - gt_full)
        denom = np.linalg.norm(gt_full)
        joint_l2s.append(diff / max(denom, 1e-12))

        for t in T_eval:
            if t > args.T_max:
                continue
            frame_t = frames[t - 1]
            acc_energy[t].append(float(np.mean(frame_t ** 2)))
            acc_maxvort[t].append(float(np.max(np.abs(frame_t))))
            acc_blowup[t].append(float(blowup_mask[:t].any()))
            if t <= args.T_out:
                gt_t = y_test[i, :, :, t - 1]
                pred_t = torch.from_numpy(frame_t)
                acc_l2[t].append(((pred_t - gt_t).norm() / gt_t.norm().clamp(min=1e-12)).item())
            else:
                acc_l2[t].append(float("nan"))

        if (i + 1) % 50 == 0 and args.T_out in T_eval and acc_l2[args.T_out]:
            print(
                f"  {i+1}/{ntest_actual}  T={args.T_out} L2={np.nanmean(acc_l2[args.T_out]):.4f}  "
                f"energy@T={args.T_out}={np.mean(acc_energy[args.T_out]):.4f}"
            )

    headline = cfg.get("best_test_l2", None)
    joint = float(np.mean(joint_l2s))
    if headline is not None:
        gap = abs(joint - headline)
        status = "OK" if gap < 0.002 else "WARNING"
        print(f"\n[sanity] full-rollout L2={joint:.4f}  headline={headline:.4f}  gap={gap:.4f}  {status}")
    else:
        print(f"\n[sanity] full-rollout L2={joint:.4f}  (no headline in config)")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "T", "mean_L2", "std_L2", "mean_energy", "max_vorticity", "blowup_frac"])
        for t in T_eval:
            if t > args.T_max:
                continue
            l2s = [v for v in acc_l2[t] if not np.isnan(v)]
            row = [
                args.label, t,
                f"{np.nanmean(acc_l2[t]):.6f}" if l2s else "nan",
                f"{np.nanstd(acc_l2[t]):.6f}" if l2s else "nan",
                f"{np.mean(acc_energy[t]):.6f}",
                f"{np.mean(acc_maxvort[t]):.6f}",
                f"{np.mean(acc_blowup[t]):.6f}",
            ]
            w.writerow(row)
            print(f"  T={t:3d}  L2={row[2]}  energy={row[4]}  blowup={row[6]}")
    print(f"Appended → {out}")

    # Refresh figure from the cumulative CSV
    try:
        rows = []
        with open(out) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        models_seen = list(dict.fromkeys(r["model"] for r in rows))
        cmap = {m: c for m, c in zip(models_seen, plt.cm.tab10(np.linspace(0, 1, len(models_seen))))}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for m in models_seen:
            m_rows = [r for r in rows if r["model"] == m]
            l2_pairs = sorted([(int(r["T"]), float(r["mean_L2"]))
                               for r in m_rows if r["mean_L2"] != "nan"])
            en_pairs = sorted([(int(r["T"]), float(r["mean_energy"])) for r in m_rows])
            if l2_pairs:
                Ts_l2, L2s = zip(*l2_pairs)
                ax1.plot(Ts_l2, L2s, marker="o", label=m, color=cmap[m])
            if en_pairs:
                Ts_en, Ens = zip(*en_pairs)
                ax2.plot(Ts_en, Ens, marker="s", label=m, color=cmap[m])
        ax1.set_xlabel("Prediction horizon T")
        ax1.set_ylabel("Mean relative L²")
        ax1.set_title("(a) Accuracy vs horizon (T ≤ T_out)")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Rollout step T")
        ax2.set_ylabel("Mean field energy ‖u_T‖²/(H·W)")
        ax2.set_title("(b) Energy drift (free rollout)")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure → {args.out_fig}")
    except Exception as e:
        print(f"Figure generation skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
