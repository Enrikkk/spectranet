#!/usr/bin/env python3
"""Zero-shot resolution-transfer evaluator (64²→128²).

Loads a SpectraNet checkpoint trained at 64² and runs it without retraining
at the target resolution after upsampling the test inputs.

The spectral path is exactly resolution-invariant by construction
(``rfft2 + irfft2`` adapt to the input grid).  The two-layer MLP head is per
pixel and so is also resolution-agnostic in the canonical configuration.

Two upsampling methods are provided:

* ``bilinear``         — ``F.interpolate(mode='bilinear', align_corners=True)``;
* ``spectral_zeropad`` — zero-pad the rfft2 array; ideal upsampling with no
                         high-frequency aliasing.

128² ground truth is not available in the public NS dataset.  The reported
L² is computed against the bandlimit-aliased upsampled test set, which is an
upper bound on the true zero-shot transfer error (paper §7.4 caveat).
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _eval_common import load_spectranet
from spectranet.data import MatReader


def upsample_bilinear(x_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    N, H, W, T = x_np.shape
    t = torch.from_numpy(x_np).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=True)
    return t.permute(0, 2, 3, 1).numpy()


def upsample_spectral(x_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Spectral zero-padding: rfft2 → pad → irfft2.

    Equivalent to ideal sinc upsampling (no high-frequency leakage beyond the
    original Nyquist).  Energy-normalised by ``(target_h * target_w)/(H * W)``.
    """
    N, H, W, T = x_np.shape
    t = torch.from_numpy(x_np).permute(0, 3, 1, 2).float()
    ft = torch.fft.rfft2(t)
    fH, fW = target_h, target_w // 2 + 1
    ft_pad = torch.zeros(N, T, fH, fW, dtype=torch.cfloat)
    cH = min(H // 2, target_h // 2)
    cW = min(W // 2 + 1, fW)
    ft_pad[:, :, :cH, :cW] = ft[:, :, :cH, :cW]
    ft_pad[:, :, -cH:, :cW] = ft[:, :, -cH:, :cW]
    scale = (target_h * target_w) / (H * W)
    out = torch.fft.irfft2(ft_pad * scale, s=(target_h, target_w))
    return out.permute(0, 2, 3, 1).numpy()


def rel_l2_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    N = pred.shape[0]
    diff = (pred - target).reshape(N, -1).norm(dim=-1)
    denom = target.reshape(N, -1).norm(dim=-1).clamp(min=1e-12)
    return (diff / denom).mean().item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data_path", default="./data/NavierStokes_V1e-5_N1200_T20.mat")
    p.add_argument("--target_res", type=int, default=128)
    p.add_argument("--upsample_method", default="bilinear",
                   choices=["bilinear", "spectral_zeropad"])
    p.add_argument("--out_csv", default="./results/resolution_transfer.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--T_out", type=int, default=10)
    p.add_argument("--ntrain", type=int, default=850)
    p.add_argument("--nval", type=int, default=150)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--source_res", type=int, default=64)
    p.add_argument("--label", default="spectranet")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    H_src = W_src = args.source_res
    H_tgt = W_tgt = args.target_res

    print(f"Loading {args.data_path} ...")
    u_all = MatReader(args.data_path).read_field("u").numpy()
    x_test_np = u_all[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest, :, :, : args.T_in]
    y_test_np = u_all[args.ntrain + args.nval : args.ntrain + args.nval + args.ntest, :, :, args.T_in : args.T_in + args.T_out]

    @torch.no_grad()
    def rollout(model, residual: bool, x_np: np.ndarray) -> torch.Tensor:
        cur = torch.from_numpy(x_np).to(device)
        pred = None
        for _ in range(args.T_out):
            last = cur[..., -1:]
            raw = model(cur)
            y = (raw + last) if residual else raw
            pred = y if pred is None else torch.cat([pred, y], dim=-1)
            cur = torch.cat([cur[..., 1:], y], dim=-1)
        return pred.cpu()

    # Native (sanity)
    print("\n── Native source-resolution evaluation ──")
    model, cfg = load_spectranet(
        args.ckpt, args.config, S=H_src, T_in=args.T_in, device=str(device),
    )
    residual = cfg.get("residual_target", False)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model: {args.label} ({n_params:,} params)  S={H_src}")
    pred_native = rollout(model, residual, x_test_np)
    y_native = torch.from_numpy(y_test_np)
    l2_native = rel_l2_batch(pred_native, y_native)
    headline = cfg.get("best_test_l2")
    if headline is not None:
        gap = abs(l2_native - headline)
        status = "OK" if gap < 0.002 else "WARNING"
        print(f"L² ({H_src}²→{H_src}²): {l2_native:.4f}  headline={headline:.4f}  gap={gap:.4f}  {status}")
    else:
        print(f"L² ({H_src}²→{H_src}²): {l2_native:.4f}")
    del model

    # Transfer
    print(f"\n── Zero-shot {H_src}²→{H_tgt}² ({args.upsample_method}) ──")
    if args.upsample_method == "bilinear":
        x_tgt = upsample_bilinear(x_test_np, H_tgt, W_tgt)
        y_tgt = upsample_bilinear(y_test_np, H_tgt, W_tgt)
    else:
        x_tgt = upsample_spectral(x_test_np, H_tgt, W_tgt)
        y_tgt = upsample_spectral(y_test_np, H_tgt, W_tgt)
    print(f"Upsampled shape: {x_tgt.shape}")

    model, _ = load_spectranet(
        args.ckpt, args.config, S=H_tgt, T_in=args.T_in, device=str(device),
    )
    pred_tgt = rollout(model, residual, x_tgt)
    y_tgt_t = torch.from_numpy(y_tgt)
    l2_transfer = rel_l2_batch(pred_tgt, y_tgt_t)
    ratio = l2_transfer / max(l2_native, 1e-12)
    print(f"L² ({H_src}²→{H_tgt}², {args.upsample_method}): {l2_transfer:.4f}")
    print(f"Transfer / native ratio: {ratio:.3f}×")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "params", "train_res", "test_res", "upsample",
                        "mean_L2", "transfer_native_ratio"])
        w.writerow([args.label, n_params, H_src, H_src, "none",
                    f"{l2_native:.6f}", "1.000000"])
        w.writerow([args.label, n_params, H_src, H_tgt, args.upsample_method,
                    f"{l2_transfer:.6f}", f"{ratio:.6f}"])
    print(f"Appended → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
