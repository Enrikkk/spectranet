#!/usr/bin/env python3
"""Generate the in-house native-128² Navier–Stokes dataset.

Faithful port of the Li et al. (2020) vorticity-form pseudo-spectral solver
used to produce the public 64² ν=10⁻⁵ release.  The numerics are:

  * 2D vorticity equation in stream-function form on a periodic ``[0,1]²``;
  * Crank–Nicolson on the linear (viscous) part, explicit forward Euler on
    the nonlinear advection;
  * 2/3 dealiasing on the convolution sum;
  * Gaussian-random-field initial conditions with Matérn-like spectrum
    ``(4π²k² + τ²)⁻ᵅ`` (defaults α=2.5, τ=7.0 — same as Li et al.);
  * forcing ``f = 0.1·(sin 2π(x+y) + cos 2π(x+y))``.

The output ``.mat`` file is bit-identical in structure to the public 64²
release (single field ``u`` of shape ``(N, S, S, T)``, scipy v5 format).

Run with the defaults to reproduce the shipped ``data/ns_v1e5_128_N1200_T20.mat``::

    python scripts/generate_ns_128.py --out ./data/ns_v1e5_128_N1200_T20.mat

GPU is required (the dt=1e-4 explicit advection is dominated by the FFT cost;
~30 minutes per 50-trajectory batch on a single H100).
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path

import scipy.io as sio
import torch


def gaussian_random_field_2d(N: int, n_traj: int, device, alpha: float = 2.5, tau: float = 7.0):
    """Sample ``n_traj`` Matérn-spectrum Gaussian random fields on an ``N×N`` grid."""
    sigma = tau ** (0.5 * (2 * alpha - 2.0))
    k_idx = torch.cat([torch.arange(0, N // 2), torch.arange(-(N // 2), 0)]).to(device).float()
    K1, K2 = torch.meshgrid(k_idx, k_idx, indexing="ij")
    sqrt_eig = (
        (N ** 2) * math.sqrt(2.0) * sigma
        * (4 * math.pi ** 2 * (K1 ** 2 + K2 ** 2) + tau ** 2).pow(-alpha / 2.0)
    )
    sqrt_eig[0, 0] = 0.0
    re = torch.randn(n_traj, N, N, device=device)
    im = torch.randn(n_traj, N, N, device=device)
    xi = re + 1j * im
    u_h = sqrt_eig.unsqueeze(0) * xi
    return torch.fft.ifft2(u_h).real


def navier_stokes_2d_batch(w0, f, visc: float, T_final: float, dt_sub: float, record_every: int):
    """Crank–Nicolson + explicit advection + 2/3 dealiasing.

    ``w0``: ``(B, N, N)`` initial vorticity, ``f``: ``(N, N)`` forcing.
    Returns ``(B, N, N, T_records)``.
    """
    B, N, _ = w0.shape
    device = w0.device

    k_y = torch.cat([torch.arange(0, N // 2), torch.arange(-(N // 2), 0)]).to(device).float()
    k_x = torch.arange(0, N // 2 + 1).to(device).float()
    K1, K2 = torch.meshgrid(k_y, k_x, indexing="ij")
    lap = 4 * math.pi ** 2 * (K1 ** 2 + K2 ** 2)
    lap_safe = lap.clone()
    lap_safe[0, 0] = 1.0

    cutoff = (2.0 / 3.0) * (N // 2)
    dealias = ((K1.abs() < cutoff) & (K2.abs() < cutoff)).float()

    f_h = torch.fft.rfft2(f).unsqueeze(0)
    w_h = torch.fft.rfft2(w0)

    n_steps = int(round(T_final / dt_sub))
    n_records = n_steps // record_every
    sol = torch.zeros(B, N, N, n_records, device=device, dtype=torch.float32)

    rec = 0
    for j in range(n_steps):
        psi_h = w_h / lap_safe.unsqueeze(0)
        psi_h[..., 0, 0] = 0.0

        u_x_h = 2j * math.pi * K2.unsqueeze(0) * psi_h
        u_y_h = -2j * math.pi * K1.unsqueeze(0) * psi_h
        w_x_h = 2j * math.pi * K1.unsqueeze(0) * w_h
        w_y_h = 2j * math.pi * K2.unsqueeze(0) * w_h

        u_x = torch.fft.irfft2(u_x_h, s=(N, N))
        u_y = torch.fft.irfft2(u_y_h, s=(N, N))
        w_x = torch.fft.irfft2(w_x_h, s=(N, N))
        w_y = torch.fft.irfft2(w_y_h, s=(N, N))

        N_h = torch.fft.rfft2(u_x * w_x + u_y * w_y)
        N_h = dealias.unsqueeze(0) * N_h

        denom = 1.0 + 0.5 * dt_sub * visc * lap.unsqueeze(0)
        numer = (1.0 - 0.5 * dt_sub * visc * lap.unsqueeze(0)) * w_h - dt_sub * N_h + dt_sub * f_h
        w_h = numer / denom

        if (j + 1) % record_every == 0 and rec < n_records:
            sol[..., rec] = torch.fft.irfft2(w_h, s=(N, N))
            rec += 1

    return sol


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=128, help="Spatial resolution")
    p.add_argument("--n_traj", type=int, default=1200, help="Number of trajectories")
    p.add_argument("--T_records", type=int, default=20, help="Recorded frames per trajectory")
    p.add_argument("--T_final", type=float, default=20.0, help="Final integration time")
    p.add_argument("--dt_sub", type=float, default=1e-4, help="Sub-step size")
    p.add_argument("--visc", type=float, default=1e-5, help="Viscosity ν")
    p.add_argument("--batch", type=int, default=50, help="Trajectories per GPU batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="./data/ns_v1e5_128_N1200_T20.mat")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        print("[generate] WARNING: CUDA unavailable; this will take many hours on CPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"[generate] device = {torch.cuda.get_device_name(0)}")

    record_every = int(round(args.T_final / args.T_records / args.dt_sub))
    n_steps = int(round(args.T_final / args.dt_sub))
    print(
        f"[generate] N={args.N}, n_traj={args.n_traj}, T_records={args.T_records}, "
        f"T_final={args.T_final}, dt_sub={args.dt_sub}, visc={args.visc}"
    )
    print(f"[generate] Sub-steps/traj: {n_steps}, record every {record_every} sub-steps")

    grid = torch.linspace(0, 1, args.N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    sol_full = torch.zeros(args.n_traj, args.N, args.N, args.T_records, dtype=torch.float32)

    t0 = time.time()
    n_batches = (args.n_traj + args.batch - 1) // args.batch
    for b in range(n_batches):
        s = b * args.batch
        e = min((b + 1) * args.batch, args.n_traj)
        bs = e - s
        w0 = gaussian_random_field_2d(args.N, bs, device)
        sol = navier_stokes_2d_batch(w0, f, args.visc, args.T_final, args.dt_sub, record_every)
        sol_full[s:e] = sol.cpu()
        elapsed = time.time() - t0
        eta = elapsed / (b + 1) * (n_batches - b - 1)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(
            f"  batch {b+1}/{n_batches}: {bs} traj | elapsed {elapsed/60:.1f} min | "
            f"ETA {eta/60:.1f} min | peak GPU {peak_gb:.1f} GB | "
            f"last min/max {sol[..., -1].min():.3f}/{sol[..., -1].max():.3f}",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[generate] saving to {out_path} ...")
    sio.savemat(out_path, {"u": sol_full.numpy()}, do_compression=False)
    print(f"  size: {os.path.getsize(out_path) / 1e6:.1f} MB")
    print(f"  shape: {tuple(sol_full.shape)}")
    print(
        f"  stats: min={sol_full.min():.3f}  max={sol_full.max():.3f}  "
        f"mean={sol_full.mean():.4f}  std={sol_full.std():.4f}"
    )
    print(f"[generate] total elapsed: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
