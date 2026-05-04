"""Inference timing for Transolver_Structured_Mesh_2D. AR rollout x10."""
import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TS_DIR   = "./third_party/Transolver/PDE-Solving-StandardBenchmark"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, TS_DIR)

from model_dict import get_model                            # noqa: E402
from common import time_forward, count_params, write_rows   # noqa: E402


def main():
    ap = argparse.ArgumentParser('Transolver inference timing')
    ap.add_argument('--checkpoint', default=f"{TS_DIR}/checkpoints/ns_transolver_our_split_best.pt")
    ap.add_argument('--output_csv', default=f"{THIS_DIR}/../results/Transolver_timing.csv")
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters',  type=int, default=50)
    args = ap.parse_args()

    device = torch.device('cuda')
    h, T_in, T_out = 64, 10, 10

    # paper defaults from ns_transolver.py
    margs = Namespace(model='Transolver_Structured_Mesh_2D')
    model = get_model(margs).Model(
        space_dim=2, n_layers=8, n_hidden=256, dropout=0.0,
        n_head=8, Time_Input=False, mlp_ratio=1, fun_dim=T_in, out_dim=1,
        slice_num=32, ref=8, unified_pos=0, H=h, W=h,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[Transolver] params={n_params:,}")

    xc, yc = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, h))
    pos_single = torch.tensor(np.c_[xc.ravel(), yc.ravel()],
                              dtype=torch.float32, device=device)

    @torch.no_grad()
    def forward_fn(B: int):
        x  = pos_single.unsqueeze(0).expand(B, -1, -1).contiguous()
        fx = torch.randn(B, h * h, T_in, device=device)
        for _ in range(T_out):
            im = model(x, fx=fx)
            fx = torch.cat((fx[..., 1:], im), dim=-1)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'Transolver', 'variant': '', 'params': n_params})
    write_rows(args.output_csv, rows)
    print(f"[Transolver] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
