"""CPU port of time_ono.py."""
import argparse
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import ONO_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, ONO_DIR)

from ONOmodel2 import ONO2                                    # noqa: E402
from common_cpu import time_forward, count_params, write_rows # noqa: E402


def main():
    ap = argparse.ArgumentParser('ONO2 CPU inference timing')
    ap.add_argument('--checkpoint', default=os.path.join(ONO_DIR, "checkpoints", "ns_ono_our_split_best.pt"))
    ap.add_argument('--output_csv', default=os.path.join(RESULTS_DIR, "ONO_timing.csv"))
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=120.0)
    args = ap.parse_args()

    device = torch.device('cpu')
    h, T_in, T_out = 64, 10, 10

    model = ONO2(
        n_hidden=64, n_layers=3, space_dim=2,
        Time_Input=False, fun_dim=T_in, n_head=4,
        momentum=0.9, orth=0, psi_dim=64,
        mlp_ratio=1, attn_type=None,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[ONO] params={n_params:,}")

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
                        extra={'model': 'ONO', 'variant': '', 'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[ONO] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
