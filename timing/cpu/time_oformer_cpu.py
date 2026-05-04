"""CPU port of time_oformer.py."""
import argparse
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import OF_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, OF_DIR)

from nn_module.encoder_module import SpatialTemporalEncoder2D  # noqa: E402
from nn_module.decoder_module import PointWiseDecoder2D        # noqa: E402
from common_cpu import time_forward, count_params, write_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser('OFormer CPU inference timing')
    ap.add_argument('--checkpoint', default=os.path.join(OF_DIR, "checkpoints", "ns_oformer_our_split_best.pt"))
    ap.add_argument('--output_csv', default=os.path.join(RESULTS_DIR, "OFormer_timing.csv"))
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=120.0)
    args = ap.parse_args()

    device = torch.device('cpu')
    H = W = 64
    N = H * W
    T_in, T_out, step = 10, 10, 1

    in_emb_dim, out_seq_emb_dim = 128, 64
    encoder_heads, encoder_depth = 1, 5
    decoder_emb_dim, out_channels, out_step = 64, 1, 1
    propagator_depth, fourier_frequency = 1, 8
    in_channels = T_in + 2

    encoder = SpatialTemporalEncoder2D(
        in_channels, in_emb_dim, out_seq_emb_dim, encoder_heads, encoder_depth
    ).to(device).eval()
    decoder = PointWiseDecoder2D(
        decoder_emb_dim, out_channels, out_step, propagator_depth,
        scale=fourier_frequency, dropout=0.0,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    n_params = count_params(encoder) + count_params(decoder)
    print(f"[OFormer] params={n_params:,}")

    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    pos = torch.stack([xx.ravel(), yy.ravel()], dim=-1)

    @torch.no_grad()
    def forward_fn(B: int):
        fx = torch.randn(B, N, T_in, device=device)
        pos_bc = pos.unsqueeze(0).expand(B, -1, -1)
        cur_fx = fx
        for _ in range(T_out):
            enc_in = torch.cat([cur_fx, pos_bc], dim=-1)
            z = encoder(enc_in, pos_bc)
            u, _ = decoder(z, pos_bc)
            out = u.permute(0, 2, 1)
            cur_fx = torch.cat([cur_fx[..., step:], out], dim=-1)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'OFormer', 'variant': '', 'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[OFormer] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
