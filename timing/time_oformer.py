"""Inference timing for OFormer (encoder + decoder, AR rollout)."""
import argparse
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OF_DIR   = "./third_party/OFormer/uniform_grids"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, OF_DIR)

from nn_module.encoder_module import SpatialTemporalEncoder2D  # noqa: E402
from nn_module.decoder_module import PointWiseDecoder2D        # noqa: E402
from common import time_forward, count_params, write_rows      # noqa: E402


def main():
    ap = argparse.ArgumentParser('OFormer inference timing')
    ap.add_argument('--checkpoint', default=f"{OF_DIR}/checkpoints/ns_oformer_our_split_best.pt")
    ap.add_argument('--output_csv', default=f"{THIS_DIR}/../results/OFormer_timing.csv")
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters',  type=int, default=50)
    args = ap.parse_args()

    device = torch.device('cuda')
    H = W = 64
    N = H * W
    T_in, T_out, step = 10, 10, 1

    # paper defaults from ns_oformer.py argparse
    in_emb_dim       = 128
    out_seq_emb_dim  = 64
    encoder_heads    = 1
    encoder_depth    = 5
    decoder_emb_dim  = 64
    out_channels     = 1
    out_step         = 1
    propagator_depth = 1
    fourier_frequency = 8
    in_channels      = T_in + 2

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
    pos = torch.stack([xx.ravel(), yy.ravel()], dim=-1)  # (N, 2)

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
                        extra={'model': 'OFormer', 'variant': '', 'params': n_params})
    write_rows(args.output_csv, rows)
    print(f"[OFormer] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
