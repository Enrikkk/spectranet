"""Inference timing for GNOT (CGPTNO). Single-shot, non-AR, graph-based."""
import argparse
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GN_DIR   = "./third_party/gnot"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, GN_DIR)

import dgl                                                    # noqa: E402
from models.cgpt import CGPTNO                                # noqa: E402
from common import time_forward, count_params, write_rows     # noqa: E402


def _make_batched_graph(B: int, N: int, T_in: int, device) -> "dgl.DGLGraph":
    """Build a batch of B graphs each with N nodes, no edges, ndata['x'] of
    shape (N, T_in+2). Mirrors FNODataset.process() output then dgl.batch().
    Uses identical coords/random vorticity each call — accurate for FLOP work."""
    grid_1d = np.linspace(0, 1, 64, dtype=np.float32)
    gx, gy  = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    coords  = np.stack([gx, gy], axis=-1).reshape(N, 2)
    coords_t = torch.from_numpy(coords).to(device)            # (N, 2)

    graphs = []
    for _ in range(B):
        g = dgl.DGLGraph().to(device)
        g.add_nodes(N)
        u_in = torch.randn(N, T_in, device=device)
        g.ndata['x'] = torch.cat([coords_t, u_in], dim=-1)    # (N, T_in+2)
        g.ndata['u_flag'] = torch.zeros(N, 1, device=device)
        graphs.append(g)
    return dgl.batch(graphs)


def main():
    ap = argparse.ArgumentParser('GNOT inference timing')
    ap.add_argument('--checkpoint', default=f"{GN_DIR}/checkpoints/ns_gnot_our_split_best.pt")
    ap.add_argument('--output_csv', default=f"{THIS_DIR}/../results/GNOT_timing.csv")
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters',  type=int, default=50)
    args = ap.parse_args()

    device = torch.device('cuda')
    H = W = 64
    N = H * W
    T_in, T_out = 10, 10

    # paper defaults from ns_gnot.py
    input_dim  = T_in + 2
    output_dim = T_out

    model = CGPTNO(
        trunk_size   = input_dim + 1,
        branch_sizes = None,
        output_size  = output_dim,
        n_layers     = 3,
        n_hidden     = 128,
        n_head       = 1,
        attn_type    = 'linear',
        ffn_dropout  = 0.0,
        attn_dropout = 0.0,
        mlp_layers   = 3,
        act          = 'gelu',
        horiz_fourier_dim = 0,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[GNOT] params={n_params:,}")

    @torch.no_grad()
    def forward_fn(B: int):
        g   = _make_batched_graph(B, N, T_in, device)
        u_p = torch.zeros(B, 1, device=device)
        # MIODataLoader feeds (g, u_p, g_u) where g_u may be None or a graph;
        # ns_gnot test loop passes g_u via g.ndata['u'] = zeros, but the
        # test loop just calls model(g, u_p, g_u). Safe to pass None.
        out = model(g, u_p, None)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'GNOT', 'variant': '', 'params': n_params})
    write_rows(args.output_csv, rows)
    print(f"[GNOT] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
