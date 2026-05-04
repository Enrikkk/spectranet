"""CPU port of time_factformer.py."""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
from einops import rearrange

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import FF_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, FF_DIR)

from einops.layers.torch import Rearrange                             # noqa: E402
from libs.factorization_module import FABlock2D                      # noqa: E402
from libs.positional_encoding_module import GaussianFourierFeatureTransform  # noqa: E402
from libs.basics import PreNorm, MLP                                  # noqa: E402
from common_cpu import time_forward, count_params, write_rows         # noqa: E402


class FactorizedTransformer(nn.Module):
    def __init__(self, dim, dim_head, heads, dim_out, depth, kernel_multiplier=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer = nn.ModuleList([
                nn.Sequential(
                    GaussianFourierFeatureTransform(2, dim // 2, 8),
                    nn.Linear(dim, dim)
                ),
                FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                          kernel_multiplier=kernel_multiplier)
            ])
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(
            torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)], indexing='ij'),
            dim=-1
        )
        for pos_enc, attn_layer in self.layers:
            u = u + pos_enc(pos.reshape(1, nx * ny, 2)).reshape(1, nx, ny, -1)
            u = attn_layer(u, pos_lst).view(b, nx, ny, -1) + u
        return u


class FactFormer2D(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, in_time_window, out_dim,
                 latent_multiplier=2.0, kernel_multiplier=2, max_latent_steps=10):
        super().__init__()
        self.max_latent_steps = max_latent_steps
        self.latent_dim = int(dim * latent_multiplier)
        self.to_in = nn.Linear(in_time_window, dim, bias=True)
        self.encoder = FactorizedTransformer(dim, dim_head, heads, dim, depth,
                                             kernel_multiplier=kernel_multiplier)
        self.expand_latent = nn.Linear(dim, self.latent_dim, bias=False)
        self.latent_time_emb = nn.Parameter(
            torch.randn(1, max_latent_steps, 1, 1, self.latent_dim) * 0.02,
            requires_grad=True
        )
        self.propagator = PreNorm(
            self.latent_dim,
            MLP([self.latent_dim, dim, self.latent_dim], act_fn=nn.GELU())
        )
        num_groups = int(8 * latent_multiplier)
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=num_groups, num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, dim, kernel_size=1, groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(dim // 2, out_dim, kernel_size=1, bias=True)
        )

    def forward(self, u, pos_lst, latent_steps=1):
        b, nx, ny, c = u.shape
        u = self.to_in(u)
        u = self.encoder(u, pos_lst)
        u = self.expand_latent(u)
        u_lst = []
        for l_t in range(latent_steps):
            u = u + self.latent_time_emb[:, l_t, ...]
            u = self.propagator(u) + u
            u_lst.append(u)
        u = torch.cat(u_lst, dim=0)
        u = self.simple_to_out(u)
        u = rearrange(u, '(t b) c (nx ny) -> b t nx ny c',
                      nx=nx, ny=ny, t=latent_steps)
        return u


def main():
    ap = argparse.ArgumentParser('FactFormer CPU inference timing')
    ap.add_argument('--checkpoint', default=os.path.join(FF_DIR, "checkpoints", "ns_factformer_our_split_best.pt"))
    ap.add_argument('--output_csv', default=os.path.join(RESULTS_DIR, "FactFormer_timing.csv"))
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=120.0)
    args = ap.parse_args()

    device = torch.device('cpu')
    H = W = 64
    T_in, T_out = 10, 10

    dim, depth, heads, dim_head = 128, 4, 8, 64
    latent_mult, kernel_mult = 2.0, 2

    model = FactFormer2D(
        dim=dim, depth=depth, heads=heads, dim_head=dim_head,
        in_time_window=T_in, out_dim=1,
        latent_multiplier=latent_mult, kernel_multiplier=kernel_mult,
        max_latent_steps=T_out,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    n_params = count_params(model)
    print(f"[FactFormer] params={n_params:,}")

    pos_lst = [torch.linspace(0, 2*math.pi, H).unsqueeze(-1).to(device),
               torch.linspace(0, 2*math.pi, W).unsqueeze(-1).to(device)]

    @torch.no_grad()
    def forward_fn(B: int):
        xu = torch.randn(B, H, W, T_in, device=device)
        out = model(xu, pos_lst, latent_steps=T_out)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'FactFormer', 'variant': '', 'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[FactFormer] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
