"""Inference timing for S-UNet (SUNet3d). Single forward, non-AR.

Two variants:
  --variant ablation_best : sunet_ns5_attn_..._N850_ep500_..._best.pt
  --variant goldstd       : sunet_ns5_goldstd_..._lr1e-03_adamw_oclr_best.pt

Both share the ablation-winner architecture
  (tdsp L3 smadd bno omm eno dno w26 fmm rwl pjl lfl opk spl kte g5 m12 S64).

Class extraction: ns_sunet_goldstd.py defines SUNet3d + helpers inline and
runs data-load/training at module level. Importing it would load 800MB and
start training. We use ast to keep only Imports/ClassDefs/FunctionDefs and
exec that filtered module body — gets the classes without side effects.
"""
import argparse
import ast
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUNET_SRC = "./ns_sunet_goldstd.py"
ABL_DIR   = "./data/FNO/AttentionAblation"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ABL_DIR)

from common import time_forward, count_params, write_rows  # noqa: E402


def _load_sunet3d():
    """AST-filter ns_sunet_goldstd.py down to imports + class/function defs,
    exec in a fresh namespace, return SUNet3d + count_params helper."""
    with open(SUNET_SRC) as f:
        src = f.read()
    tree = ast.parse(src)
    keep_types = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef,
                  ast.AsyncFunctionDef)
    tree.body = [n for n in tree.body if isinstance(n, keep_types)]
    code = compile(tree, '<sunet_extracted>', 'exec')
    ns = {'__name__': '_sunet_lib', '__file__': SUNET_SRC}
    exec(code, ns)
    return ns['SUNet3d']


CKPTS = {
    'ablation_best': os.path.join(
        ABL_DIR, 'model',
        'sunet_ns5_attn_tdsp_L3_smadd_bno_omm_eno_dno_w26_fmm_rwl_pjl_lfl_opk_spl_kte_g5_N850_ep500_m12_S64_best.pt'),
    'goldstd': os.path.join(
        ABL_DIR, 'model',
        'sunet_ns5_goldstd_tdsp_L3_smadd_bno_omm_eno_dno_w26_fmm_rwl_pjl_lfl_opk_spl_kte_g5_N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
}


def main():
    ap = argparse.ArgumentParser('S-UNet inference timing')
    ap.add_argument('--variant', required=True, choices=list(CKPTS.keys()))
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_csv', default=None)
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters',  type=int, default=50)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = CKPTS[args.variant]
    if args.output_csv is None:
        args.output_csv = f"{THIS_DIR}/../results/SUNet_{args.variant}_timing.csv"

    device = torch.device('cuda')
    S, T_in, T_out = 64, 10, 10

    SUNet3d = _load_sunet3d()

    # ablation-winner architecture (matches both ckpts)
    kan_flags = {
        'fno_mlp':     False,   # fmm = mlp
        'residual_w':  False,   # rwl = linear
        'projections': False,   # pjl = linear
        'lifting':     False,   # lfl = linear
        'output':      True,    # opk = kan
        'spectral':    False,   # spl = linear
    }
    kan_cfg = dict(kan_type='efficient', grid_size=5, spline_order=3)

    model = SUNet3d(
        base_modes_xy     = 12,
        base_modes_t      = T_out // 2 + 1,
        width             = 26,
        in_channels       = 10,        # T_cond * N_FIELDS
        levels            = 3,
        time_down         = 'spatial',
        skip_merge        = 'add',
        bottleneck_attn   = 'none',
        output_mode       = 'multiscale_mlp',
        encoder_skip_attn = 'none',
        decoder_out_attn  = 'none',
        S                 = S,
        T                 = T_out + 1,
        kan_flags         = kan_flags,
        kan_cfg           = kan_cfg,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[SUNet/{args.variant}] params={n_params:,}  ckpt={os.path.basename(args.checkpoint)}")

    # Input shape mirrors prep_X in goldstd:
    #   X = u[..., :T_cond] -> (N, S, S, T_cond) -> (N, S, S, 1, T_cond)
    #   prep_X repeats along T_pred axis -> (N, S, S, T_pred, T_cond)
    @torch.no_grad()
    def forward_fn(B: int):
        x = torch.randn(B, S, S, T_out, T_in, device=device)
        out = model(x)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'SUNet', 'variant': args.variant,
                               'params': n_params})
    write_rows(args.output_csv, rows)
    print(f"[SUNet/{args.variant}] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
