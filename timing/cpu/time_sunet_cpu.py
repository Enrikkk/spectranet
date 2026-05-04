"""CPU port of time_sunet.py."""
import argparse
import ast
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import ABL_DIR, SUNET_SRC, RESULTS_DIR  # noqa: E402
sys.path.insert(0, ABL_DIR)

from common_cpu import time_forward, count_params, write_rows  # noqa: E402


def _load_sunet3d():
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
    ap = argparse.ArgumentParser('S-UNet CPU inference timing')
    ap.add_argument('--variant', required=True, choices=list(CKPTS.keys()))
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_csv', default=None)
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=180.0)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = CKPTS[args.variant]
    if args.output_csv is None:
        args.output_csv = os.path.join(RESULTS_DIR, f"SUNet_{args.variant}_timing.csv")

    device = torch.device('cpu')
    S, T_in, T_out = 64, 10, 10

    SUNet3d = _load_sunet3d()

    kan_flags = {
        'fno_mlp':     False,
        'residual_w':  False,
        'projections': False,
        'lifting':     False,
        'output':      True,
        'spectral':    False,
    }
    kan_cfg = dict(kan_type='efficient', grid_size=5, spline_order=3)

    model = SUNet3d(
        base_modes_xy=12, base_modes_t=T_out // 2 + 1, width=26,
        in_channels=10, levels=3, time_down='spatial',
        skip_merge='add', bottleneck_attn='none',
        output_mode='multiscale_mlp', encoder_skip_attn='none',
        decoder_out_attn='none', S=S, T=T_out + 1,
        kan_flags=kan_flags, kan_cfg=kan_cfg,
    ).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[SUNet/{args.variant}] params={n_params:,}  ckpt={os.path.basename(args.checkpoint)}")

    @torch.no_grad()
    def forward_fn(B: int):
        x = torch.randn(B, S, S, T_out, T_in, device=device)
        out = model(x)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'SUNet', 'variant': args.variant,
                               'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[SUNet/{args.variant}] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
