"""Inference timing for autoregressive S-UNet (AR-2D and AR-3D), GPU.

Two variants:
  --variant ar2d  : ns_sunet_ar2d.py / SUNet2d with T_out=step=1
  --variant ar3d  : ns_sunet_ar3d.py / SUNet3dAR (wraps a 3D-spectral SUNet3d
                    over the input window + Linear(T_in,1) temporal collapse)

Both take input (B, 64, 64, T_in=10), emit (B, 64, 64, 1) per call, and
rollout 10 steps to produce a full T_out=10 prediction. Unit of work timed
here is the **full 10-step rollout**.

Class extraction follows the same AST-filter trick as time_sunet.py: keep
only Imports + ClassDefs + FunctionDefs from the original training script,
exec in a fresh namespace; gets us the model classes without triggering the
data-load / training side effects at module level.
"""
import argparse
import ast
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ABL_DIR  = "./data/FNO/AttentionAblation"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ABL_DIR)

from common import time_forward, count_params, write_rows  # noqa: E402


SRC = {
    'ar2d':     os.path.join(ABL_DIR, 'ns_sunet_ar2d.py'),
    'ar2d_w32': os.path.join(ABL_DIR, 'ns_sunet_ar2d.py'),
    'ar2d_w48': os.path.join(ABL_DIR, 'ns_sunet_ar2d.py'),
    'ar2d_w64': os.path.join(ABL_DIR, 'ns_sunet_ar2d.py'),
    'ar3d':     os.path.join(ABL_DIR, 'ns_sunet_ar3d.py'),
}

CKPTS = {
    'ar2d': os.path.join(
        ABL_DIR, 'model',
        'sunet_ar2d_ns5_L3_smadd_bno_omm_eno_dno_w20_fmm_rwl_pjl_lfl_opk_spl_kte_g5_'
        'N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
    'ar2d_w32': os.path.join(
        ABL_DIR, 'model',
        'sunet_ar2d_ns5_L3_smadd_bno_omm_eno_dno_w32_fmm_rwl_pjl_lfl_opk_spl_kte_g5_'
        'N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
    'ar2d_w48': os.path.join(
        ABL_DIR, 'model',
        'sunet_ar2d_ns5_L3_smadd_bno_omm_eno_dno_w48_fmm_rwl_pjl_lfl_opk_spl_kte_g5_'
        'N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
    'ar2d_w64': os.path.join(
        ABL_DIR, 'model',
        'sunet_ar2d_ns5_L3_smadd_bno_omm_eno_dno_w64_fmm_rwl_pjl_lfl_opk_spl_kte_g5_'
        'N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
    'ar3d': os.path.join(
        ABL_DIR, 'model',
        'sunet_ar3d_ns5_tdsp_L3_smadd_bno_omm_eno_dno_w20_fmm_rwl_pjl_lfl_opk_spl_kte_g5_'
        'N850_ep500_m12_S64_lr1e-03_adamw_oclr_best.pt'),
}


def _load_classes(src_path):
    with open(src_path) as f:
        src = f.read()
    tree = ast.parse(src)
    keep_types = (ast.Import, ast.ImportFrom, ast.ClassDef,
                  ast.FunctionDef, ast.AsyncFunctionDef)
    tree.body = [n for n in tree.body if isinstance(n, keep_types)]
    code = compile(tree, f'<ar_extracted:{os.path.basename(src_path)}>', 'exec')
    ns = {'__name__': '_ar_lib', '__file__': src_path}
    exec(code, ns)
    return ns


def _build_ar2d_with_width(device, width):
    ns = _load_classes(SRC['ar2d'])
    SUNet2d = ns['SUNet2d']
    kan_flags = dict(fno_mlp=False, residual_w=False, projections=False,
                     lifting=False, output=True, spectral=False)
    kan_cfg = dict(kan_type='efficient', grid_size=5, spline_order=3)
    model = SUNet2d(
        base_modes        = 12,
        width             = width,
        in_channels       = 10,    # T_in=10 past frames as channels
        T_out             = 1,     # step=1 — single next frame
        levels            = 3,
        skip_merge        = 'add',
        bottleneck_attn   = 'none',
        output_mode       = 'multiscale_mlp',
        encoder_skip_attn = 'none',
        decoder_out_attn  = 'none',
        S                 = 64,
        kan_flags         = kan_flags,
        kan_cfg           = kan_cfg,
    ).to(device).eval()
    return model


def _build_ar2d(device):     return _build_ar2d_with_width(device, 20)
def _build_ar2d_w32(device): return _build_ar2d_with_width(device, 32)
def _build_ar2d_w48(device): return _build_ar2d_with_width(device, 48)
def _build_ar2d_w64(device): return _build_ar2d_with_width(device, 64)


def _build_ar3d(device):
    ns = _load_classes(SRC['ar3d'])
    SUNet3dAR = ns['SUNet3dAR']
    kan_flags = dict(fno_mlp=False, residual_w=False, projections=False,
                     lifting=False, output=True, spectral=False)
    kan_cfg = dict(kan_type='efficient', grid_size=5, spline_order=3)
    model = SUNet3dAR(
        T_in              = 10,
        base_modes_xy     = 12,
        base_modes_t      = 10 // 2 + 1,   # 6
        width             = 20,
        levels            = 3,
        time_down         = 'spatial',
        skip_merge        = 'add',
        bottleneck_attn   = 'none',
        output_mode       = 'multiscale_mlp',
        encoder_skip_attn = 'none',
        decoder_out_attn  = 'none',
        S                 = 64,
        kan_flags         = kan_flags,
        kan_cfg           = kan_cfg,
    ).to(device).eval()
    return model


BUILDERS = {
    'ar2d':     _build_ar2d,
    'ar2d_w32': _build_ar2d_w32,
    'ar2d_w48': _build_ar2d_w48,
    'ar2d_w64': _build_ar2d_w64,
    'ar3d':     _build_ar3d,
}


def main():
    ap = argparse.ArgumentParser('AR S-UNet GPU inference timing')
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
    S, T_in, T_out, step = 64, 10, 10, 1

    model = BUILDERS[args.variant](device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    n_params = count_params(model)
    print(f"[SUNet/{args.variant}] params={n_params:,}  ckpt={os.path.basename(args.checkpoint)}")

    @torch.no_grad()
    def forward_fn(B: int):
        cur = torch.randn(B, S, S, T_in, device=device)
        for _ in range(0, T_out, step):
            y_hat = model(cur)                          # (B, S, S, 1)
            cur   = torch.cat((cur[..., step:], y_hat), dim=-1)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'SUNet', 'variant': args.variant,
                               'params': n_params})
    write_rows(args.output_csv, rows)
    print(f"[SUNet/{args.variant}] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
