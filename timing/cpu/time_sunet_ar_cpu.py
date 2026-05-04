"""CPU port of time_sunet_ar.py — runs on the local laptop, matches edge-device regime."""
import argparse
import ast
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import ABL_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, ABL_DIR)

from common_cpu import time_forward, count_params, write_rows  # noqa: E402


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
        base_modes=12, width=width, in_channels=10, T_out=1, levels=3,
        skip_merge='add', bottleneck_attn='none',
        output_mode='multiscale_mlp', encoder_skip_attn='none',
        decoder_out_attn='none', S=64,
        kan_flags=kan_flags, kan_cfg=kan_cfg,
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
        T_in=10, base_modes_xy=12, base_modes_t=10 // 2 + 1,
        width=20, levels=3, time_down='spatial',
        skip_merge='add', bottleneck_attn='none',
        output_mode='multiscale_mlp', encoder_skip_attn='none',
        decoder_out_attn='none', S=64,
        kan_flags=kan_flags, kan_cfg=kan_cfg,
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
    ap = argparse.ArgumentParser('AR S-UNet CPU inference timing')
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
            y_hat = model(cur)
            cur   = torch.cat((cur[..., step:], y_hat), dim=-1)

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
