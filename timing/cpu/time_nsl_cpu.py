"""CPU port of time_nsl.py — same model architecture + checkpoint, device='cpu'."""
import argparse
import importlib
import os
import sys
from argparse import Namespace

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import NSL_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, NSL_DIR)

from common_cpu import time_forward, count_params, write_rows  # noqa: E402

VALID = ['LSM', 'F_FNO', 'U_FNO', 'U_NO', 'MWT',
         'Galerkin_Transformer', 'FNO', 'U_Net', 'Transformer']

MODEL_DEFAULTS = {
    'LSM':                  {'n_hidden': 64,  'unified_pos': 0, 'mlp_ratio': 1},
    'F_FNO':                {'n_hidden': 20,  'unified_pos': 0, 'mlp_ratio': 2},
    'U_FNO':                {'n_hidden': 64,  'unified_pos': 0, 'mlp_ratio': 1},
    'U_NO':                 {'n_hidden': 64,  'unified_pos': 0, 'mlp_ratio': 1},
    'MWT':                  {'n_hidden': 256, 'unified_pos': 1, 'mlp_ratio': 2},
    'Galerkin_Transformer': {'n_hidden': 256, 'unified_pos': 1, 'mlp_ratio': 2},
    'FNO':                  {'n_hidden': 64,  'unified_pos': 0, 'mlp_ratio': 1},
    'U_Net':                {'n_hidden': 256, 'unified_pos': 1, 'mlp_ratio': 2},
    'Transformer':          {'n_hidden': 256, 'unified_pos': 1, 'mlp_ratio': 2},
}

_MODEL_MODULES = {
    'LSM':                  'models.LSM',
    'F_FNO':                'models.F_FNO',
    'U_FNO':                'models.U_FNO',
    'U_NO':                 'models.U_NO',
    'MWT':                  'models.MWT',
    'Galerkin_Transformer': 'models.Galerkin_Transformer',
    'FNO':                  'models.FNO',
    'U_Net':                'models.U_Net',
    'Transformer':          'models.Transformer',
}


def build_nsl_model(name: str, h: int = 64, T_in: int = 10):
    md = MODEL_DEFAULTS[name]
    model_args = Namespace(
        task='dynamic_autoregressive',
        geotype='structured_2D',
        shapelist=[h, h],
        unified_pos=md['unified_pos'],
        ref=8, fun_dim=T_in, space_dim=2, out_dim=1,
        n_hidden=md['n_hidden'], n_heads=8, n_layers=8,
        act='gelu', time_input=False, modes=12,
        mlp_ratio=md['mlp_ratio'], dropout=0.0, slice_num=32,
        model=name, mwt_k=3,
    )
    mod = importlib.import_module(_MODEL_MODULES[name])
    return mod.Model(model_args)


def main():
    ap = argparse.ArgumentParser('NSL CPU inference timing')
    ap.add_argument('--model_name', required=True, choices=VALID)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--output_csv', default=None)
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=120.0)
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(NSL_DIR, "checkpoints",
                                       f"ns_{args.model_name}_our_split_best.pt")
    if args.output_csv is None:
        args.output_csv = os.path.join(RESULTS_DIR, f"{args.model_name}_timing.csv")

    device = torch.device('cpu')
    h, T_in, T_out = 64, 10, 10

    print(f"[NSL/{args.model_name}] building model")
    model = build_nsl_model(args.model_name, h=h, T_in=T_in).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state and not any(
            k in state for k in ('weight', 'bias')):
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

    n_params = count_params(model)
    print(f"[NSL/{args.model_name}] params={n_params:,}  ckpt={args.checkpoint}")

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
                        extra={'model': args.model_name, 'variant': '', 'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[NSL/{args.model_name}] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
