"""CPU port of time_kno.py."""
import argparse
import importlib.util
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from paths_cpu import KNO_DIR, RESULTS_DIR  # noqa: E402
sys.path.insert(0, KNO_DIR)

from common_cpu import time_forward, count_params, write_rows  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kno_module",
    os.path.join(KNO_DIR, "koopmanlab", "models", "kno.py"))
_kno = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_kno)
KNO2d = _kno.KNO2d
encoder_conv2d = _kno.encoder_conv2d
decoder_conv2d = _kno.decoder_conv2d


def main():
    ap = argparse.ArgumentParser('KNO2d CPU inference timing')
    ap.add_argument('--checkpoint', default=os.path.join(KNO_DIR, "checkpoints", "ns_kno_our_split_best.pt"))
    ap.add_argument('--output_csv', default=os.path.join(RESULTS_DIR, "KNO2d_timing.csv"))
    ap.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters',  type=int, default=20)
    ap.add_argument('--per_iter_timeout_s', type=float, default=120.0)
    args = ap.parse_args()

    device = torch.device('cpu')
    H = W = 64
    T_in, T_out = 10, 10

    op_size, modes, decompose = 32, 16, 6
    enc = encoder_conv2d(t_len=T_in, op_size=op_size)
    dec = decoder_conv2d(t_len=T_out, op_size=op_size)
    model = KNO2d(enc, dec, op_size=op_size,
                  modes_x=modes, modes_y=modes,
                  decompose=decompose, linear_type=True).to(device).eval()

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    n_params = count_params(model)
    print(f"[KNO2d] params={n_params:,}")

    @torch.no_grad()
    def forward_fn(B: int):
        inp = torch.randn(B, H, W, T_in, device=device)
        pred, rec = model(inp)

    rows = time_forward(forward_fn,
                        batch_sizes=tuple(args.batch_sizes),
                        warmup=args.warmup, iters=args.iters,
                        extra={'model': 'KNO2d', 'variant': '', 'params': n_params},
                        per_iter_timeout_s=args.per_iter_timeout_s)
    write_rows(args.output_csv, rows)
    print(f"[KNO2d] wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == '__main__':
    main()
