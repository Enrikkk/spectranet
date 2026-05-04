"""SpectraNet training script.

One unified trainer for the canonical SpectraNet across all 7 dataset
families.  Replaces the legacy collection of near-duplicate trainer scripts
(``ns_sunet_ar2d.py`` and its 6 cross-PDE variants ``_am``, ``_dr``, ``_sw``,
``_v1e3``, ``_v1e4``, ``_128``).

Usage::

    # canonical NS ν = 1e-5 reproduction
    python scripts/train_spectranet.py \
           --config configs/spectranet_ns_v1e5.yaml \
           --data_root ./data

    # CLI flags override config values, both override defaults
    python scripts/train_spectranet.py \
           --config configs/spectranet_ns_v1e5.yaml \
           --data_root ./data \
           --epochs 50 --seed 1
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import scipy.io as scio
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectranet.config import KANConfig, SpectralConfig, default_kan_flags
from spectranet.data import DATASETS, load_dataset
from spectranet.losses import LpLoss
from spectranet.model import SUNet2d
from spectranet.utils import count_params, make_run_tag, seed_all


# ─── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train SpectraNet on a PDE benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data + run management
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (CLI flags override config values)")
    p.add_argument("--dataset", type=str, default="ns_v1e5", choices=sorted(DATASETS.keys()))
    p.add_argument("--data_root", type=str, default="./data",
                   help="Directory containing the dataset .mat file")
    p.add_argument("--output_dir", type=str, default="./runs",
                   help="Directory to write logs/, model/, plots/, pred/")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <output_dir>/<tag>_checkpoint.pt if it exists")

    # Architecture
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--levels", type=int, default=3, choices=[2, 3])
    p.add_argument("--ch_cap", type=int, default=2,
                   help="Channel-doubling cap (canonical 2; set 3 to widen the bottleneck)")
    p.add_argument("--skip_merge", type=str, default="add",
                   choices=["add", "concat", "concat_kan", "add_kan", "add_kan_res",
                            "spectral_attn", "attn_gate", "kan_attn_gate"])
    p.add_argument("--bottleneck_attn", type=str, default="none",
                   choices=["none", "self_attn", "channel", "spatial_channel"])
    p.add_argument("--output_mode", type=str, default="single",
                   choices=["single", "multiscale_mlp", "multiscale_kan"],
                   help="Canonical SpectraNet uses 'single'.  'multiscale_mlp' is the decorated head.")
    p.add_argument("--encoder_skip_attn", type=str, default="none",
                   choices=["none", "channel", "spatial", "both", "progressive"])
    p.add_argument("--decoder_out_attn", type=str, default="none",
                   choices=["none", "channel", "spatial", "both", "progressive"])
    p.add_argument("--dropout", type=float, default=0.0)

    # KAN sub-layer switches (canonical: all standard)
    for name in ["fno_mlp", "residual_w", "projections", "lifting", "output", "spectral"]:
        default_val = "kan" if name == "output" else (
            "linear" if name in ("residual_w", "projections", "lifting", "spectral") else "mlp"
        )
        p.add_argument(f"--{name}", type=str, default=default_val,
                       choices=["mlp", "kan"] if name in ("fno_mlp", "output") else ["linear", "kan"])
    p.add_argument("--kan_type", type=str, default="efficient", choices=["efficient", "fourier"])
    p.add_argument("--kan_grid_size", type=int, default=5)
    p.add_argument("--kan_spline_order", type=int, default=3)
    p.add_argument("--no_kan", action="store_true",
                   help="Force Linear output head regardless of --output (E7).")

    # Spectral block (E2 / E5 / E8)
    p.add_argument("--mode_truncation", type=str, default="box", choices=["box", "disk"])
    p.add_argument("--mlp_groups", type=int, default=1)
    p.add_argument("--spectral_envelope", action="store_true")
    p.add_argument("--spectral_dropout", type=float, default=0.0)

    # Residual + multistep (E3, E6)
    p.add_argument("--residual_target", action="store_true",
                   help="Predict Δu = u_{t+1} − u_t and integrate at inference (E3).")
    p.add_argument("--two_step_lambda", type=float, default=0.0,
                   help="Weight on the f(f(u_t)) ≈ u_{t+2} semigroup-consistency penalty (E6).")

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # Window
    p.add_argument("--T_in", type=int, default=10)
    p.add_argument("--T_out", type=int, default=10)
    p.add_argument("--step", type=int, default=1)
    return p


def merge_yaml_into_args(args: argparse.Namespace, yaml_path: str) -> argparse.Namespace:
    """Update *args* in place from YAML, but never overwrite a value the user
    typed on the CLI.  Returns the merged namespace."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    cli_provided = {
        a.lstrip("-").split("=")[0].replace("-", "_")
        for a in sys.argv[1:]
        if a.startswith("--")
    }
    for k, v in cfg.items():
        if k in cli_provided:
            continue
        if not hasattr(args, k):
            print(f"[train] WARN: unknown config key '{k}' (ignored)")
            continue
        setattr(args, k, v)
    return args


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()
    if args.config:
        args = merge_yaml_into_args(args, args.config)
    seed_all(args.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    splits, info = load_dataset(args.dataset, args.data_root, T_in=args.T_in, T_out=args.T_out)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = splits
    ntrain = X_train.shape[0]
    nval = X_val.shape[0]
    ntest = X_test.shape[0]
    S = info["S"]

    # ── Run tag + output dirs ───────────────────────────────────────────────
    args.ntrain = ntrain
    tag = make_run_tag(args)
    out_root = Path(args.output_dir)
    (out_root / "model").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "pred").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "model" / f"{tag}_checkpoint.pt"
    best_path = out_root / "model" / f"{tag}_best.pt"
    final_path = out_root / "model" / f"{tag}_final.pt"
    loss_csv = out_root / "plots" / f"{tag}_losses.csv"
    test_csv = out_root / "plots" / f"{tag}_test_results.csv"
    cfg_json = out_root / "plots" / f"{tag}_config.json"
    pred_mat = out_root / "pred" / f"{tag}.mat"

    print(f"[train] tag = {tag}")
    print(f"[train] split: ntrain={ntrain}  nval={nval}  ntest={ntest}  S={S}")

    # ── KAN flags ───────────────────────────────────────────────────────────
    kan_flags = {
        "fno_mlp":     args.fno_mlp == "kan",
        "residual_w":  args.residual_w == "kan",
        "projections": args.projections == "kan",
        "lifting":     args.lifting == "kan",
        "output":      args.output == "kan",
        "spectral":    args.spectral == "kan",
    }
    if args.no_kan:
        kan_flags["output"] = False
        if args.output_mode == "multiscale_kan":
            args.output_mode = "multiscale_mlp"

    kan_cfg = KANConfig(
        kan_type=args.kan_type,
        grid_size=args.kan_grid_size,
        spline_order=args.kan_spline_order,
    )
    spec_cfg = SpectralConfig(
        mode_truncation=args.mode_truncation,
        mlp_groups=args.mlp_groups,
        spectral_envelope=args.spectral_envelope,
        spectral_dropout=args.spectral_dropout,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    model = SUNet2d(
        base_modes=args.modes, width=args.width, in_channels=args.T_in, T_out=args.step,
        levels=args.levels, ch_cap=args.ch_cap,
        skip_merge=args.skip_merge, bottleneck_attn=args.bottleneck_attn,
        output_mode=args.output_mode,
        encoder_skip_attn=args.encoder_skip_attn, decoder_out_attn=args.decoder_out_attn,
        S=S, kan_flags=kan_flags, kan_cfg=kan_cfg,
        dropout=args.dropout, spectral_cfg=spec_cfg,
    ).to(device)
    n_params = count_params(model)
    print(f"[train] parameters: {n_params:,}")

    # ── Loaders ─────────────────────────────────────────────────────────────
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=args.batch_size, shuffle=True, generator=gen,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), batch_size=20, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test), batch_size=1, shuffle=False,
    )

    # ── Optimizer + scheduler ───────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_l2 = float("inf")
    best_val_epoch = -1
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val_l2 = ck["best_val_l2"]
        best_val_epoch = ck["best_val_epoch"]
        print(f"[train] resumed from epoch {ck['epoch']}, best_val={best_val_l2:.4f}")

    # ── Training (teacher-forcing AR) ───────────────────────────────────────
    myloss = LpLoss(size_average=False)
    csv_mode = "a" if (args.resume and ckpt_path.exists()) else "w"
    losses_file = open(loss_csv, csv_mode, newline="")
    losses_writer = csv.writer(losses_file)
    if csv_mode == "w":
        losses_writer.writerow(["epoch", "train_l2", "val_l2"])

    t_start = default_timer()
    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="SpectraNet", unit="epoch")
    T_out = args.T_out
    step = args.step
    for ep in epoch_bar:
        model.train()
        train_full = 0.0
        for x_window, y_full in train_loader:
            x_window = x_window.to(device); y_full = y_full.to(device)
            bsz = x_window.shape[0]
            cur = x_window
            loss, pred = 0.0, None
            for t in range(0, T_out, step):
                y_t = y_full[..., t : t + step]
                last_in = cur[..., -step:]
                raw = model(cur)
                y_hat = (raw + last_in) if args.residual_target else raw
                tgt_train = (y_t - last_in) if args.residual_target else y_t
                raw_tgt = tgt_train if args.residual_target else y_t
                loss = loss + myloss(raw.reshape(bsz, -1), raw_tgt.reshape(bsz, -1))
                if args.two_step_lambda > 0 and (t + 2 * step) <= T_out:
                    cur2 = torch.cat((cur[..., step:], y_hat), dim=-1)
                    raw2 = model(cur2)
                    y_hat2 = (raw2 + y_hat) if args.residual_target else raw2
                    y_t2 = y_full[..., t + step : t + 2 * step]
                    loss = loss + args.two_step_lambda * myloss(
                        y_hat2.reshape(bsz, -1), y_t2.reshape(bsz, -1)
                    )
                pred = y_hat if pred is None else torch.cat((pred, y_hat), dim=-1)
                cur = torch.cat((cur[..., step:], y_t), dim=-1)
            train_full += myloss(pred.reshape(bsz, -1), y_full.reshape(bsz, -1)).item()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        train_l2 = train_full / ntrain

        # Validation: free rollout
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x_window, y_full in val_loader:
                x_window = x_window.to(device); y_full = y_full.to(device)
                bsz = x_window.shape[0]
                cur, pred = x_window, None
                for t in range(0, T_out, step):
                    last_in = cur[..., -step:]
                    raw = model(cur)
                    y_hat = (raw + last_in) if args.residual_target else raw
                    pred = y_hat if pred is None else torch.cat((pred, y_hat), dim=-1)
                    cur = torch.cat((cur[..., step:], y_hat), dim=-1)
                val_total += myloss(pred.reshape(bsz, -1), y_full.reshape(bsz, -1)).item()
        val_l2 = val_total / nval
        losses_writer.writerow([ep + 1, train_l2, val_l2]); losses_file.flush()

        if val_l2 < best_val_l2:
            best_val_l2 = val_l2
            best_val_epoch = ep + 1
            torch.save(model.state_dict(), best_path)

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "epoch": ep,
            "best_val_l2": best_val_l2,
            "best_val_epoch": best_val_epoch,
        }, ckpt_path)
        epoch_bar.set_postfix(
            train=f"{train_l2:.4f}", val=f"{val_l2:.4f}",
            best=f"{best_val_l2:.4f}@{best_val_epoch}",
        )

    losses_file.close()
    torch.save(model.state_dict(), final_path)
    print(f"\n[train] training done — best val: epoch {best_val_epoch}, val L2 = {best_val_l2:.4f}")
    print(f"[train] total time: {(default_timer() - t_start)/60:.1f} min")

    # ── Held-out test (free rollout, both checkpoints) ──────────────────────
    def _eval_test(ckpt_path_eval, label):
        state = torch.load(ckpt_path_eval, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        per = []
        total = 0.0
        pred_all = torch.zeros(Y_test.shape)
        truth_all = torch.zeros(Y_test.shape)
        idx = 0
        with torch.no_grad():
            for x_window, y_full in test_loader:
                x_window = x_window.to(device); y_full = y_full.to(device)
                bsz = x_window.shape[0]
                cur, pred = x_window, None
                for t in range(0, T_out, step):
                    last_in = cur[..., -step:]
                    raw = model(cur)
                    y_hat = (raw + last_in) if args.residual_target else raw
                    pred = y_hat if pred is None else torch.cat((pred, y_hat), dim=-1)
                    cur = torch.cat((cur[..., step:], y_hat), dim=-1)
                pred_all[idx] = pred.cpu().squeeze(0)
                truth_all[idx] = y_full.cpu().squeeze(0)
                s = myloss(pred.view(1, -1), y_full.view(1, -1)).item()
                per.append(s); total += s; idx += 1
        mean = total / ntest
        print(f"[train/{label}] test L2 = {mean:.4f}")
        return mean, pred_all, truth_all, per

    print("\n── Held-out test evaluation ──")
    best_test_l2, pred_best, truth_best, per_best = _eval_test(best_path, "best-val")
    final_test_l2, pred_final, truth_final, per_final = _eval_test(final_path, "final-ep")

    with open(test_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "test_l2_best", "test_l2_final"])
        for si, (sb, sf) in enumerate(zip(per_best, per_final)):
            w.writerow([si, sb, sf])

    cfg = {
        "model": "SpectraNet",
        "dataset": args.dataset,
        "dataset_file": info["filename"],
        "optimizer": "AdamW", "weight_decay": args.weight_decay,
        "scheduler": "OneCycleLR", "max_lr": args.lr,
        "levels": args.levels, "width": args.width, "modes": args.modes, "ch_cap": args.ch_cap,
        "skip_merge": args.skip_merge, "bottleneck_attn": args.bottleneck_attn,
        "output_mode": args.output_mode,
        "encoder_skip_attn": args.encoder_skip_attn,
        "decoder_out_attn": args.decoder_out_attn,
        "T_in": args.T_in, "T_out": args.T_out, "step": args.step,
        "batch_size": args.batch_size, "lr": args.lr, "epochs": args.epochs,
        "ntrain": ntrain, "nval": nval, "ntest": ntest, "S": S,
        "kan_flags": kan_flags, "kan_cfg": vars(kan_cfg), "n_params": n_params,
        "dropout": args.dropout, "seed": args.seed,
        "residual_target": args.residual_target, "no_kan": args.no_kan,
        "mode_truncation": args.mode_truncation,
        "mlp_groups": args.mlp_groups, "two_step_lambda": args.two_step_lambda,
        "spectral_envelope": args.spectral_envelope,
        "spectral_dropout": args.spectral_dropout,
        "best_val_epoch": best_val_epoch, "best_val_l2": best_val_l2,
        "best_test_l2": best_test_l2, "final_test_l2": final_test_l2,
    }
    with open(cfg_json, "w") as f:
        json.dump(cfg, f, indent=2)

    scio.savemat(pred_mat, mdict={"pred": pred_best.numpy(), "truth": truth_best.numpy()})
    print(f"[train] artifacts written under {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
