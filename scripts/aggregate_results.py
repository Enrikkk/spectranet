#!/usr/bin/env python3
"""Aggregate per-run training artifacts into the canonical ``results/leaderboard.csv``.

Walks ``--input`` (default: ``runs/plots/``) for ``*_config.json`` files
produced by ``scripts/train_spectranet.py``, extracts the headline numbers,
and writes a sorted CSV.  Used at the end of a multi-dataset reproduction
to fold per-run JSONs into a single table that ``scripts/figures/`` can plot.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./runs/plots", help="Directory of *_config.json files")
    p.add_argument("--out", default="./results/leaderboard_runs.csv")
    p.add_argument("--pattern", default="*_config.json")
    args = p.parse_args()

    rows = []
    for cfg_path in sorted(Path(args.input).rglob(args.pattern)):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"  skip {cfg_path}: {e}")
            continue
        rows.append({
            "tag": cfg_path.stem.replace("_config", ""),
            "model": cfg.get("model", "?"),
            "dataset": cfg.get("dataset", "?"),
            "params": cfg.get("n_params", 0),
            "best_val_l2": cfg.get("best_val_l2", float("nan")),
            "best_val_epoch": cfg.get("best_val_epoch", -1),
            "best_test_l2": cfg.get("best_test_l2", float("nan")),
            "final_test_l2": cfg.get("final_test_l2", float("nan")),
            "epochs": cfg.get("epochs", 0),
            "width": cfg.get("width", 0),
            "modes": cfg.get("modes", 0),
            "levels": cfg.get("levels", 0),
            "ch_cap": cfg.get("ch_cap", 2),
            "output_mode": cfg.get("output_mode", "?"),
            "residual_target": cfg.get("residual_target", False),
            "two_step_lambda": cfg.get("two_step_lambda", 0.0),
            "no_kan": cfg.get("no_kan", False),
            "seed": cfg.get("seed", 0),
        })

    if not rows:
        print(f"[aggregate] no *_config.json files found under {args.input}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["dataset"], r["best_test_l2"])):
            w.writerow(row)
    print(f"[aggregate] {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # noqa: F821
