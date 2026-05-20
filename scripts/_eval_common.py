"""Shared utilities for the four evaluation scripts.

Loads a SpectraNet checkpoint together with its config JSON (produced by
``scripts/train_spectranet.py``) and reconstructs the model.  Replaces the
AST-extraction hack from the legacy eval scripts: with the package layout,
``from spectranet.model import SUNet2d`` Just Works.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectranet.config import KANConfig, SpectralConfig, default_kan_flags  # noqa: E402
from spectranet.model import SUNet2d  # noqa: E402


def infer_config_path(ckpt_path: str) -> Optional[str]:
    """Walk standard layouts to find the matching ``_config.json``."""
    base = os.path.basename(ckpt_path)
    for sfx in ("_best.pt", "_final.pt", "_checkpoint.pt"):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
            break
    d = os.path.dirname(os.path.abspath(ckpt_path))
    candidates = [
        os.path.join(d, "..", "plots", base + "_config.json"),
        os.path.join(d, "plots", base + "_config.json"),
        os.path.join(d, base + "_config.json"),
    ]
    for cand in candidates:
        cand = os.path.normpath(cand)
        if os.path.exists(cand):
            return cand
    return None


def load_spectranet(
    ckpt_path: str,
    config_path: Optional[str] = None,
    *,
    S: int = 64,
    T_in: int = 10,
    device: str = "cuda",
    cfg_overrides: Optional[dict] = None,
):
    """Load a SpectraNet checkpoint into a freshly constructed :class:`SUNet2d`.

    Parameters
    ----------
    ckpt_path : str
        Path to a ``*_best.pt``, ``*_final.pt``, or ``*_checkpoint.pt``.
    config_path : str | None
        Path to the matching ``*_config.json``.  Inferred from ``ckpt_path``
        when omitted.  If neither exists, a minimal fallback config is used
        (the SpectraNet at width=32, modes=12, levels=3, single
        head, no KAN, residual_target+two_step_lambda 0.1).
    S, T_in : int
        Native grid size and input-window length.  Override S to load the
        128² checkpoint at native resolution.
    cfg_overrides : dict | None
        Override individual config fields after loading the JSON.

    Returns
    -------
    model : SUNet2d
        On the requested device, in eval mode.
    cfg : dict
        The (possibly synthesised) config dictionary.
    """
    config_path = config_path or infer_config_path(ckpt_path)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg: dict[str, Any] = json.load(f)
    else:
        cfg = dict(
            modes=12, width=32, levels=3, ch_cap=2,
            skip_merge="add", bottleneck_attn="none", output_mode="single",
            encoder_skip_attn="none", decoder_out_attn="none",
            residual_target=True, two_step_lambda=0.1, no_kan=True,
            mode_truncation="box", mlp_groups=1,
            spectral_envelope=False, spectral_dropout=0.0,
            kan_flags={k: "std" for k in
                       ["fno_mlp", "residual_w", "projections", "lifting", "output", "spectral"]},
            kan_cfg={"kan_type": "efficient", "grid_size": 5, "spline_order": 3},
            dropout=0.0, T_in=T_in, T_out=10, step=1,
        )
    if cfg_overrides:
        cfg.update(cfg_overrides)

    kf = cfg.get("kan_flags", {})
    if isinstance(next(iter(kf.values())) if kf else None, str):
        kan_flags = {k: (v == "kan") for k, v in kf.items()}
    elif kf:
        kan_flags = {k: bool(v) for k, v in kf.items()}
    else:
        kan_flags = default_kan_flags()
    if cfg.get("no_kan"):
        kan_flags["output"] = False
    kc_raw = cfg.get("kan_cfg", {})
    kan_cfg = KANConfig(
        kan_type=kc_raw.get("kan_type", "efficient"),
        grid_size=kc_raw.get("grid_size", 5),
        spline_order=kc_raw.get("spline_order", 3),
    )
    spec_cfg = SpectralConfig(
        mode_truncation=cfg.get("mode_truncation", "box"),
        mlp_groups=cfg.get("mlp_groups", 1),
        spectral_envelope=cfg.get("spectral_envelope", False),
        spectral_dropout=cfg.get("spectral_dropout", 0.0),
    )

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = SUNet2d(
        base_modes=cfg["modes"], width=cfg["width"], in_channels=T_in, T_out=cfg.get("step", 1),
        levels=cfg["levels"], ch_cap=cfg.get("ch_cap", 2),
        skip_merge=cfg.get("skip_merge", "add"),
        bottleneck_attn=cfg.get("bottleneck_attn", "none"),
        output_mode=cfg.get("output_mode", "single"),
        encoder_skip_attn=cfg.get("encoder_skip_attn", "none"),
        decoder_out_attn=cfg.get("decoder_out_attn", "none"),
        S=S, kan_flags=kan_flags, kan_cfg=kan_cfg,
        dropout=cfg.get("dropout", 0.0), spectral_cfg=spec_cfg,
    ).to(dev)
    raw = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def free_rollout_ar2d(model, x_in: torch.Tensor, T_steps: int, residual_target: bool):
    """Run the autoregressive free rollout used by SpectraNet at test time.

    ``x_in``: ``(B, H, W, T_in)``.  Returns ``(B, H, W, T_steps)`` with no
    ground-truth feedback (each step's prediction feeds the next window).
    """
    cur = x_in.clone()
    pred = None
    for _ in range(T_steps):
        last = cur[..., -1:]
        raw = model(cur)
        y = (raw + last) if residual_target else raw
        pred = y if pred is None else torch.cat([pred, y], dim=-1)
        cur = torch.cat([cur[..., 1:], y], dim=-1)
    return pred


def position_grid(H: int, W: int) -> torch.Tensor:
    """``(N=H*W, 2)`` row-major position grid in [0,1]² (used by NSL models)."""
    xc, yc = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W))
    return torch.from_numpy(np.c_[xc.ravel(), yc.ravel()].astype(np.float32))
