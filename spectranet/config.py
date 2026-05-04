"""Spectral-block configuration.

Replaces the module-level ``AR2D_CFG`` global used in the original trainer with
an explicit, picklable dataclass.  Pass an instance to ``SUNet2d`` (or to
``SpectralConv2d`` / ``MLP2d`` directly) instead of relying on globals.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SpectralConfig:
    """Spectral-truncation, mode-shape, envelope, and dropout knobs.

    Parameters
    ----------
    mode_truncation : str
        ``'box'`` (rectangular cut at modes1×modes2; the FNO default) or
        ``'disk'`` (SO(2)-isotropic cut at ``kx² + ky² ≤ M²``).
    mlp_groups : int
        Group count for the 1×1 channel-mixing convolutions inside ``MLP2d``.
        Defaults to 1 (no grouping).  Set >1 for grouped channel mixing; the
        layer silently falls back to 1 when channel counts aren't divisible.
    spectral_envelope : bool
        Apply a per-output-channel learnable Gaussian envelope to the spectral
        weights (initialised at the cutoff so it is a no-op at start).
    spectral_dropout : float
        Bernoulli dropout probability applied to outer-ring spectral modes
        during training (no-op at eval).
    """

    mode_truncation: str = "box"
    mlp_groups: int = 1
    spectral_envelope: bool = False
    spectral_dropout: float = 0.0


@dataclass
class KANConfig:
    """KAN sub-layer configuration (kept for ablation reproducibility)."""

    kan_type: str = "efficient"   # 'efficient' or 'fourier'
    grid_size: int = 5
    spline_order: int = 3


def default_kan_flags() -> dict:
    """All-False KAN flags — the canonical SpectraNet uses no KAN sub-layers."""
    return {
        "fno_mlp": False,
        "residual_w": False,
        "projections": False,
        "lifting": False,
        "output": False,
        "spectral": False,
    }
