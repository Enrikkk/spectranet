"""Param-count sanity tests.

These tests do not require a GPU, a dataset, or any pretrained weights.
They confirm that the package's `SUNet2d` constructor reproduces the exact
parameter counts cited in the paper.
"""

from __future__ import annotations
import pytest

import torch  # noqa: F401  (failure here is the most informative)

from spectranet.model import SUNet2d
from spectranet.utils import count_params


def _make(**overrides):
    """Default canonical configuration; ``overrides`` selectively diverges."""
    cfg = dict(
        base_modes=12, width=32, in_channels=10, T_out=1,
        levels=3, ch_cap=2,
        skip_merge="add", bottleneck_attn="none", output_mode="single",
        S=64,
    )
    cfg.update(overrides)
    return SUNet2d(**cfg)


def test_canonical_param_count():
    """Headline configuration → 2,040,705 params (paper Section 6)."""
    m = _make(output_mode="single")
    assert count_params(m) == 2040705


def test_decorated_head_param_count():
    """Decorated multi-resolution head → 2,124,166 params (paper Table 7)."""
    m = _make(output_mode="multiscale_mlp")
    assert count_params(m) == 2124166


def test_bottleneck_widened_param_count():
    """``ch_cap=3`` Session-24 capacity sanity check → 2,319,745 params."""
    m = _make(ch_cap=3)
    assert count_params(m) == 2319745


def test_forward_shape():
    """Forward pass must preserve the (B, X, Y, T_out) contract."""
    m = _make()
    x = torch.randn(2, 64, 64, 10)
    y = m(x)
    assert y.shape == (2, 64, 64, 1)


def test_forward_different_resolution():
    """SpectraNet's spectral path is resolution-invariant: a 128² input
    should produce a 128² output without modifying the model."""
    m = _make()
    x = torch.randn(1, 128, 128, 10)
    y = m(x)
    assert y.shape == (1, 128, 128, 1)


@pytest.mark.parametrize("output_mode", ["single", "multiscale_mlp"])
def test_state_dict_round_trip(output_mode):
    """save → load must succeed without missing/unexpected keys."""
    m1 = _make(output_mode=output_mode)
    m2 = _make(output_mode=output_mode)
    res = m2.load_state_dict(m1.state_dict())
    assert not res.missing_keys
    assert not res.unexpected_keys
