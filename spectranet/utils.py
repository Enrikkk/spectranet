"""Generic utilities: parameter counting, seeding, run tags."""

from __future__ import annotations
import os
import random
import operator
from functools import reduce
from typing import Any

import numpy as np
import torch


def count_params(model: torch.nn.Module) -> int:
    """Standard PyTorch parameter count: ``sum(p.numel() for p in model.parameters())``.

    This is the convention used throughout the paper.  A complex parameter of
    shape ``(a, b)`` contributes ``a*b`` (one tensor element), regardless of
    real/imag halves.
    """
    return sum(p.numel() for p in model.parameters())


def count_params_complex_doubled(model: torch.nn.Module) -> int:
    """Complex-doubled parameter count for compatibility with older convention.

    Counts each complex-valued parameter as 2× its element count.  Used in some
    earlier neural-operator papers; included here only for cross-checking
    legacy numbers.
    """
    c = 0
    for p in model.parameters():
        c += reduce(
            operator.mul,
            list(p.size() + (2,) if p.is_complex() else p.size()),
        )
    return c


def seed_all(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices).

    Does not configure ``torch.use_deterministic_algorithms`` — set that flag
    yourself if you need bit-for-bit determinism (it can slow training down).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_tag(args: Any, *, prefix: str = "spectranet") -> str:
    """Build a deterministic filename-safe tag from the run-config namespace.

    The tag captures every knob that affects the trained checkpoint (width,
    modes, levels, residual_target, two_step_lambda, no_kan, output_mode,
    seed, channel-cap …).  Two runs with identical tags should produce
    bit-identical checkpoints (modulo nondeterministic CUDA kernels).
    """

    def g(name, default=None):
        return getattr(args, name, default)

    parts = [
        prefix,
        f"w{g('width', 32)}",
        f"m{g('modes', 12)}",
        f"L{g('levels', 3)}",
        f"sm{g('skip_merge', 'add')}",
        f"om{g('output_mode', 'single')}",
        f"ep{g('epochs', 500)}",
        f"N{g('ntrain', 850)}",
    ]
    if g("residual_target", False):
        parts.append("resi")
    if g("no_kan", False):
        parts.append("nokan")
    if g("two_step_lambda", 0.0) > 0:
        parts.append(f"ts{g('two_step_lambda'):g}")
    if g("mode_truncation", "box") != "box":
        parts.append(g("mode_truncation"))
    if g("mlp_groups", 1) != 1:
        parts.append(f"mg{g('mlp_groups')}")
    if g("spectral_envelope", False):
        parts.append("env")
    if g("spectral_dropout", 0.0) > 0:
        parts.append(f"sd{g('spectral_dropout'):.2f}")
    if g("ch_cap", 2) != 2:
        parts.append(f"chc{g('ch_cap')}")
    if g("seed", 0) != 0:
        parts.append(f"s{g('seed')}")
    return "_".join(parts)
