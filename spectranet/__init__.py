"""SpectraNet — a parameter-efficient autoregressive spectral U-Net for PDE surrogates.

Top-level imports for the most common entry points::

    from spectranet import SUNet2d, SpectralConfig
    from spectranet.data import load_dataset, MatReader
    from spectranet.losses import LpLoss
    from spectranet.utils import count_params, seed_all
"""

from .config import SpectralConfig
from .model import SUNet2d, SpectralConv2d
from .losses import LpLoss, HsLoss
from .utils import count_params, seed_all

__all__ = [
    "SUNet2d",
    "SpectralConv2d",
    "SpectralConfig",
    "LpLoss",
    "HsLoss",
    "count_params",
    "seed_all",
]
__version__ = "0.1.0"
