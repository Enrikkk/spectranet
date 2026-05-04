"""Data loading: ``MatReader`` (.mat v5 + v7.3 / HDF5), Gaussian-RF IC sampler,
normalizers, and the per-dataset :func:`load_dataset` dispatcher.

The dispatcher knows the canonical split sizes for each PDE benchmark used in
the paper:

============= ========================== ====================== ====
Dataset name  File expected under data/  ntrain / nval / ntest  Note
============= ========================== ====================== ====
ns_v1e5       NavierStokes_V1e-5_N1200_T20.mat  850 / 150 / 200  Public (Li et al., 2020)
ns_v1e4       ns_V1e-4_N10000_T30.mat           850 / 150 / 200  Public (extended FNO suite)
ns_v1e3       ns_V1e-3_N5000_T50.mat            850 / 150 / 200  Public (extended FNO suite)
sw            ShallowWater.mat                  700 / 150 / 150  Public (PDEBench)
dr            DiffusionReaction.mat             700 / 150 / 150  Public (PDEBench)
am            ActiveMatter.mat                  135 / 48 / 42    Public (The Well)
ns_v1e5_128   ns_v1e5_128_N1200_T20.mat         850 / 150 / 200  In-house (shipped in data/)
============= ========================== ====================== ====

All datasets share the autoregressive contract: input is a window of ``T_in``
past frames, target is the next ``T_out`` frames.
"""

from __future__ import annotations
import math
import operator
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import scipy.io as scio
except ImportError as e:                   # pragma: no cover
    raise RuntimeError("SpectraNet requires scipy>=1.10 for .mat I/O") from e

try:
    import h5py
except ImportError as e:                   # pragma: no cover
    raise RuntimeError("SpectraNet requires h5py>=3.8 for .mat (v7.3) I/O") from e


# ─── MAT reader ─────────────────────────────────────────────────────────────

class MatReader(object):
    """Read variables from MATLAB .mat files (v5 and v7.3/HDF5).

    The reader transparently handles both formats and returns either a NumPy
    array or a torch.Tensor depending on ``to_torch``.
    """

    def __init__(self, file_path, to_torch: bool = True, to_cuda: bool = False, to_float: bool = True):
        super().__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = str(file_path)
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scio.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = str(file_path)
        self._load_file()

    def read_field(self, field: str):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x


# ─── Normalizers (kept for completeness; canonical SpectraNet does not normalize) ─

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps: float = 1e-5, time_last: bool = True):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std, mean = self.std + self.eps, self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            else:
                std = self.std[..., sample_idx] + self.eps
                mean = self.mean[..., sample_idx]
        return x * std + mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self


class GaussianNormalizer(object):
    def __init__(self, x, eps: float = 1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        return x * (self.std + self.eps) + self.mean


class RangeNormalizer(object):
    def __init__(self, x, low: float = 0.0, high: float = 1.0):
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)
        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        return x.view(s)

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        return x.view(s)


# ─── Gaussian random fields (used by 128² generator) ────────────────────────

class GaussianRF(object):
    """Spectral Gaussian-random-field IC sampler (Matérn-like spectrum).

    The default ``alpha=2, tau=3`` matches Li et al. 2020 for NS ν=10⁻⁵
    initial conditions.  ``sample(N)`` returns a real-valued ``(N, *size)``
    tensor.
    """

    def __init__(self, dim: int, size: int, alpha: float = 2, tau: float = 3,
                 sigma: Optional[float] = None, boundary: str = "periodic", device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2

        if dim == 1:
            k = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            self.sqrt_eig = (
                size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0] = 0.0
        elif dim == 2:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (
                (size ** 2) * math.sqrt(2.0) * sigma
                * ((4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (
                (size ** 3) * math.sqrt(2.0) * sigma
                * ((4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0, 0, 0] = 0.0
        else:
            raise ValueError(f"Unsupported dim={dim}")

        self.size = tuple([size] * self.dim)

    def sample(self, N: int):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


# ─── Per-dataset metadata ───────────────────────────────────────────────────

DATASETS = {
    # name           filename                                    splits           field S    notes
    "ns_v1e5":       ("NavierStokes_V1e-5_N1200_T20.mat",        (850, 150, 200), "u",  64,
                      "NS ν=10⁻⁵, public (Li et al., 2020)."),
    "ns_v1e4":       ("ns_V1e-4_N10000_T30.mat",                 (850, 150, 200), "u",  64,
                      "NS ν=10⁻⁴, public (extended FNO suite)."),
    "ns_v1e3":       ("ns_V1e-3_N5000_T50.mat",                  (850, 150, 200), "u",  64,
                      "NS ν=10⁻³, public (extended FNO suite)."),
    "sw":            ("ShallowWater.mat",                        (700, 150, 150), "u",  64,
                      "Shallow Water, public (PDEBench)."),
    "dr":            ("DiffusionReaction.mat",                   (700, 150, 150), "u",  64,
                      "Diffusion-Reaction, public (PDEBench)."),
    "am":            ("ActiveMatter.mat",                        (135,  48,  42), "u",  64,
                      "Active Matter, public (The Well)."),
    "ns_v1e5_128":   ("ns_v1e5_128_N1200_T20.mat",               (850, 150, 200), "u", 128,
                      "NS ν=10⁻⁵ at 128², in-house (shipped in data/)."),
}


def list_datasets():
    """Return the table of supported dataset names."""
    return {k: dict(filename=v[0], splits=v[2], field=v[3], S=v[4], note=v[5])
            for k, v in DATASETS.items()}


def load_dataset(
    name: str,
    root: str | Path,
    *,
    T_in: int = 10,
    T_out: int = 10,
) -> Tuple[Tuple[torch.Tensor, ...], dict]:
    """Load a benchmark dataset.

    Parameters
    ----------
    name : str
        One of the keys in :data:`DATASETS` (``ns_v1e5``, ``ns_v1e4``, etc.).
    root : str or Path
        Directory containing the dataset's ``.mat`` file.
    T_in, T_out : int
        Window lengths for the autoregressive split.  Together they must be
        ``≤`` the dataset's available timestep count.

    Returns
    -------
    arrays : tuple
        ``((X_train, Y_train), (X_val, Y_val), (X_test, Y_test))`` —
        each ``X_*`` of shape ``(B, S, S, T_in)`` and each ``Y_*`` of shape
        ``(B, S, S, T_out)``.
    info : dict
        Dataset metadata (``filename``, ``splits``, ``S``, ``field``, ``note``).
    """
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {sorted(DATASETS.keys())}"
        )
    filename, splits, field, S, note = DATASETS[name]
    path = Path(root) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. See docs/DATA.md for download instructions."
        )
    ntrain, nval, ntest = splits
    info = dict(filename=filename, splits=splits, field=field, S=S, note=note)

    print(f"[data] loading {path} (dataset={name}, splits={splits}, S={S}) ...")
    reader = MatReader(str(path))
    u = reader.read_field(field)
    print(f"[data]   u shape: {tuple(u.shape)}")
    assert u.ndim == 4 and u.shape[-1] >= T_in + T_out, (
        f"Dataset '{name}' shape {tuple(u.shape)} cannot serve T_in={T_in}, T_out={T_out}"
    )

    X_train = u[:ntrain, :, :, :T_in]
    Y_train = u[:ntrain, :, :, T_in : T_in + T_out]
    X_val = u[ntrain : ntrain + nval, :, :, :T_in]
    Y_val = u[ntrain : ntrain + nval, :, :, T_in : T_in + T_out]
    X_test = u[ntrain + nval : ntrain + nval + ntest, :, :, :T_in]
    Y_test = u[ntrain + nval : ntrain + nval + ntest, :, :, T_in : T_in + T_out]

    return ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test)), info
