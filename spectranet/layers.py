"""KAN sub-layers and attention modules used by the SpectraNet ablation set.

The canonical SpectraNet (paper headline) does NOT use any of these — the
ablation table reports them as a tested-and-rejected output-head decoration.
They are kept here so the ablation reproduces verbatim, and so the decorated
checkpoint at ``checkpoints/spectranet_ns_v1e5_decorated_best.pt`` loads.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── KAN primitives ─────────────────────────────────────────────────────────

class KANLinear(nn.Module):
    """Spline-basis Kolmogorov–Arnold layer.

    A residual ``base_weight`` (linear with SiLU activation) is summed with a
    spline expansion of the input over a fixed grid.  See Liu et al., 2024.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                self.scale_spline
                * self._curve2coeff(self.grid[self.spline_order : -self.spline_order], noise)
            )

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)] + 1e-8) * bases[:, :, :-1]
                + (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:-k] + 1e-8) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        A = self.b_splines(x.unsqueeze(1).expand(-1, self.in_features)).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        base_out = F.linear(self.base_activation(x_flat), self.base_weight * self.scale_base)
        spline_basis = self.b_splines(x_flat)
        spline_out = torch.einsum(
            "nig,oig->no", spline_basis, self.spline_weight * self.scale_spline
        )
        return (base_out + spline_out).reshape(*shape[:-1], self.out_features)


class FourierKANLinear(nn.Module):
    """Fourier-basis KAN variant — replaces the spline with cos/sin of a
    learnable mixing of input.  Cheaper at inference."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.fourier_weight = nn.Parameter(
            torch.empty(out_features, in_features, 2 * grid_size + 1)
        )
        nn.init.normal_(self.fourier_weight, std=0.01)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)
        xk = x_flat.unsqueeze(-1) * k.view(1, 1, -1) * math.pi
        basis = torch.cat(
            [
                torch.ones(*x_flat.shape, 1, device=x.device, dtype=x.dtype),
                torch.cos(xk),
                torch.sin(xk),
            ],
            dim=-1,
        )
        out = torch.einsum("nif,oif->no", basis, self.fourier_weight)
        return out.reshape(*shape[:-1], self.out_features)


class KANConv2d1x1(nn.Module):
    """Apply a KAN layer per spatial location of a (B, C, H, W) feature map."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kan_type: str = "efficient",
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        if kan_type == "efficient":
            self.kan = KANLinear(in_channels, out_channels, grid_size, spline_order)
        else:
            self.kan = FourierKANLinear(in_channels, out_channels, grid_size)

    def forward(self, x):
        B, C, X, Y = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.kan(x)
        return x.reshape(B, X, Y, -1).permute(0, 3, 1, 2)


class KANMLP2d(nn.Module):
    """Two-layer KAN MLP for channel mixing (parallels :class:`MLP2d`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kan_type: str = "efficient",
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.two_layer = out_channels != in_channels
        self.kan1 = KANConv2d1x1(
            in_channels,
            mid_channels if self.two_layer else out_channels,
            kan_type,
            grid_size,
            spline_order,
        )
        if self.two_layer:
            self.kan2 = KANConv2d1x1(mid_channels, out_channels, kan_type, grid_size, spline_order)

    def forward(self, x):
        x = self.kan1(x)
        if self.two_layer:
            x = self.kan2(x)
        return x


class KANLinearND(nn.Module):
    """KAN layer applied along the last dimension of an arbitrary-shape tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kan_type: str = "efficient",
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        if kan_type == "efficient":
            self.kan = KANLinear(in_features, out_features, grid_size, spline_order)
        else:
            self.kan = FourierKANLinear(in_features, out_features, grid_size)

    def forward(self, x):
        return self.kan(x)


class KANSpectralConv2d(nn.Module):
    """KAN-based 2D spectral convolution.

    Replaces the per-mode complex linear in :class:`SpectralConv2d` with a KAN
    layer over the (real, imag) channel features.  Used only in the KAN
    spectral-block ablation; the canonical SpectraNet uses
    :class:`spectranet.model.SpectralConv2d` instead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        kan_type: str = "efficient",
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        feat_in = in_channels * 2
        feat_out = out_channels * 2

        def _mk():
            if kan_type == "efficient":
                return KANLinear(feat_in, feat_out, grid_size, spline_order)
            return FourierKANLinear(feat_in, feat_out, grid_size)

        self.kan1 = _mk()
        self.kan2 = _mk()

    def _apply_kan(self, kan, modes_slice):
        B = modes_slice.shape[0]
        m1, m2 = modes_slice.shape[2], modes_slice.shape[3]
        n_tok = m1 * m2
        ri = torch.view_as_real(modes_slice.contiguous())
        ri = ri.permute(0, 2, 3, 1, 4).reshape(B * n_tok, self.in_channels * 2)
        out_ri = kan(ri)
        out_ri = out_ri.reshape(B, m1, m2, self.out_channels, 2)
        out_ri = out_ri.permute(0, 3, 1, 2, 4).contiguous()
        return torch.view_as_complex(out_ri)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = self._apply_kan(self.kan1, x_ft[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._apply_kan(self.kan2, x_ft[:, :, -m1:, :m2])
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ─── KAN-aware factories ────────────────────────────────────────────────────

def _kan_kwargs(kan_cfg):
    """Translate a :class:`spectranet.config.KANConfig` (or dict) into kwargs."""
    if kan_cfg is None:
        return dict(kan_type="efficient", grid_size=5, spline_order=3)
    if isinstance(kan_cfg, dict):
        return dict(kan_cfg)
    return dict(kan_type=kan_cfg.kan_type, grid_size=kan_cfg.grid_size, spline_order=kan_cfg.spline_order)


def make_mlp2d(in_ch, out_ch, mid_ch, use_kan: bool, kan_cfg, mlp_groups: int = 1):
    """Return either :class:`KANMLP2d` (if ``use_kan``) or :class:`MLP2d`."""
    if use_kan:
        return KANMLP2d(in_ch, out_ch, mid_ch, **_kan_kwargs(kan_cfg))
    from .model import MLP2d
    return MLP2d(in_ch, out_ch, mid_ch, mlp_groups=mlp_groups)


def make_conv1x1(in_ch, out_ch, use_kan: bool, kan_cfg):
    if use_kan:
        return KANConv2d1x1(in_ch, out_ch, **_kan_kwargs(kan_cfg))
    return nn.Conv2d(in_ch, out_ch, 1)


def make_linear(in_f, out_f, use_kan: bool, kan_cfg):
    if use_kan:
        return KANLinearND(in_f, out_f, **_kan_kwargs(kan_cfg))
    return nn.Linear(in_f, out_f)


# ─── Attention modules (ablation) ───────────────────────────────────────────

class SpectralAttention2d(nn.Module):
    """Multi-head attention over the low-frequency spectral modes shared
    between the decoder feature map and its skip connection."""

    def __init__(self, channels: int, modes1: int, modes2: int, n_heads: int = 4):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.n_heads = n_heads
        feat_dim = channels * 2
        assert feat_dim % n_heads == 0
        self.d_k = feat_dim // n_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, decoder, skip):
        B, C, X, Y = decoder.shape
        m1, m2 = self.modes1, self.modes2
        H, d = self.n_heads, self.d_k
        dec_ft = torch.fft.rfft2(decoder)
        skip_ft = torch.fft.rfft2(skip)
        dec_modes = dec_ft[:, :, :m1, :m2]
        skip_modes = skip_ft[:, :, :m1, :m2]
        n_tok = m1 * m2
        dec_ri = torch.view_as_real(dec_modes.contiguous())
        skip_ri = torch.view_as_real(skip_modes.contiguous())
        dec_tok = dec_ri.permute(0, 2, 3, 1, 4).reshape(B, n_tok, C * 2)
        skip_tok = skip_ri.permute(0, 2, 3, 1, 4).reshape(B, n_tok, C * 2)
        Q = self.q_proj(dec_tok).view(B, n_tok, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(skip_tok).view(B, n_tok, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(skip_tok).view(B, n_tok, H, d).permute(0, 2, 1, 3)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, n_tok, C * 2)
        out = self.out_proj(out)
        out = out.view(B, m1, m2, C, 2).permute(0, 3, 1, 2, 4).contiguous()
        out_modes = torch.view_as_complex(out)
        out_ft = torch.zeros_like(skip_ft)
        out_ft[:, :, :m1, :m2] = out_modes
        return torch.fft.irfft2(out_ft, s=(X, Y))


class AttentionGate2d(nn.Module):
    """Oktay-style additive attention gate for skip connections."""

    def __init__(self, channels: int, use_kan: bool = False, kan_cfg=None):
        super().__init__()
        mid = max(channels // 2, 1)
        if use_kan and kan_cfg is not None:
            self.Wg = KANConv2d1x1(channels, mid, **_kan_kwargs(kan_cfg))
            self.Wx = KANConv2d1x1(channels, mid, **_kan_kwargs(kan_cfg))
            self.psi = KANConv2d1x1(mid, 1, **_kan_kwargs(kan_cfg))
        else:
            self.Wg = nn.Conv2d(channels, mid, 1)
            self.Wx = nn.Conv2d(channels, mid, 1)
            self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, g, x_skip):
        q = F.relu(self.Wg(g) + self.Wx(x_skip))
        alpha = torch.sigmoid(self.psi(q))
        return g + alpha * x_skip


class BottleneckSelfAttn2d(nn.Module):
    """Standard multi-head self-attention over the bottleneck spatial tokens."""

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        B, C, X, Y = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, X * Y, C)
        normed = self.norm(tokens)
        out, _ = self.attn(normed, normed, normed)
        tokens = tokens + out
        return tokens.reshape(B, X, Y, C).permute(0, 3, 1, 2)


class ChannelAttention2d(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        s = x.mean(dim=[2, 3])
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.view(B, C, 1, 1)


class SpatialAttention2d(nn.Module):
    """CBAM-style separable spatial attention."""

    def __init__(self, k: int = 7):
        super().__init__()
        p = k // 2
        self.conv_x = nn.Conv2d(2, 1, kernel_size=(k, 1), padding=(p, 0))
        self.conv_y = nn.Conv2d(2, 1, kernel_size=(1, k), padding=(0, p))

    def _pool(self, x):
        return torch.cat(
            [x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1
        )

    def forward(self, x):
        p = self._pool(x)
        mask = torch.sigmoid(self.conv_x(p) + self.conv_y(p))
        return x * mask


def resolve_attn(mode: str, channels: int) -> str:
    """Resolve the ``'progressive'`` placeholder against the channel count."""
    if mode != "progressive":
        return mode
    return "both" if channels >= 52 else "spatial"
