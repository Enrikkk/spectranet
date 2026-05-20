"""SpectraNet — autoregressive spectral U-Net for PDE surrogates.

Module map (line numbers refer to this file):

  * :class:`SpectralConv2d`    — FNO-style 2D Fourier convolution with
                                  optional disk truncation, learnable Gaussian
                                  envelope, and outer-ring spectral dropout
                                  (the *Residual-Target Spectral Block* core).
  * :class:`MLP2d`             — two-layer 1×1 channel-mixing MLP with GeLU.
  * :class:`MultiScaleOutput2d`— softmax-weighted sum of one MLP head per
                                  decoder level (decorated variant only).
  * :class:`EncoderBlock2d`    — per-level spectral conv + MLP + 1×1 residual
                                  with optional skip-side attention.
  * :class:`DecoderBlock2d`    — bilinear up-sample + 1×1 projection + skip
                                  merge + spectral conv + MLP, with eight
                                  selectable skip-merge variants.
  * :class:`SUNet2d`           — the SpectraNet model.  Parameters are
                                  exposed as named keyword arguments; the
                                  canonical operating point is
                                  ``width=32, modes=12, levels=3, ch_cap=2,
                                  output_mode='single', kan_flags=all-False,
                                  T_in=10, T_out=1, S=64`` for 2,040,705
                                  parameters and L²=0.0822 on NS ν=10⁻⁵.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectralConfig, default_kan_flags
from .layers import (
    BottleneckSelfAttn2d,
    ChannelAttention2d,
    KANSpectralConv2d,
    KANConv2d1x1,
    SpatialAttention2d,
    SpectralAttention2d,
    AttentionGate2d,
    _kan_kwargs,
    make_conv1x1,
    make_linear,
    make_mlp2d,
    resolve_attn,
)


class SpectralConv2d(nn.Module):
    """2D Fourier-truncated convolution.

    Multiplies low-frequency Fourier modes of the input by learnable complex
    weights; high-frequency modes pass through unchanged (truncation).

    Parameters
    ----------
    in_channels, out_channels : int
    modes1, modes2 : int
        Number of low-frequency modes retained along each spatial axis.
    cfg : SpectralConfig | None
        Optional configuration.  ``cfg.mode_truncation='disk'`` switches the
        rectangular cut to an isotropic ``kx²+ky² ≤ M²`` disk.
        ``cfg.spectral_envelope`` adds a per-output-channel learnable Gaussian
        envelope.  ``cfg.spectral_dropout`` enables outer-ring Bernoulli
        dropout during training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        cfg: SpectralConfig | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

        cfg = cfg or SpectralConfig()
        self.mode_truncation = cfg.mode_truncation
        self.spectral_envelope = cfg.spectral_envelope
        self.spectral_dropout = cfg.spectral_dropout

        # Disk-shaped truncation: SO(2)-isotropic kx²+ky² ≤ M².  rfft2 stores
        # positive ky in [:m2]; block 1 covers kx ∈ [0, m1), block 2 covers
        # kx ∈ [-m1, 0).
        if self.mode_truncation == "disk":
            kx = torch.arange(modes1).float()
            ky = torch.arange(modes2).float()
            KX, KY = torch.meshgrid(kx, ky, indexing="ij")
            mask = (KX ** 2 + KY ** 2 <= modes1 ** 2).to(torch.cfloat)
            self.register_buffer("disk_mask", mask)
        else:
            self.disk_mask = None

        # Per-output-channel learnable Gaussian envelope (initialised at the
        # cutoff so it is a no-op at start of training).
        if self.spectral_envelope:
            self.kc = nn.Parameter(torch.full((out_channels,), float(modes1)))
        else:
            self.kc = None

    def _apply_envelope(self, w):
        if self.kc is None:
            return w
        kx = torch.arange(self.modes1, device=w.device).float()
        ky = torch.arange(self.modes2, device=w.device).float()
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        kmag = torch.sqrt(KX ** 2 + KY ** 2)
        kc = self.kc.clamp(min=1e-3).view(1, -1, 1, 1)
        env = torch.exp(-(kmag.view(1, 1, self.modes1, self.modes2) / kc) ** 2)
        return w * env.to(w.dtype)

    @staticmethod
    def compl_mul2d(inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def _effective_weight(self, w):
        if self.disk_mask is not None:
            w = w * self.disk_mask
        if self.kc is not None:
            w = self._apply_envelope(w)
        if self.training and self.spectral_dropout > 0.0:
            kx = torch.arange(self.modes1, device=w.device).float()
            ky = torch.arange(self.modes2, device=w.device).float()
            KX, KY = torch.meshgrid(kx, ky, indexing="ij")
            outer = KX ** 2 + KY ** 2 > (self.modes1 / 2.0) ** 2
            keep = torch.bernoulli(
                torch.full(
                    (self.modes1, self.modes2),
                    1.0 - self.spectral_dropout,
                    device=w.device,
                )
            )
            mask = torch.where(outer, keep, torch.ones_like(keep))
            w = w * mask.to(w.dtype)
        return w

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
        w1 = self._effective_weight(self.weights1)
        w2 = self._effective_weight(self.weights2)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], w1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class MLP2d(nn.Module):
    """Two-layer 1×1 channel-mixing MLP with GeLU (the SpectraNet
    output head when ``output_mode='single'``).

    With ``mlp_groups > 1`` the convolutions become grouped (the layer
    silently falls back to ``groups=1`` when channels aren't divisible).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        mlp_groups: int = 1,
    ):
        super().__init__()
        g = mlp_groups
        g1 = g if (in_channels % g == 0 and mid_channels % g == 0) else 1
        g2 = g if (mid_channels % g == 0 and out_channels % g == 0) else 1
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1, groups=g1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1, groups=g2)

    def forward(self, x):
        return self.mlp2(F.gelu(self.mlp1(x)))


def make_spectral_conv2d(
    in_ch: int,
    out_ch: int,
    m1: int,
    m2: int,
    use_kan: bool,
    kan_cfg,
    spectral_cfg: SpectralConfig | None = None,
):
    """Factory: KAN-spectral conv if ``use_kan``, else plain SpectralConv2d."""
    if use_kan:
        return KANSpectralConv2d(in_ch, out_ch, m1, m2, **_kan_kwargs(kan_cfg))
    return SpectralConv2d(in_ch, out_ch, m1, m2, cfg=spectral_cfg)


class MultiScaleOutput2d(nn.Module):
    """Softmax-weighted sum of one MLP head per decoder level.

    Used in the *decorated* SpectraNet ablation; the canonical model uses a
    single 1×1 linear projection (``output_mode='single'``) instead.
    """

    def __init__(self, ch_list, T_out, use_kan: bool, kan_cfg, mlp_groups: int = 1):
        super().__init__()
        self.T_out = T_out
        self.heads = nn.ModuleList(
            [make_mlp2d(ch, T_out, ch * 4, use_kan, kan_cfg, mlp_groups=mlp_groups) for ch in ch_list]
        )
        init = torch.zeros(len(ch_list))
        init[-1] = 1.0
        self.log_weights = nn.Parameter(init)

    def forward(self, decoder_outs):
        finest = decoder_outs[-1]
        tgt_size = finest.shape[2:]
        w = F.softmax(self.log_weights, dim=0)
        result = None
        for i, (feat, head) in enumerate(zip(decoder_outs, self.heads)):
            h = head(feat)
            if h.shape[2:] != tgt_size:
                h = F.interpolate(h, size=tgt_size, mode="bilinear", align_corners=False)
            result = w[i] * h if result is None else result + w[i] * h
        return result


class EncoderBlock2d(nn.Module):
    """One U-Net encoder level: spectral conv + MLP + 1×1 residual, then
    avg-pool 2× and 1×1 channel-grow projection."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        modes: int,
        kan_flags: dict,
        kan_cfg,
        skip_attn: str = "none",
        dropout: float = 0.0,
        spectral_cfg: SpectralConfig | None = None,
        mlp_groups: int = 1,
    ):
        super().__init__()
        self.conv = make_spectral_conv2d(
            ch_in, ch_in, modes, modes,
            kan_flags["spectral"], kan_cfg, spectral_cfg=spectral_cfg,
        )
        self.mlp = make_mlp2d(ch_in, ch_in, ch_in, kan_flags["fno_mlp"], kan_cfg, mlp_groups=mlp_groups)
        self.w = make_conv1x1(ch_in, ch_in, kan_flags["residual_w"], kan_cfg)
        self.down_proj = make_conv1x1(ch_in, ch_out, kan_flags["projections"], kan_cfg)
        actual = resolve_attn(skip_attn, ch_in)
        if actual in ("channel", "both"):
            self.skip_chan_attn = ChannelAttention2d(ch_in)
        if actual in ("spatial", "both"):
            self.skip_spatial_attn = SpatialAttention2d()
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x1 = self.mlp(self.conv(x))
        x2 = self.w(x)
        skip = F.gelu(x1 + x2)
        if hasattr(self, "skip_chan_attn"):
            skip = self.skip_chan_attn(skip)
        if hasattr(self, "skip_spatial_attn"):
            skip = self.skip_spatial_attn(skip)
        skip = self.drop(skip)
        down = F.avg_pool2d(skip, 2)
        down = F.gelu(self.down_proj(down))
        return down, skip


class DecoderBlock2d(nn.Module):
    """One U-Net decoder level: bilinear up-sample + 1×1 projection + skip
    merge + spectral conv + MLP."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        modes: int,
        skip_merge: str,
        kan_flags: dict,
        kan_cfg,
        attn_modes=None,
        is_last: bool = False,
        post_spatial_attn: bool = False,
        decoder_out_attn: str = "none",
        dropout: float = 0.0,
        spectral_cfg: SpectralConfig | None = None,
        mlp_groups: int = 1,
    ):
        super().__init__()
        self.skip_merge = skip_merge
        self.is_last = is_last
        self.up_proj = make_conv1x1(ch_in, ch_out, kan_flags["projections"], kan_cfg)
        if skip_merge == "concat":
            self.merge = nn.Conv2d(ch_out * 2, ch_out, 1)
        elif skip_merge == "concat_kan":
            self.merge = KANConv2d1x1(ch_out * 2, ch_out, **_kan_kwargs(kan_cfg))
        elif skip_merge in ("add_kan", "add_kan_res"):
            self.merge = KANConv2d1x1(ch_out, ch_out, **_kan_kwargs(kan_cfg))
        elif skip_merge == "spectral_attn":
            assert attn_modes is not None
            self.attn = SpectralAttention2d(ch_out, *attn_modes)
        elif skip_merge == "attn_gate":
            self.gate = AttentionGate2d(ch_out, use_kan=False)
        elif skip_merge == "kan_attn_gate":
            self.gate = AttentionGate2d(ch_out, use_kan=True, kan_cfg=kan_cfg)
        self.conv = make_spectral_conv2d(
            ch_out, ch_out, modes, modes, kan_flags["spectral"], kan_cfg, spectral_cfg=spectral_cfg,
        )
        self.mlp = make_mlp2d(ch_out, ch_out, ch_out, kan_flags["fno_mlp"], kan_cfg, mlp_groups=mlp_groups)
        self.w = make_conv1x1(ch_out, ch_out, kan_flags["residual_w"], kan_cfg)
        actual_doa = (
            decoder_out_attn if decoder_out_attn != "none"
            else ("spatial" if post_spatial_attn else "none")
        )
        actual_doa = resolve_attn(actual_doa, ch_out)
        if actual_doa in ("channel", "both"):
            self.out_chan_attn = ChannelAttention2d(ch_out)
        if actual_doa in ("spatial", "both"):
            self.out_spatial_attn = SpatialAttention2d()
        self.drop = nn.Dropout2d(p=dropout) if (dropout > 0 and not is_last) else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = F.gelu(self.up_proj(x))
        if self.skip_merge == "add":
            x = F.gelu(x + skip)
        elif self.skip_merge in ("concat", "concat_kan"):
            x = F.gelu(self.merge(torch.cat([x, skip], dim=1)))
        elif self.skip_merge == "add_kan":
            x = F.gelu(self.merge(x + skip))
        elif self.skip_merge == "add_kan_res":
            merged = x + skip
            x = F.gelu(merged + self.merge(merged))
        elif self.skip_merge == "spectral_attn":
            x = F.gelu(x + skip + self.attn(x, skip))
        elif self.skip_merge in ("attn_gate", "kan_attn_gate"):
            x = F.gelu(self.gate(x, skip))
        x1 = self.mlp(self.conv(x))
        x2 = self.w(x)
        out = (x1 + x2) if self.is_last else F.gelu(x1 + x2)
        if hasattr(self, "out_chan_attn"):
            out = self.out_chan_attn(out)
        if hasattr(self, "out_spatial_attn"):
            out = self.out_spatial_attn(out)
        out = self.drop(out)
        return out


class SUNet2d(nn.Module):
    """SpectraNet — Spectral U-Net for autoregressive PDE rollout.

    Forward signature: ``(B, X, Y, T_in) → (B, X, Y, T_out)``.

    Inside, the input is concatenated with a (gx, gy) coordinate grid,
    lifted to ``width`` channels, descended through ``levels`` encoder blocks
    (each spectral-conv + MLP + 1×1-residual + avg-pool), processed by a
    bottleneck spectral conv + MLP, then ascended through ``levels``
    decoder blocks (bilinear up-sample + 1×1 projection + skip merge +
    spectral-conv + MLP).  The final feature map is reduced to ``T_out``
    output channels by a 1×1 head (``output_mode='single'``) or by the
    multi-scale head (``output_mode='multiscale_mlp'``).

    Parameters
    ----------
    base_modes : int
        Spectral truncation radius at level 0 (halved at each deeper level).
    width : int
        Channel count at level 0; doubles per level up to ``ch_cap`` levels.
    in_channels : int
        Input window length (``T_in``).  Time history enters as channels.
    T_out : int, default 1
        Output frame count.  The autoregressive trainer uses ``T_out=1``.
    levels : int, default 3
        U-Net depth (encoder + decoder symmetric levels).
    ch_cap : int, default 2
        Cap on channel-doubling exponent.  ``2`` is canonical (channels =
        ``width * 2**min(level, 2)``); set to ``3`` to widen the bottleneck.
    skip_merge : str, default 'add'
        How decoder + skip features combine; one of 'add', 'concat',
        'concat_kan', 'add_kan', 'add_kan_res', 'spectral_attn',
        'attn_gate', 'kan_attn_gate'.
    bottleneck_attn : str, default 'none'
        'none', 'self_attn', 'channel', or 'spatial_channel'.
    output_mode : str, default 'single'
        'single' → 1×1 MLP head; 'multiscale_mlp' → softmax-weighted sum of
        one MLP per decoder level (decorated variant); 'multiscale_kan' →
        same with KAN sub-heads.
    encoder_skip_attn, decoder_out_attn : str, default 'none'
        Optional per-block attention; see :class:`EncoderBlock2d` /
        :class:`DecoderBlock2d`.
    S : int, default 64
        Native input grid resolution (only used to size the modes schedule;
        the model itself is resolution-agnostic at inference).
    kan_flags : dict[str, bool] | None
        Per-sublayer KAN switches (canonical: all False).
    kan_cfg : KANConfig | dict | None
    dropout : float, default 0.0
    spectral_cfg : SpectralConfig | None
        Disk truncation, learnable envelope, spectral dropout.
    mlp_groups : int, default 1
        Group count for MLP2d's 1×1 convolutions (E4 ablation).

    Notes
    -----
    *Canonical headline configuration*: ``width=32, base_modes=12, levels=3,
    ch_cap=2, in_channels=10, T_out=1, output_mode='single', kan_flags=all-False,
    skip_merge='add', bottleneck_attn='none', S=64`` → 2,040,705 parameters.
    """

    def __init__(
        self,
        base_modes: int,
        width: int,
        in_channels: int,
        T_out: int = 1,
        levels: int = 3,
        skip_merge: str = "add",
        bottleneck_attn: str = "none",
        output_mode: str = "single",
        encoder_skip_attn: str = "none",
        decoder_out_attn: str = "none",
        S: int = 64,
        kan_flags: dict | None = None,
        kan_cfg=None,
        dropout: float = 0.0,
        ch_cap: int = 2,
        spectral_cfg: SpectralConfig | None = None,
        mlp_groups: int = 1,
    ):
        super().__init__()
        if kan_flags is None:
            kan_flags = default_kan_flags()
        if kan_cfg is None:
            kan_cfg = dict(kan_type="efficient", grid_size=5, spline_order=3)
        if spectral_cfg is None:
            spectral_cfg = SpectralConfig(mlp_groups=mlp_groups)
        elif spectral_cfg.mlp_groups == 1 and mlp_groups != 1:
            # Allow caller to pass mlp_groups separately.
            spectral_cfg = SpectralConfig(
                mode_truncation=spectral_cfg.mode_truncation,
                mlp_groups=mlp_groups,
                spectral_envelope=spectral_cfg.spectral_envelope,
                spectral_dropout=spectral_cfg.spectral_dropout,
            )
        self.levels = levels
        self.width = width
        self.output_mode = output_mode
        self.T_out = T_out

        self.p = make_linear(in_channels + 2, width, kan_flags["lifting"], kan_cfg)

        grids, modes_list, ch_list = [], [], []
        cx = cy = S
        cmodes = base_modes
        for lvl in range(levels + 1):
            ch = width * (2 ** min(lvl, ch_cap))
            grids.append((cx, cy))
            safe_m = min(cmodes, cx // 2)
            modes_list.append(safe_m)
            ch_list.append(ch)
            if lvl < levels:
                cx, cy = cx // 2, cy // 2
                cmodes = max(1, cmodes // 2)
        self.grids = grids

        self.encoders = nn.ModuleList()
        for lvl in range(levels):
            self.encoders.append(
                EncoderBlock2d(
                    ch_list[lvl], ch_list[lvl + 1], modes_list[lvl],
                    kan_flags, kan_cfg,
                    skip_attn=encoder_skip_attn, dropout=dropout,
                    spectral_cfg=spectral_cfg, mlp_groups=spectral_cfg.mlp_groups,
                )
            )

        bot = levels
        self.bot_conv = make_spectral_conv2d(
            ch_list[bot], ch_list[bot], modes_list[bot], modes_list[bot],
            kan_flags["spectral"], kan_cfg, spectral_cfg=spectral_cfg,
        )
        self.bot_mlp = make_mlp2d(
            ch_list[bot], ch_list[bot], ch_list[bot],
            kan_flags["fno_mlp"], kan_cfg, mlp_groups=spectral_cfg.mlp_groups,
        )
        self.bot_w = make_conv1x1(ch_list[bot], ch_list[bot], kan_flags["residual_w"], kan_cfg)
        if bottleneck_attn == "self_attn":
            self.bot_self_attn = BottleneckSelfAttn2d(ch_list[bot])
        elif bottleneck_attn in ("channel", "spatial_channel"):
            self.bot_chan_attn = ChannelAttention2d(ch_list[bot])
        post_spatial_attn = bottleneck_attn == "spatial_channel"
        self.bot_drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.decoders = nn.ModuleList()
        for i, lvl in enumerate(range(levels - 1, -1, -1)):
            is_last = (i == levels - 1) and (output_mode == "single")
            attn_modes = (
                (modes_list[lvl], modes_list[lvl]) if skip_merge == "spectral_attn" else None
            )
            self.decoders.append(
                DecoderBlock2d(
                    ch_list[lvl + 1], ch_list[lvl], modes_list[lvl],
                    skip_merge, kan_flags, kan_cfg,
                    attn_modes=attn_modes, is_last=is_last,
                    post_spatial_attn=post_spatial_attn,
                    decoder_out_attn=decoder_out_attn,
                    dropout=dropout,
                    spectral_cfg=spectral_cfg,
                    mlp_groups=spectral_cfg.mlp_groups,
                )
            )

        if output_mode == "single":
            self.q = make_mlp2d(
                width, T_out, width * 4, kan_flags["output"], kan_cfg,
                mlp_groups=spectral_cfg.mlp_groups,
            )
        else:
            use_kan_out = output_mode == "multiscale_kan"
            dec_ch_list = [ch_list[levels - 1 - i] for i in range(levels)]
            self.q = MultiScaleOutput2d(
                dec_ch_list, T_out, use_kan_out, kan_cfg,
                mlp_groups=spectral_cfg.mlp_groups,
            )

    def get_grid(self, B, X, Y, device):
        gx = torch.linspace(0, 1, X, device=device).reshape(1, X, 1, 1).repeat(B, 1, Y, 1)
        gy = torch.linspace(0, 1, Y, device=device).reshape(1, 1, Y, 1).repeat(B, X, 1, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x):
        # x: (B, X, Y, T_in)
        B, X, Y, _ = x.shape
        grid = self.get_grid(B, X, Y, x.device)
        x = torch.cat((x, grid), dim=-1)        # (B, X, Y, T_in+2)
        x = self.p(x)                           # (B, X, Y, W)
        x = x.permute(0, 3, 1, 2)               # (B, W, X, Y)
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x1 = self.bot_mlp(self.bot_conv(x))
        x2 = self.bot_w(x)
        x = F.gelu(x1 + x2)
        if hasattr(self, "bot_self_attn"):
            x = self.bot_self_attn(x)
        if hasattr(self, "bot_chan_attn"):
            x = self.bot_chan_attn(x)
        x = self.bot_drop(x)
        if self.output_mode == "single":
            for dec, skip in zip(self.decoders, reversed(skips)):
                x = dec(x, skip)
            x = self.q(x)                        # (B, T_out, X, Y)
            return x.permute(0, 2, 3, 1)         # (B, X, Y, T_out)
        else:
            decoder_outs = []
            for dec, skip in zip(self.decoders, reversed(skips)):
                x = dec(x, skip)
                decoder_outs.append(x)
            x = self.q(decoder_outs)             # (B, T_out, X, Y)
            return x.permute(0, 2, 3, 1)
