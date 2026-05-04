# SpectraNet architecture walkthrough

This document mirrors paper §4 and points each named module to its
implementation.  All file:line references are absolute paths inside the
release.

## At a glance

SpectraNet is a U-Net-shaped autoregressive operator with three named building blocks:

1. **Residual-Target Spectral Block** — the encoder/bottleneck/decoder unit, combining a Fourier-truncated spectral convolution with a 1×1 channel-mixing MLP and a residual skip.  Defined in `spectranet/model.py:73` (`SpectralConv2d`), `spectranet/model.py:160` (`MLP2d`), `spectranet/model.py:241` (`EncoderBlock2d`), `spectranet/model.py:282` (`DecoderBlock2d`).
2. **Semigroup-Consistency Loss** — a training-time penalty that enforces `f(f(u_t)) ≈ u_{t+2}` in addition to the per-step L² loss; weighted by `--two_step_lambda` (canonical 0.1).  Implemented in the trainer at `scripts/train_spectranet.py:240`.
3. **Output projection** — a two-layer 1×1 MLP (canonical: hidden 4×width=128, GeLU between).  Defined in `spectranet/model.py:160` (`MLP2d`); selected as the head at `spectranet/model.py:434` when `output_mode='single'`.

The decorated multi-resolution + KAN-on-output head is preserved as `output_mode='multiscale_mlp'` for ablation reproducibility, but is **not** part of canonical SpectraNet — see paper §8.

## Forward pass (canonical configuration)

Input: `(B, 64, 64, 10)` window of past vorticity frames.

1. Concatenate a `(gx, gy) ∈ [0, 1]²` coordinate grid → `(B, 64, 64, 12)`.  See `spectranet/model.py:466`.
2. Pointwise lift `nn.Linear(12, 32)` → `(B, 32, 64, 64)` (channels-first thereafter).
3. Encoder, three levels:
   - Level 0: SpectralConv2d (modes 12, 32→32) + MLP2d (32→32) + 1×1 residual; downsample 2× and channel-grow 32→64.
   - Level 1: spectral modes drop to 6, channels 64→128.
   - Level 2: spectral modes drop to 3, channels 128 → bottleneck input 128.
4. Bottleneck at 8×8: SpectralConv2d (modes 1, 128→128) + MLP2d + 1×1 residual.
5. Decoder, three levels (mirror of encoder):
   - Bilinear up-sample 2× + 1×1 channel-shrink + skip-add (`skip_merge='add'`) + SpectralConv2d + MLP2d + 1×1 residual.
6. Output projection `MLP2d(32, 1, 128)` → `(B, 1, 64, 64)`.
7. Permute to `(B, 64, 64, 1)`.

Total parameter count: **2,040,705** (PyTorch convention).

## Autoregressive rollout

Predicted frame is consumed by the next call's input window.  In residual-target mode (canonical), the network predicts `Δu = u_{t+1} − u_t`; the absolute frame is recovered by adding `Δu` to the last input frame.  See `scripts/_eval_common.py:160` for the inference-time `free_rollout_ar2d` helper.

## Key design choices

| Knob | Value | Justification |
|---|---|---|
| width | 32 | Pareto-optimal in our width sweep (`results/width_scaling.csv`); larger widths regress past 48. |
| modes | 12 | 12 retained / 32 Nyquist on the 64² grid (~38% spectral coverage). |
| levels | 3 | Three encoder/decoder levels; deeper hierarchies do not help at 64². |
| ch_cap | 2 | Channel-doubling cap; ablation in Appendix C.3 confirms 3 is a wash. |
| skip_merge | `add` | Additive U-Net skips; `concat` performed identically at higher param cost. |
| residual_target | True | Predicting Δu lets the spectral block focus on the dynamic component. |
| two_step_lambda | 0.1 | Semigroup-Consistency Loss weight (Theorem 1 stability spine). |
| output_mode | `single` | Two-layer 1×1 MLP head; the decorated multi-res+KAN head is a no-op (Δ=0.0001 at noise floor). |

## Resolution invariance

`SpectralConv2d.forward` calls `torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))`; the spatial output size adapts to the input.  The `(gx, gy)` lift is also resolution-agnostic.  This means the same trained checkpoint can run zero-shot at 128² (paper §7.4) — the only resolution-dependent step is the per-pixel two-layer MLP head, which trades a small transfer-ratio gap for the empirical 12% improvement at native 128².

## Why the multi-resolution + KAN head was removed

The decorated head added 80 K parameters (`MultiScaleOutput2d` + KAN sub-modules) for a Δ = 0.0001 accuracy difference at noise level σ = 0.0001.  Following the polish-pass discussion in `RESEARCH_STATE.md` §22, the canonical model was redefined at the simpler operating point, with the decorated variant kept as a tested-and-rejected ablation row in `results/micro_ablation.csv`.
