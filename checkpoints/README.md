# Pretrained checkpoints

Each `.pt` file is a bare PyTorch `state_dict`.  Load with the matching `SUNet2d` configuration from `docs/ARTIFACTS.md`.

| Filename | Architecture | Trained on | Best test L² | Param count |
|---|---|---|---|---|
| `spectranet_ns_v1e5_canonical_best.pt` | canonical (single head, no KAN) | NS ν=10⁻⁵, 64² | 0.0822 | 2,040,705 |
| `spectranet_ns_v1e5_decorated_best.pt` | decorated (multi-res + KAN head) | NS ν=10⁻⁵, 64² | 0.0821 | 2,124,166 |
| `spectranet_ns_v1e5_chc3_best.pt`      | canonical, `--ch_cap 3` | NS ν=10⁻⁵, 64² | 0.0819 | 2,319,745 |
| `spectranet_ns_v1e3_best.pt`           | decorated | NS ν=10⁻³ | 0.0011 | 2,124,166 |
| `spectranet_ns_v1e4_best.pt`           | canonical | NS ν=10⁻⁴ | 0.01521 | 2,040,705 |
| `spectranet_sw_best.pt`                | decorated | Shallow Water | 0.0012 | 2,124,166 |
| `spectranet_dr_best.pt`                | decorated | Diffusion-Reaction | 0.0201 | 2,124,166 |
| `spectranet_am_best.pt`                | decorated | Active Matter | 0.00170 | 2,124,166 |
| `spectranet_ns_v1e5_128_best.pt`       | decorated, S=128 | NS ν=10⁻⁵, 128² | 0.0724 | 2,124,166 |
| `fno_canonical_best.pt`                | FNO (NSL) | NS ν=10⁻⁵, 64² | 0.1024 | 4,749,377 |
| `transformer_canonical_best.pt`        | NSL Transformer (full softmax) | NS ν=10⁻⁵, 64² | 0.0284 | 4,381,441 |

The "decorated" variant differs from "canonical" only in the output projection (`output_mode='multiscale_mlp'` instead of `'single'`); see `docs/ARCHITECTURE.md`.

For loading code, see `docs/ARTIFACTS.md` or use `scripts/_eval_common.load_spectranet`, which auto-resolves the architecture from a sibling `*_config.json`.
