# SpectraNet

> Anonymous code release for the NeurIPS 2026 submission **"Bridging Spectral Operator Learning and U-Net Hierarchies: SpectraNet for Stable Autoregressive PDE Surrogates"**.

**Headline.** SpectraNet reaches **0.0822 test L²** on the canonical Navier–Stokes ν = 10⁻⁵ benchmark with **2,040,705 parameters** — **2.33× fewer than canonical FNO** (4.75 M, L² = 0.1024) at ≈20% lower error. The architectural advantage transports to **5 of 6** additional dataset/regime combinations and **strengthens at higher resolution** (native 128²: SpectraNet 0.0724 vs FNO 0.3080).

## Quick start

```bash
# install
pip install -e .

# canonical reproduction (NS ν = 1e-5, 64², ~100 min on a single H100)
python scripts/train_spectranet.py \
       --config configs/spectranet_ns_v1e5.yaml \
       --data_root ./data
# → prints best_test_l2 ≈ 0.0822

# eval-only path (uses shipped pretrained weights)
python scripts/eval_long_horizon.py \
       --ckpt   checkpoints/spectranet_ns_v1e5_canonical_best.pt \
       --model_kind ar2d
```

## What's in this repository

| Path | Purpose |
|---|---|
| `spectranet/` | Installable Python package: `SUNet2d` (the SpectraNet model), spectral and KAN layers, data loaders, losses, utilities. |
| `scripts/` | One trainer (`train_spectranet.py`) for SpectraNet across all 7 datasets, one trainer (`train_baseline.py`) for the 17 baselines, four eval scripts (`eval_{lipschitz,long_horizon,resolution_transfer,cross_viscosity}.py`), the persistence floor, the in-house 128² generator, and the figure-regeneration scripts under `scripts/figures/`. |
| `baselines/` | First-party adapter scripts that wire each third-party operator library to our gold-standard protocol, plus `install_baselines.sh` that `git clone`s the upstream repos at pinned commits. |
| `timing/` | GPU + CPU inference-timing harness used to produce Figure 4 and the timing-section appendix. |
| `configs/` | One YAML per canonical run (per-dataset SpectraNet config + per-baseline config). |
| `results/` | Canonical CSVs that back every table and figure in the paper (leaderboard, cross-PDE, multi-seed, Lipschitz, long-horizon, resolution transfer, …). |
| `figures/` | The exact PDF figures embedded in the paper, regenerable from `results/` via `scripts/figures/`. |
| `checkpoints/` | Pretrained weights for the canonical SpectraNet on every dataset, the decorated-head ablation variant, the bottleneck-widened sanity check, the canonical FNO, and the canonical Transformer. |
| `data/` | Holds only a README pointing to data sources.  The in-house native-128² dataset is hosted on Zenodo (see `docs/ARTIFACTS.md`); other datasets are public — see `docs/DATA.md` for download instructions. |
| `paper/paper.pdf` | The anonymized submission PDF. |
| `tests/` | Smoke tests reviewers can run without an H100 or a full dataset (param-count, imports, eval). |
| `slurm/` | Optional H100 sbatch templates for cluster reproduction. |
| `docs/` | `REPRODUCING.md` (paper-claim → command map), `DATA.md`, `BASELINES.md`, `ARCHITECTURE.md`, `ARTIFACTS.md`. |

## Reproducing every paper number

Read [`docs/REPRODUCING.md`](docs/REPRODUCING.md). It maps every table and figure in the paper to the exact command sequence that regenerates the underlying CSV and the rendered PDF.

## Datasets

- **Navier–Stokes ν = 10⁻⁵, 10⁻⁴, 10⁻³** at 64²: public, from the FNO release (Li et al., 2020). See `docs/DATA.md` for the download URL.
- **Shallow Water, Diffusion–Reaction**: public, from PDEBench. See `docs/DATA.md`.
- **Active Matter**: public, from The Well. See `docs/DATA.md`.
- **Navier–Stokes ν = 10⁻⁵ at 128²**: *in-house*. Hosted separately on Zenodo (the file is too large for GitHub) — see [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for the DOI.  Equivalently, the dataset is regenerable bit-identically by running `scripts/generate_ns_128.py` (~6 hours on a single H100); provenance and protocol are documented in [`docs/DATA.md`](docs/DATA.md).

## Baselines

The 17 baseline operators come from eight upstream repositories (NSL, CNO, FactFormer, GNOT, KoopmanLab, OFormer, ONO, Transolver). To respect upstream licenses we **do not vendor** their code; instead, run `bash baselines/install_baselines.sh` to clone each at the pinned commit we used. Our adapter scripts in `baselines/ns_*.py` then wire each operator to the gold-standard training protocol described in the paper.

## Anonymity

This release is anonymous for double-blind review. See [`ANONYMITY.md`](ANONYMITY.md).

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@inproceedings{anonymous2026spectranet,
  title     = {Bridging Spectral Operator Learning and U-Net Hierarchies:
               {SpectraNet} for Stable Autoregressive {PDE} Surrogates},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
