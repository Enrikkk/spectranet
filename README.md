# SpectraNet

**Parameter-efficient autoregressive spectral U-Net for PDE surrogates.**

SpectraNet bridges spectral operator learning and U-Net hierarchies through a *Residual-Target Spectral Block* and a *Semigroup-Consistency Loss*. On the canonical Navier–Stokes ν = 10⁻⁵ benchmark it reaches **0.0822 test L²** with **2,040,705 parameters** — **2.33× fewer than FNO** (4.75 M, L² = 0.1024) at ≈20 % lower error. The architectural advantage transports to **5 of 6** additional dataset/regime combinations and **strengthens at higher resolution** (native 128²: SpectraNet 0.0724 vs FNO 0.3080).

The accompanying paper is *Bridging Spectral Operator Learning and U-Net Hierarchies: SpectraNet for Stable Autoregressive PDE Surrogates* (Hernández Noguera, Ferdaus, Ioup, Abdelguerfi, Simeonov; 2026). The compiled PDF is at [`paper/paper.pdf`](paper/paper.pdf).

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

## Repository layout

| Path | Purpose |
|---|---|
| `spectranet/` | Installable Python package: the SpectraNet model, spectral and KAN layers, data loaders, losses, utilities. |
| `scripts/` | One trainer (`train_spectranet.py`) for SpectraNet across all 7 datasets, one trainer (`train_baseline.py`) for the 17 baselines, four eval scripts (`eval_{lipschitz,long_horizon,resolution_transfer,cross_viscosity}.py`), the persistence floor, the in-house 128² generator, and the figure-regeneration scripts under `scripts/figures/`. |
| `baselines/` | Adapter scripts that wire each third-party operator library to our training protocol, plus `install_baselines.sh` that `git clone`s the upstream repos at pinned commits. |
| `timing/` | GPU + CPU inference-timing harness used to produce Figure 4 and the timing-section appendix. |
| `configs/` | One YAML per canonical run (per-dataset SpectraNet config + per-baseline config). |
| `results/` | Canonical CSVs that back every table and figure in the paper (leaderboard, cross-PDE, multi-seed, Lipschitz, long-horizon, resolution transfer, …). |
| `figures/` | The exact PDF figures embedded in the paper, regenerable from `results/` via `scripts/figures/`. |
| `checkpoints/` | Pretrained weights for the SpectraNet on every dataset, the decorated-head ablation variant, the bottleneck-widened sanity check, the FNO, and the Transformer. |
| `data/` | Pointer-only README. The in-house native-128² dataset is hosted on Zenodo (see [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md)); other datasets are public — see [`docs/DATA.md`](docs/DATA.md) for download instructions. |
| `paper/paper.pdf` | The compiled paper PDF. |
| `tests/` | Smoke tests you can run without an H100 or a full dataset (param-count, imports, eval). |
| `slurm/` | H100 sbatch templates for cluster reproduction. |
| `docs/` | [`REPRODUCING.md`](docs/REPRODUCING.md) (paper-claim → command map), [`DATA.md`](docs/DATA.md), [`BASELINES.md`](docs/BASELINES.md), [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`ARTIFACTS.md`](docs/ARTIFACTS.md). |

## Reproducing every paper number

Read [`docs/REPRODUCING.md`](docs/REPRODUCING.md). It maps every table and figure in the paper to the exact command sequence that regenerates the underlying CSV and the rendered PDF.

## Datasets

- **Navier–Stokes ν = 10⁻⁵, 10⁻⁴, 10⁻³** at 64²: public, from the FNO release (Li et al., 2020). See [`docs/DATA.md`](docs/DATA.md) for the download URL.
- **Shallow Water, Diffusion–Reaction**: public, from PDEBench. See [`docs/DATA.md`](docs/DATA.md).
- **Active Matter**: public, from The Well. See [`docs/DATA.md`](docs/DATA.md).
- **Navier–Stokes ν = 10⁻⁵ at 128²**: *in-house*. Hosted on Zenodo — see [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for the DOI. Equivalently, the dataset is regenerable bit-identically by running `scripts/generate_ns_128.py` (~6 hours on a single H100); provenance and protocol are documented in [`docs/DATA.md`](docs/DATA.md).

## Baselines

The 17 baseline operators come from eight upstream repositories (NSL, CNO, FactFormer, GNOT, KoopmanLab, OFormer, ONO, Transolver). To respect upstream licenses we **do not vendor** their code; instead, run `bash baselines/install_baselines.sh` to clone each at the pinned commit we used. Our adapter scripts in `baselines/ns_*.py` then wire each operator to the shared training protocol described in the paper.

## Citation

```bibtex
@misc{hernandez2026spectranet,
  title  = {Bridging Spectral Operator Learning and U-Net Hierarchies:
            {SpectraNet} for Stable Autoregressive {PDE} Surrogates},
  author = {Hern{\'a}ndez Noguera, Enrique and Ferdaus, Md Meftahul and
            Ioup, Elias and Abdelguerfi, Mahdi and Simeonov, Julian},
  year   = {2026},
  url    = {https://github.com/Enrikkk/spectranet}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Authors

- **Enrique Hernández Noguera**¹ · ehernan8@uno.edu
- **Md Meftahul Ferdaus**¹ · mferdaus@uno.edu
- **Elias Ioup**²
- **Mahdi Abdelguerfi**¹
- **Julian Simeonov**²

¹ University of New Orleans, New Orleans, LA, USA
² Naval Research Laboratory, Stennis Space Center, MS, USA
