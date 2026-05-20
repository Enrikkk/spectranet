# Baseline operator adapters

These adapter scripts wire 17 published neural operator architectures into the shared SpectraNet protocol (850/150/200 split, T_in=10, T_out=10, 500 epochs, AdamW + OneCycleLR, relative L² loss).  We do not vendor the upstream code; instead, run `install_baselines.sh` to clone each repository at a pinned commit, then invoke the adapter you need.

## Install

```bash
bash baselines/install_baselines.sh
```

This populates `./third_party/{NSL,Transolver,FactFormer,gnot,OFormer,KoopmanLab,Orthogonal-Neural-operator,CNO}/`.  Each upstream repository carries its own license; please respect them before modifying or redistributing.

## Adapter inventory

| Adapter | Models covered | Upstream | Notes |
|---|---|---|---|
| `ns_nsl.py` | FNO, F-FNO, U-FNO, U-NO, LSM, MWT, U-Net, Galerkin Transformer, Transformer | thuml/Neural-Solver-Library | One scaffold dispatches all nine via `--model_name`.  Per-model paper-default hyperparameters live in `MODEL_DEFAULTS`. |
| `ns_transolver.py` | Transolver | thuml/Transolver | Unified-position embedding; slice attention. |
| `ns_factformer.py` | FactFormer | BaratiLab/FactFormer | Factorized 2D attention. |
| `ns_gnot.py` | GNOT | HaoZhongkai/GNOT | Linear cross-attention. |
| `ns_oformer.py` | OFormer | BaratiLab/OFormer | Spatial-temporal encoder + point-wise decoder.  A minimal `fourier_neural_operator.py` stub may be needed to satisfy the upstream `cnn_module` import chain — see comments in the adapter. |
| `ns_kno.py` | KNO2d | Koopman-Laboratory/KoopmanLab | Non-autoregressive: maps `(B, H, W, T_in) → (B, H, W, T_out)` in one forward pass.  We bypass the package `__init__.py` (which imports `numpy.lib.arraypad`, removed in NumPy ≥1.25) by loading `kno.py` directly via `importlib.util`. |
| `ns_ono.py` | ONO | zwei-lin/ONO | Orthogonal Neural Operator. |
| `ns_lsm.py` | LSM | thuml/Neural-Solver-Library | Standalone variant; the NSL adapter also covers LSM via `--model_name LSM`. |
| `ns_cno.py` | CNO | bogdanraonic3/ConvolutionalNeuralOperator | Convolutional Neural Operator. |

## Running a baseline

All adapters take a `--data_path` (default `./data/NavierStokes_V1e-5_N1200_T20.mat`) and a `--baseline_root` (default `./third_party/<name>/`).  Output artifacts (`*_losses.csv`, `*_test_results.csv`, `*_config.json`, `checkpoints/*_best.pt`) land in the working directory under `plots/` and `checkpoints/`.

```bash
# FNO run (yields test L² ≈ 0.1024, 4.75 M params)
python baselines/ns_nsl.py --model_name FNO \
       --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \
       --baseline_root ./third_party/NSL

# Transformer (full softmax over 4096 tokens) — yields test L² ≈ 0.0284
python baselines/ns_nsl.py --model_name Transformer \
       --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \
       --baseline_root ./third_party/NSL
```

A SLURM template for batch reproduction is provided at `slurm/train_baseline.sbatch`.

## Known caveats

- **Resolution.**  Numbers in the paper are reported at 64².  At higher resolution the comparative ranking can shift (notably the full-attention Transformer's ``O(N²)`` cost makes it impractical past ≈128²); see `docs/REPRODUCING.md`.
- **Hyperparameter parity.**  We use the paper-default lr / n_hidden / batch size for each model.  Different defaults can change the ranking; we deliberately do **not** retune per-baseline because that would re-introduce the unfairness that the unified protocol was designed to remove.
- **License.**  Each upstream repository has its own license.  KoopmanLab is GPL-3.0; the rest are MIT or BSD-3.  Our adapters are MIT (matching the rest of this release).  When publishing a fork that bundles upstream code, propagate each license appropriately.
