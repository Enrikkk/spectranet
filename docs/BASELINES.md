# Running the baselines

The 17 baseline neural operators in the paper come from 8 upstream repositories.  We ship our **first-party adapter scripts** that wire each baseline into the gold-standard SpectraNet protocol; we do **not** vendor third-party code.  Run `bash baselines/install_baselines.sh` to clone every upstream at the pinned commit we used.

## Adapter inventory

See [`baselines/README.md`](../baselines/README.md) for the complete adapter table.  Quick summary:

| Adapter | Models | Upstream |
|---|---|---|
| `baselines/ns_nsl.py` | FNO, F-FNO, U-FNO, U-NO, LSM, MWT, U-Net, Galerkin Transformer, Transformer | thuml/Neural-Solver-Library |
| `baselines/ns_transolver.py` | Transolver | thuml/Transolver |
| `baselines/ns_factformer.py` | FactFormer | BaratiLab/FactFormer |
| `baselines/ns_gnot.py` | GNOT | HaoZhongkai/GNOT |
| `baselines/ns_oformer.py` | OFormer | BaratiLab/OFormer |
| `baselines/ns_kno.py` | KNO2d | Koopman-Laboratory/KoopmanLab |
| `baselines/ns_ono.py` | ONO | zwei-lin/ONO |
| `baselines/ns_lsm.py` | LSM | thuml/Neural-Solver-Library |
| `baselines/ns_cno.py` | CNO | bogdanraonic3/ConvolutionalNeuralOperator |

## Canonical FNO reproduction (paper baseline)

```bash
bash baselines/install_baselines.sh
python baselines/ns_nsl.py \
       --model_name FNO \
       --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \
       --baseline_root ./third_party/NSL
# expected: best_test_l2 ≈ 0.1024 (4.75 M params)
```

## Full-attention Transformer (the leaderboard winner — see paper §6.4)

```bash
python baselines/ns_nsl.py \
       --model_name Transformer \
       --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \
       --baseline_root ./third_party/NSL
# expected: best_test_l2 ≈ 0.0284 (4.38 M params)
```

The paper analyzes why the NSL Transformer wins at 64² — it is the only architecture in the leaderboard that uses exact `O(N²)` softmax attention over all 4096 tokens, while every other Transformer-family baseline trades that away for sub-quadratic scaling at higher resolutions.

## Per-baseline notes

| Baseline | Caveat |
|---|---|
| OFormer | A minimal `fourier_neural_operator.py` stub may need to be placed in `third_party/OFormer/` to satisfy upstream's `cnn_module` import chain.  See comments in `baselines/ns_oformer.py`. |
| KNO2d | Bypasses upstream `__init__.py` (which imports `numpy.lib.arraypad`, removed in NumPy ≥1.25) by loading `kno.py` directly via `importlib.util`. |
| GNOT, Transolver, FactFormer | Use unified-position embeddings via `--unified_pos 1`; the adapter sets this automatically. |

## SLURM template

For batch reproduction on an H100 cluster, see `slurm/train_baseline.sbatch` (template; adapt to your scheduler).
