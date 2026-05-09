# `data/` directory

This directory **does not contain any dataset files**. Datasets are too large for GitHub (the in-house 128² file alone is 1.5 GB, well past GitHub's 100 MB per-file limit), so the actual data lives elsewhere.

## In-house native-128² Navier–Stokes dataset

The only dataset *we* generated ourselves is hosted on **Zenodo**.

- **DOI**: [`10.5281/zenodo.20030455`](https://doi.org/10.5281/zenodo.20030455)
- **SHA-256**: `c0323e8dce484f6bee24a12273866a8fb9d34b78eb857ab315013518b72139a7`
- See [`../docs/ARTIFACTS.md`](../docs/ARTIFACTS.md) for the full record (provenance, verification, and regeneration fallback).

After downloading, place the file at `data/ns_v1e5_128_N1200_T20.mat` and the trainers / eval scripts will pick it up automatically.

If you prefer to regenerate the file from scratch (it is bit-identical given the same seed), run from the release root:

```bash
python scripts/generate_ns_128.py --out ./data/ns_v1e5_128_N1200_T20.mat
# ~6 hours on a single H100; produces a 1.5 GB scipy v5 .mat file
# sha256 matches the Zenodo deposit
```

The protocol (Li et al. 2020 vorticity-form pseudo-spectral solver — 2/3 dealiasing, Crank–Nicolson on the linear part, explicit Euler on the nonlinear advection, Matérn-spectrum Gaussian random-field initial conditions) is documented in [`../docs/DATA.md`](../docs/DATA.md).

## Public datasets (NS ν = 10⁻⁵/⁻⁴/⁻³ at 64², Shallow Water, Diffusion-Reaction, Active Matter)

These are not redistributed in this repository.  Each carries its own license; download from the original sources documented in [`../docs/DATA.md`](../docs/DATA.md).

## Layout contract for trainers

Once a `.mat` file is in this directory, the trainers and eval scripts pick it up automatically through `--data_root ./data` (their default).  Each `.mat` must carry a single field `u` of shape `(N_traj, S, S, T)` in fp32 — `MatReader` (in `spectranet/data.py`) handles both scipy v5 and HDF5 v7.3 formats transparently.

## License

The in-house 128² dataset (when downloaded from Zenodo) is released under the same MIT license as the rest of this repository (see top-level `LICENSE`).
