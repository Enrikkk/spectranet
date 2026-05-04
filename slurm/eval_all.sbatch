#!/usr/bin/env bash
# Template SLURM job to run every evaluation script against the canonical
# SpectraNet checkpoint.  Append rows to the per-eval CSVs in results/.
# Adapt the partition / qos / account lines to your local cluster.

#SBATCH --job-name=spectranet_eval_all
#SBATCH --partition=gpu              # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

CKPT=${CKPT:-./checkpoints/spectranet_ns_v1e5_canonical_best.pt}
DATA=${DATA:-./data/NavierStokes_V1e-5_N1200_T20.mat}

echo "=== eval suite start $(date) ==="

python scripts/eval_lipschitz.py          --ckpt "$CKPT" --data_path "$DATA" --rollout_T 1 --label spectranet_T1
python scripts/eval_lipschitz.py          --ckpt "$CKPT" --data_path "$DATA" --rollout_T 10 --label spectranet_T10
python scripts/eval_long_horizon.py       --ckpt "$CKPT" --data_path "$DATA" --T_max 100 --label spectranet
python scripts/eval_resolution_transfer.py --ckpt "$CKPT" --data_path "$DATA" --target_res 128 --upsample_method spectral_zeropad --label spectranet
python scripts/persistence_baseline.py    --data_path "$DATA"

echo "=== eval suite done  $(date) ==="
