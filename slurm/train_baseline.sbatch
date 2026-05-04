#!/usr/bin/env bash
# Template SLURM job for a baseline operator run.
# Adapt the partition / qos / account lines to your local cluster.

#SBATCH --job-name=baseline_train
#SBATCH --partition=gpu              # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Source your venv / conda environment here.
# Then make sure third_party/ is populated:
#     bash baselines/install_baselines.sh

MODEL=${MODEL:-FNO}
DATA_PATH=${DATA_PATH:-./data/NavierStokes_V1e-5_N1200_T20.mat}

echo "=== baseline $MODEL start $(date) ==="
python baselines/ns_nsl.py \
    --model_name "$MODEL" \
    --data_path  "$DATA_PATH" \
    --baseline_root ./third_party/NSL
echo "=== baseline $MODEL done  $(date) ==="
