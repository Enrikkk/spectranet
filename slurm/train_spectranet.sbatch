#!/usr/bin/env bash
# Template SLURM job for SpectraNet training on a single H100.
# Adapt the partition / qos / account lines to your local cluster.

#SBATCH --job-name=spectranet_train
#SBATCH --partition=gpu             # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00              # ~100 minutes per 500-epoch run + slack
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to release/ root

mkdir -p logs runs

# Source your venv / conda environment here, e.g.:
# source ~/venvs/spectranet/bin/activate

CONFIG=${CONFIG:-configs/spectranet_ns_v1e5.yaml}
DATA_ROOT=${DATA_ROOT:-./data}

echo "=== SpectraNet train start $(date) ==="
echo "  config:    $CONFIG"
echo "  data_root: $DATA_ROOT"
python scripts/train_spectranet.py --config "$CONFIG" --data_root "$DATA_ROOT"
echo "=== SpectraNet train done  $(date) ==="
