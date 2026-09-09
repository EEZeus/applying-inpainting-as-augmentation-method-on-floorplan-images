#!/bin/bash
#SBATCH --job-name=floorplan-agnostic-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/eval-agnostic-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME=/netscratch/espandar/torch_cache
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints_agnostic/best.pth}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

"$PYTHON" scripts/create_class_agnostic_dataset.py
"$PYTHON" -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
exec "$PYTHON" scripts/evaluate_baseline.py \
    --dataset-config configs/dataset_agnostic.yaml \
    --model-config configs/baseline_agnostic.yaml \
    --checkpoint "$CHECKPOINT" \
    --split "$EVAL_SPLIT"
