#!/bin/bash
#SBATCH --job-name=generate-lora-b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
#SBATCH --output=outputs/logs/generate-lora-b-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" scripts/generate_lora_strategy_b.py --config configs/augmentation_lora_b.yaml --retry-errors
"$PYTHON" scripts/validate_augmented_dataset.py data/augmented_lora_b/instances_train_augmented.json
