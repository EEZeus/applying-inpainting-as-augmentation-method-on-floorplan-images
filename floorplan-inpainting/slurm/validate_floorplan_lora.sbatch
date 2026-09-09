#!/bin/bash
#SBATCH --job-name=validate-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
#SBATCH --output=outputs/logs/validate-lora-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" scripts/validate_lora_generator.py --config configs/lora_training.yaml
