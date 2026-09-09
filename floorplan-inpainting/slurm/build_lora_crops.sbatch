#!/bin/bash
#SBATCH --job-name=lora-crops
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/lora-crops-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" scripts/build_lora_training_crops.py --config configs/lora_training.yaml
