#!/bin/bash
#SBATCH --job-name=lora-multi-build
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/lora-multi-build-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
REPLACEMENT_PERCENTAGE="${REPLACEMENT_PERCENTAGE:-5}"
PERCENTAGE_JITTER="${PERCENTAGE_JITTER:-2}"
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/build_lora_multiroom_dataset.py \
  --replacement-percentage "$REPLACEMENT_PERCENTAGE" --percentage-jitter "$PERCENTAGE_JITTER"
