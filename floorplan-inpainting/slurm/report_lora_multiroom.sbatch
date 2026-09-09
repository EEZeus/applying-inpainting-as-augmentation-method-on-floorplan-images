#!/bin/bash
#SBATCH --job-name=lora-multi-report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/lora-multi-report-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
REPLACEMENT_PERCENTAGE="${REPLACEMENT_PERCENTAGE:-5}"
PERCENTAGE_JITTER="${PERCENTAGE_JITTER:-2}"
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/report_lora_multiroom.py \
  --replacement-percentage "$REPLACEMENT_PERCENTAGE" --percentage-jitter "$PERCENTAGE_JITTER"
