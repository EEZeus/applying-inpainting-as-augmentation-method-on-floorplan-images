#!/bin/bash
#SBATCH --job-name=fp-sam3-collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/sam3-collect-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-PCI,RTXA6000,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/collect_sam3_results.py
