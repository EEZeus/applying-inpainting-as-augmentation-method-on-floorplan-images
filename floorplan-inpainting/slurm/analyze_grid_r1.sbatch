#!/bin/bash
# Aggregate the Round-1 grid once the training array has drained.
#SBATCH --job-name=fp-grid-r1-analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/grid-r1-analyze-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
exec "${PY}" scripts/analyze_grid.py --tracks multiclass agnostic
