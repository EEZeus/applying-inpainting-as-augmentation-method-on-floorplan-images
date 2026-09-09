#!/bin/bash
# Aggregate the Round-1 grid once the training array has drained.
#SBATCH --job-name=fp-grid-r1-analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/grid-r1-analyze-%j.out
#SBATCH --partition=RTXA6000,RTX3090,L40S,L40S-DSA,V100-32GB,A100-40GB,A100-PCI,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
exec "${PY}" scripts/analyze_grid.py --tracks multiclass agnostic
