#!/bin/bash
#SBATCH --job-name=fp-cv-analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/cv-analyze-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-PCI,RTXA6000,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
exec "${PY}" scripts/analyze_cv.py --folds 5 --tracks multiclass agnostic
