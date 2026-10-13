#!/bin/bash
# OCR study, stages 1-2 (descriptive): text vs visual room classification and
# two-stage AP on the 12 agnostic e0 dumps. CPU only.
#SBATCH --job-name=fp-ocr-diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/ocr-diag-%j.out
#SBATCH --partition=batch,RTXA6000,L40S,L40S-DSA,RTX3090,V100-32GB,A100-40GB,A100-PCI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=8
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PY" scripts/ocr_diagnostics.py --out "${FINAL_ROOT:-outputs/final/ocr_study}/diagnostics"
