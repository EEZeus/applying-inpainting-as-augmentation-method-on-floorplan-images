#!/bin/bash
# OCR study: pre-registered analysis (docs/preregistration_ocr.md) + report section.
#SBATCH --job-name=fp-ocr-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/ocr-analyze-%j.out
#SBATCH --partition=batch,RTXA6000,L40S,L40S-DSA,RTX3090,V100-32GB,A100-40GB,A100-PCI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-ocr-${SLURM_JOB_ID:-local}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
SEED_ARGS=()
if [[ -n "${SEEDS:-}" ]]; then read -r -a S <<< "${SEEDS}"; SEED_ARGS=(--seeds "${S[@]}"); fi
"$PY" scripts/analyze_ocr_study.py --study-root "${CONFIRM_ROOT:-outputs/ocr_study}" \
  --out "${FINAL_ROOT:-outputs/final/ocr_study}" --boot "${BOOT:-2000}" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"}
