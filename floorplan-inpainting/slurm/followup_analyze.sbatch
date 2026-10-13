#!/bin/bash
# Follow-up analysis (docs/preregistration_followup.md) + report section.
#SBATCH --job-name=fp-followup-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/followup-analyze-%j.out
#SBATCH --partition=batch,RTXA6000,L40S,L40S-DSA,RTX3090,V100-32GB,A100-40GB,A100-PCI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-followup-${SLURM_JOB_ID:-local}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PY" scripts/analyze_followup.py --followup-root "${CONFIRM_ROOT:-outputs/followup}" \
  --out "${FINAL_ROOT:-outputs/final/followup}" --boot "${BOOT:-2000}"
