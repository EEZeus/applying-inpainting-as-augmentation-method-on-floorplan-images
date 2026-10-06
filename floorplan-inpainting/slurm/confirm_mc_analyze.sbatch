#!/bin/bash
# Multiclass follow-up: pre-registered analysis and report subsection.
#SBATCH --job-name=fp-confirm-mc-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/confirm-mc-analyze-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-confirm-mc-${SLURM_JOB_ID:-local}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PY" scripts/analyze_confirm_mc.py --confirm-root "${CONFIRM_ROOT:-outputs/confirm_mc}" \
  --out "${FINAL_ROOT:-outputs/final/confirm_mc}" --boot "${BOOT:-2000}" --permutations "${PERMUTATIONS:-200000}"
