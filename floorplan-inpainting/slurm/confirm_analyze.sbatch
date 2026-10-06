#!/bin/bash
# Final stage: pre-registered analysis and the report bundle.
#SBATCH --job-name=fp-confirm-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/confirm-analyze-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-confirm-analyze-${SLURM_JOB_ID:-local}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
FINAL_ROOT="${FINAL_ROOT:-outputs/final/confirm}"
"$PY" scripts/analyze_confirm.py --rescore-root "${RESCORE_ROOT:-outputs/rescore}" \
  --confirm-root "${CONFIRM_ROOT:-outputs/confirm}" --out "${FINAL_ROOT}" \
  --boot "${BOOT:-2000}" --rescore-boot "${RESCORE_BOOT:-1000}" --permutations "${PERMUTATIONS:-200000}"
"$PY" scripts/make_confirm_report.py --summary "${FINAL_ROOT}/summary.json" \
  --confirm-root "${CONFIRM_ROOT:-outputs/confirm}" --matched-root "${MATCHED_ROOT:-data/augmented_c_matched}" \
  --qa-root "${QA_ROOT:-outputs/augmentation_c_wide}" --out "${FINAL_ROOT}/report"
