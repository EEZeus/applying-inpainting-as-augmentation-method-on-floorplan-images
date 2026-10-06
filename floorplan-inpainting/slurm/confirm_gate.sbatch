#!/bin/bash
# Apply the pre-registered gates (futility for e8, generation rule for e16/e17).
#SBATCH --job-name=fp-confirm-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/logs/confirm-gate-%j.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
FORCE_ARGS=()
if [[ "${FORCE_ALL:-0}" == 1 ]]; then FORCE_ARGS=(--force-all); fi
"$PY" scripts/confirm_gate.py --rescore-root "${RESCORE_ROOT:-outputs/rescore}" \
  --matched-root "${MATCHED_ROOT:-data/augmented_c_matched}" \
  --out "${CONFIRM_ROOT:-outputs/confirm}/decisions.json" ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
