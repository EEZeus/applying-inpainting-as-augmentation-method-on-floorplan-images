#!/bin/bash
# Confirmatory training (ARMS x SEEDS on TRACK), then last.pth is evaluated on the 21 held-out plans.
# Task id = arm_idx * N_SEEDS + seed_idx. Arms dropped by the gate exit 0.
#SBATCH --job-name=fp-confirm-train
#SBATCH --array=0-47%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/logs/confirm-train-%A_%a.out
#SBATCH --partition=L40S,L40S-DSA,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
CONFIRM_ROOT="${CONFIRM_ROOT:-outputs/confirm}"
MATCHED_ROOT="${MATCHED_ROOT:-data/augmented_c_matched}"
TRACK="${TRACK:-agnostic}"
read -r -a SEEDS <<< "${SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012}"
read -r -a ARMS <<< "${ARMS:-e0 e8 e16 e17}"
N_SEEDS=${#SEEDS[@]}
declare -A CFGMAP=(
  [e0]=configs/experiments/e0_original.yaml
  [e1]=configs/experiments/e1_oversampling.yaml
  [e2]=configs/experiments/e2_strategy_a.yaml
  [e8]=configs/experiments_c/e8_strategy_c_copy_paste.yaml
  [e16]=configs/experiments_confirm/e16_c_copy_paste_matched.yaml
  [e17]=configs/experiments_confirm/e17_c_inpaint_wide_matched.yaml
  [e18]=configs/experiments_ocr/e18_text_relabel.yaml
  [e19]=configs/experiments_ocr/e19_text_aware.yaml
  [e20]=configs/experiments_ocr/e20_text_aware_relabel.yaml
  [e21]=configs/experiments_followup/e21_text_aware_adamw.yaml
  [e22]=configs/experiments_followup/e22_c_balance_b20.yaml
  [e23]=configs/experiments_followup/e23_c_balance_b50.yaml
  [e24]=configs/experiments_followup/e24_c_balance_b100.yaml
)
TID=${SLURM_ARRAY_TASK_ID}
ARM_IDX=$(( TID / N_SEEDS ))
(( ARM_IDX < ${#ARMS[@]} )) || { echo "task ${TID} out of range" >&2; exit 2; }
ARM=${ARMS[$ARM_IDX]}
CONFIG=${CFGMAP[$ARM]}
SEED=${SEEDS[$(( TID % N_SEEDS ))]}
echo "task ${TID}: arm=${ARM} track=${TRACK} seed=${SEED} config=${CONFIG}"
if [[ "${SKIP_GATE:-0}" != 1 ]]; then
  RUN=$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['run'][sys.argv[2]]))" \
    "${CONFIRM_ROOT}/decisions.json" "${ARM}")
  if [[ "${RUN}" != 1 ]]; then
    echo "arm ${ARM} dropped by the pre-registered gate; nothing to do"; exit 0
  fi
fi
EXTRA=()
if [[ "${ARM}" == e16 || "${ARM}" == e17 ]]; then EXTRA=(--matched-data-root "${MATCHED_ROOT}"); fi
EPOCH_ARGS=()
if [[ -n "${EPOCHS:-}" ]]; then EPOCH_ARGS=(--epochs "${EPOCHS}"); fi
RUN_DIR=${CONFIRM_ROOT}/${TRACK}/${ARM}/seed_${SEED}
EVAL_DIR=${RUN_DIR}/heldout21_last
if [[ ! -f "${RUN_DIR}/result.json" ]]; then
  "$PY" scripts/run_experiment.py --config "${CONFIG}" --track "${TRACK}" --seed "${SEED}" \
    --output-root "${CONFIRM_ROOT}" --resume ${EXTRA[@]+"${EXTRA[@]}"} ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
fi
test -f "${RUN_DIR}/checkpoints/last.pth" || { echo "missing ${RUN_DIR}/checkpoints/last.pth" >&2; exit 3; }
FP_SPLIT_DIR=data/splits_heldout21 "$PY" scripts/run_experiment.py --config "${CONFIG}" --track "${TRACK}" \
  --eval-checkpoint "${RUN_DIR}/checkpoints/last.pth" --eval-split test --output-root "${CONFIRM_ROOT}" \
  --eval-tag "seed_${SEED}/heldout21_last" --dump-predictions ${EXTRA[@]+"${EXTRA[@]}"}
