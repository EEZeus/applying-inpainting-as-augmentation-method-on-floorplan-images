#!/bin/bash
# Confirm stage R: evaluate last.pth of all 84 grid runs on the 21 held-out plans.
# Task id = (arm_idx * 2 + track_idx) * 3 + seed_idx.
#SBATCH --job-name=fp-confirm-rescore
#SBATCH --array=0-83%24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/confirm-rescore-%A_%a.out
#SBATCH --partition=L40S,L40S-DSA,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
RESCORE_ROOT="${RESCORE_ROOT:-outputs/rescore}"

ARMS=(e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13)
CFG=(
  configs/experiments/e0_original.yaml
  configs/experiments/e1_oversampling.yaml
  configs/experiments/e2_strategy_a.yaml
  configs/experiments/e3_strategy_a_weighted.yaml
  configs/experiments/e4_strategy_a_crops.yaml
  configs/experiments/e5_strategy_a_weighted_crops.yaml
  configs/experiments/e6_strategy_b.yaml
  configs/experiments/e7_strategy_b_weighted.yaml
  configs/experiments_c/e8_strategy_c_copy_paste.yaml
  configs/experiments_c/e9_strategy_c_feather_blend.yaml
  configs/experiments_c/e10_strategy_c_inpaint.yaml
  configs/experiments_c/e11_strategy_c_inpaint_weighted.yaml
  configs/experiments_lora/e12_lora_b.yaml
  configs/experiments_lora/e13_lora_b_weighted.yaml
)
# Same checkpoint roots as scripts/analyze_grid.py.
CKPT_ROOT=(
  outputs/experiments outputs/experiments outputs/experiments outputs/experiments
  outputs/experiments outputs/experiments outputs/experiments outputs/experiments
  outputs/experiments_c_multi outputs/experiments_c_multi outputs/experiments_c_multi
  outputs/experiments_c_multi outputs/experiments_lora outputs/experiments_lora
)
TRACKS=(multiclass agnostic)
SEEDS=(42 123 2026)

TID=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( TID % 3 ))]}
BLOCK=$(( TID / 3 ))
TRACK=${TRACKS[$(( BLOCK % 2 ))]}
ARM_IDX=$(( BLOCK / 2 ))
(( ARM_IDX < ${#ARMS[@]} )) || { echo "task ${TID} out of range" >&2; exit 2; }
ARM=${ARMS[$ARM_IDX]}
CONFIG=${CFG[$ARM_IDX]}
CKPT=${CKPT_ROOT[$ARM_IDX]}/${TRACK}/${ARM}/seed_${SEED}/checkpoints/last.pth
OUT_DIR=${RESCORE_ROOT}/${TRACK}/${ARM}/seed_${SEED}/heldout21_last
echo "task ${TID}: arm=${ARM} track=${TRACK} seed=${SEED} ckpt=${CKPT}"
test -f "${CONFIG}" || { echo "missing config ${CONFIG}" >&2; exit 2; }
test -f "${CKPT}" || { echo "missing checkpoint ${CKPT}" >&2; exit 2; }
test -f data/splits_heldout21/test.txt || { echo "run scripts/make_heldout21_split.py first" >&2; exit 2; }
if [[ -f "${OUT_DIR}/result.json" && -f "${OUT_DIR}/metrics/coco_results.json.gz" ]]; then
  echo "already done: ${OUT_DIR}"; exit 0
fi
FP_SPLIT_DIR=data/splits_heldout21 "${PY}" scripts/run_experiment.py \
  --config "${CONFIG}" --track "${TRACK}" --eval-checkpoint "${CKPT}" --eval-split test \
  --output-root "${RESCORE_ROOT}" --eval-tag "seed_${SEED}/heldout21_last" --dump-predictions
