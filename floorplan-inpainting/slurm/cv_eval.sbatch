#!/bin/bash
# Rotated held-out CV: evaluate 6 seed-42 grid checkpoints on 5 folds.
# 6 arms x 2 tracks x 5 folds = 60 eval-only tasks.
#SBATCH --job-name=fp-cv-eval
#SBATCH --array=0-59%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=outputs/logs/cv-eval-%A_%a.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

# Task ids encode (arm, track, fold); do not reorder.
ARMS=(e0 e1 e6 e8 e10 e12)
CFG=(
  configs/experiments/e0_original.yaml
  configs/experiments/e1_oversampling.yaml
  configs/experiments/e6_strategy_b.yaml
  configs/experiments_c/e8_strategy_c_copy_paste.yaml
  configs/experiments_c/e10_strategy_c_inpaint.yaml
  configs/experiments_lora/e12_lora_b.yaml
)
CKPT_ROOT=(
  outputs/experiments
  outputs/experiments
  outputs/experiments
  outputs/experiments_c_multi
  outputs/experiments_c_multi
  outputs/experiments_lora
)
TRACKS=(multiclass agnostic)

TID=${SLURM_ARRAY_TASK_ID}
FOLD=$(( TID % 5 ))
T=$(( TID / 5 ))
TRACK_IDX=$(( T % 2 ))
ARM_IDX=$(( T / 2 ))

ARM=${ARMS[$ARM_IDX]}
TRACK=${TRACKS[$TRACK_IDX]}
CONFIG=${CFG[$ARM_IDX]}
CKPT=${CKPT_ROOT[$ARM_IDX]}/${TRACK}/${ARM}/seed_42/checkpoints/best.pth

echo "task ${TID}: arm=${ARM} track=${TRACK} fold=${FOLD} ckpt=${CKPT}"
test -f "${CONFIG}" || { echo "missing config ${CONFIG}" >&2; exit 2; }
test -f "${CKPT}"   || { echo "missing checkpoint ${CKPT}" >&2; exit 2; }
test -f "data/splits_cv/fold${FOLD}/test.txt" || { echo "run scripts/make_cv_splits.py first" >&2; exit 2; }

export FP_SPLIT_DIR="data/splits_cv/fold${FOLD}"
exec "${PY}" scripts/run_experiment.py \
  --config "${CONFIG}" --track "${TRACK}" \
  --eval-checkpoint "${CKPT}" --eval-split test \
  --output-root "outputs/cv/fold${FOLD}" --eval-tag eval
