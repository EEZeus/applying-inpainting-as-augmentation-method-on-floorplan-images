#!/bin/bash
# Round-1 Mask R-CNN grid: seeds {123,2026} x {multiclass,agnostic} x 14 configs = 56 tasks.
# Seed 42 already exists and is read from disk by scripts/analyze_grid.py.
#SBATCH --job-name=fp-grid-r1
#SBATCH --array=0-55%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# 32G OOM-killed the augmented-data arms; seed 42 used 48G.
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/logs/grid-r1-%A_%a.out
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2

PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

# Task ids encode this order; do not reorder.
CONFIGS=(
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
SEEDS=(123 2026)
TRACKS=(multiclass agnostic)

TID=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$(( TID % 2 ))
BLOCK=$(( TID / 2 ))
TRACK_IDX=$(( BLOCK % 2 ))
CFG_IDX=$(( BLOCK / 2 ))

if (( CFG_IDX >= ${#CONFIGS[@]} )); then
  echo "task ${TID}: CFG_IDX ${CFG_IDX} out of range" >&2
  exit 2
fi

CONFIG=${CONFIGS[$CFG_IDX]}
TRACK=${TRACKS[$TRACK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "task ${TID}: config=${CONFIG} track=${TRACK} seed=${SEED}"
test -f "${CONFIG}" || { echo "missing config ${CONFIG}" >&2; exit 2; }

exec "${PY}" scripts/run_experiment.py \
  --config "${CONFIG}" --track "${TRACK}" --seed "${SEED}" --resume
