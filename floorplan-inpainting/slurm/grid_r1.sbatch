#!/bin/bash
# Round-1 experiment grid: Mask R-CNN downstream, existing augmentation arms.
# seed 42 is already complete for all 14 arms x 2 tracks (read from disk by
# scripts/analyze_grid.py); this array fills the 2 new seeds:
#   seeds {123,2026} x tracks {multiclass,agnostic} x 14 configs = 56 tasks.
# No new pipeline code: every config + augmented dataset already exists.
# Small-GPU partitions only (batch 1 Mask R-CNN peaks well under 24 GB); the
# 80 GB tier is intentionally excluded so tasks are not stuck behind SAM3-class
# queues.
#SBATCH --job-name=fp-grid-r1
#SBATCH --array=0-55%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
# 32G OOM-killed the augmented-data arms (e6/e7/e9/e10/e11/e12/e13) mid-run;
# seed_42 used 48G. 64G gives headroom (these nodes have 480G+).
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/logs/grid-r1-%A_%a.out
# Broad node pool for throughput. V100-32GB nodes ran ~8x slower here, but the
# 16 h walltime absorbs that and Slurm prefers faster free nodes anyway.
#SBATCH --partition=L40S,L40S-DSA,V100-32GB,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2

PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

# 14 configs, index order is fixed and must not be reordered (task ids encode it).
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
