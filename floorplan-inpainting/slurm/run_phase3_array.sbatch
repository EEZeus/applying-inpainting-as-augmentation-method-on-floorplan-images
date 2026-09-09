#!/bin/bash
#SBATCH --job-name=floorplan-p3
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/phase3-%A_%a.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
CONFIGS=(
  e0_original.yaml e1_oversampling.yaml e2_strategy_a.yaml e3_strategy_a_weighted.yaml
  e4_strategy_a_crops.yaml e5_strategy_a_weighted_crops.yaml e6_strategy_b.yaml e7_strategy_b_weighted.yaml
)
INDEX=$((SLURM_ARRAY_TASK_ID % 8))
if (( SLURM_ARRAY_TASK_ID < 8 )); then TRACK=multiclass; else TRACK=agnostic; fi
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/run_experiment.py \
  --config "configs/experiments/${CONFIGS[$INDEX]}" --track "$TRACK" --seed 42 --resume

