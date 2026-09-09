#!/bin/bash
#SBATCH --job-name=floorplan-p3-seeds
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/phase3-seeds-%A_%a.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
ROLES=(e0 e1 best_a best_b)
ROLE_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
BLOCK=$((SLURM_ARRAY_TASK_ID / 4))
if (( BLOCK < 2 )); then TRACK=multiclass; else TRACK=agnostic; fi
if (( BLOCK % 2 == 0 )); then SEED=123; else SEED=2026; fi
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/run_selected_seed.py \
  --role "${ROLES[$ROLE_INDEX]}" --track "$TRACK" --seed "$SEED"

