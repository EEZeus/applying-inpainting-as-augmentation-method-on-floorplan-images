#!/bin/bash
#SBATCH --job-name=floorplan-c-exp
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/strategy-c-exp-%A_%a.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
CONFIGS=(e8_strategy_c_copy_paste.yaml e9_strategy_c_feather_blend.yaml
         e10_strategy_c_inpaint.yaml e11_strategy_c_inpaint_weighted.yaml)
INDEX=$((SLURM_ARRAY_TASK_ID % 4))
if (( SLURM_ARRAY_TASK_ID < 4 )); then TRACK=multiclass; else TRACK=agnostic; fi
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/run_experiment.py \
  --config "configs/experiments_c/${CONFIGS[$INDEX]}" --track "$TRACK" --seed 42 --resume

