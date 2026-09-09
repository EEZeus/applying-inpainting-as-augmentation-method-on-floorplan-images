#!/bin/bash
#SBATCH --job-name=lora-multi-exp
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/lora-multi-exp-%A_%a.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
REPLACEMENT_PERCENTAGE="${REPLACEMENT_PERCENTAGE:-5}"
PERCENTAGE_JITTER="${PERCENTAGE_JITTER:-2}"
CONFIGS=(e14_lora_multi.yaml e15_lora_multi_weighted.yaml)
INDEX=$((SLURM_ARRAY_TASK_ID % 2))
if (( SLURM_ARRAY_TASK_ID < 2 )); then TRACK=multiclass; else TRACK=agnostic; fi
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/run_experiment.py \
  --config "configs/experiments_lora/${CONFIGS[$INDEX]}" --track "$TRACK" --seed 42 --resume \
  --replacement-percentage "$REPLACEMENT_PERCENTAGE" --percentage-jitter "$PERCENTAGE_JITTER"
