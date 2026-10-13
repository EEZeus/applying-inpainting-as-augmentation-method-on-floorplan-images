#!/bin/bash
# Follow-up X1: Strategy C copy-paste at one dose per array task (0=b20, 1=b50, 2=b100),
# then compose the multi-room dataset. No diffusion, CPU only.
#SBATCH --job-name=fp-balance-gen
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/balance-gen-%A_%a.out
#SBATCH --partition=batch,RTXA6000,L40S,L40S-DSA,RTX3090,V100-32GB,A100-40GB,A100-PCI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=2
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
DOSES=(20 50 100)
DOSE=${DOSES[${SLURM_ARRAY_TASK_ID}]}
CONFIG=configs/augmentation_c_balance_b${DOSE}.yaml
LIMIT_ARGS=()
if [[ -n "${GEN_LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${GEN_LIMIT}"); fi
"$PY" scripts/generate_strategy_c.py --config "$CONFIG" --mode copy_paste --num-workers 6 --retry-errors \
  ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
"$PY" scripts/build_balance_dataset.py --config "$CONFIG"
"$PY" scripts/validate_augmented_dataset.py "data/augmented_c_balance/b${DOSE}/copy_paste_multi/instances_train_augmented.json"
