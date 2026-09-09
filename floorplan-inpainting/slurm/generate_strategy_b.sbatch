#!/bin/bash
#SBATCH --job-name=floorplan-strategy-b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/strategy-b-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NUM_WORKERS="${NUM_WORKERS:-2}"
RESOLUTION="${RESOLUTION:-512}"
MODEL_CACHE="${MODEL_CACHE:-$PWD/data/cache/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-data/augmented_b}"
QA_DIR="${QA_DIR:-outputs/augmentation_b}"
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
ARGS=(--config configs/augmentation_b.yaml --num-workers "$NUM_WORKERS" --resolution "$RESOLUTION"
      --model-cache "$MODEL_CACHE" --output-dir "$OUTPUT_DIR" --qa-dir "$QA_DIR" --retry-errors)
"$PYTHON" scripts/generate_strategy_b.py "${ARGS[@]}" --limit 1
if [ "${SMOKE_ONLY:-0}" = 1 ]; then exit 0; fi
"$PYTHON" scripts/generate_strategy_b.py "${ARGS[@]}"
"$PYTHON" scripts/build_strategy_b_review.py --dataset-dir "$OUTPUT_DIR" --review "$QA_DIR/review.csv"
"$PYTHON" scripts/validate_augmented_dataset.py "$OUTPUT_DIR/instances_train_aug_b.json"
"$PYTHON" scripts/report_augmentation.py
