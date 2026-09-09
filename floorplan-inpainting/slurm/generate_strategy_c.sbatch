#!/bin/bash
#SBATCH --job-name=floorplan-strategy-c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/strategy-c-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-strategy-c-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NUM_WORKERS="${NUM_WORKERS:-2}"
RESOLUTION="${RESOLUTION:-512}"
MODEL_CACHE="${MODEL_CACHE:-$PWD/data/cache/huggingface}"
DATASET_ROOT="${DATASET_ROOT:-data/augmented_c}"
QA_ROOT="${QA_ROOT:-outputs/augmentation_c}"
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
ARGS=(--config configs/augmentation_c.yaml --num-workers "$NUM_WORKERS" --resolution "$RESOLUTION"
      --model-cache "$MODEL_CACHE" --dataset-root "$DATASET_ROOT" --qa-root "$QA_ROOT" --retry-errors)
# Reusable preflight candidates use the same fingerprints and stable IDs as the full run.
"$PYTHON" scripts/generate_strategy_c.py "${ARGS[@]}" --mode all --limit 1
if [ "${SMOKE_ONLY:-0}" = 1 ]; then exit 0; fi
"$PYTHON" scripts/generate_strategy_c.py "${ARGS[@]}" --mode all
"$PYTHON" scripts/build_strategy_c_multiroom_dataset.py \
  --replacement-percentage "${REPLACEMENT_PERCENTAGE:-5}" \
  --percentage-jitter "${PERCENTAGE_JITTER:-2}" \
  --candidate-dir "$DATASET_ROOT/copy_paste" \
  --output-dir "$DATASET_ROOT/copy_paste_multi"
for MODE in copy_paste feather_blend diffusion_inpaint; do
  "$PYTHON" scripts/validate_augmented_dataset.py \
    "$DATASET_ROOT/$MODE/instances_train_aug_c.json"
done
"$PYTHON" scripts/validate_augmented_dataset.py \
  "$DATASET_ROOT/copy_paste_multi/instances_train_augmented.json"
