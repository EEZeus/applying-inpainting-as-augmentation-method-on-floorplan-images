#!/bin/bash
#SBATCH --job-name=floorplan-inpaint-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/inpaint-a-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Override on submission, e.g. NUM_WORKERS=4 OUTPUT_DIR=/scratch/run sbatch ...
NUM_WORKERS="${NUM_WORKERS:-2}"
RESOLUTION="${RESOLUTION:-512}"
MODEL_CACHE="${MODEL_CACHE:-$PWD/data/cache/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-data/augmented_a}"
SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
CONDA_EXE=/netscratch/espandar/miniconda3/bin/conda
CONDA_ENV=inpainting

mkdir -p "$MODEL_CACHE" "$OUTPUT_DIR" outputs/logs

"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python -c "import torch, diffusers; print('GPU:', torch.cuda.get_device_name(0)); print('Diffusers:', diffusers.__version__)"

# The smoke sample is reused by the full run.
"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/generate_inpainting_a.py \
        --config configs/augmentation_a.yaml \
        --limit "$SMOKE_LIMIT" \
        --num-workers "$NUM_WORKERS" \
        --resolution "$RESOLUTION" \
        --model-cache "$MODEL_CACHE" \
        --output-dir "$OUTPUT_DIR" \
        --retry-errors

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/generate_inpainting_a.py \
        --config configs/augmentation_a.yaml \
        --num-workers "$NUM_WORKERS" \
        --resolution "$RESOLUTION" \
        --model-cache "$MODEL_CACHE" \
        --output-dir "$OUTPUT_DIR" \
        --retry-errors
