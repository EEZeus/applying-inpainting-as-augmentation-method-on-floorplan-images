#!/bin/bash
#SBATCH --job-name=floorplan-maskrcnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/train-%j.out
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_HOME=/netscratch/espandar/torch_cache

# Training parameters: edit these before submitting the job.
TRAIN_EPOCHS=25

CONDA_EXE=/netscratch/espandar/miniconda3/bin/conda
CONDA_ENV=inpainting

"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python -c "import torch, torchvision, pycocotools; print('Python environment: $CONDA_ENV'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

exec "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" \
    python scripts/train_baseline.py --epochs "$TRAIN_EPOCHS"
