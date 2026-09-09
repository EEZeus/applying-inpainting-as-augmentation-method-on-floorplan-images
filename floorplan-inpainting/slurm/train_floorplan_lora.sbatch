#!/bin/bash
#SBATCH --job-name=floorplan-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=L40S-DSA,V100-32GB,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
#SBATCH --output=outputs/logs/floorplan-lora-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-lora-${SLURM_JOB_ID:-local}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
PYTHON=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PYTHON" -c "import torch, peft; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), peft.__version__)"
"$PYTHON" scripts/train_floorplan_lora.py --config configs/lora_training.yaml
