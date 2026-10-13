#!/bin/bash
# OCR study, stage 4 data: erase OCR'd labels of planned frequent rooms with SD
# inpainting and render rare-class names (training plans only).
#SBATCH --job-name=fp-text-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/text-relabel-gen-%j.out
#SBATCH --partition=RTXA6000,L40S,L40S-DSA,A100-40GB,A100-PCI,A100-80GB,A100-RP,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=4
# The stored HF token expired (401 even for public repos); SD weights are cached locally.
export HF_HUB_OFFLINE=1
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
LIMIT_ARGS=()
if [[ -n "${GEN_LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${GEN_LIMIT}"); fi
"$PY" scripts/generate_text_relabel.py --config "${TEXT_CONFIG:-configs/augmentation_text.yaml}" \
  --model-cache "$PWD/data/cache/huggingface" --retry-errors ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
