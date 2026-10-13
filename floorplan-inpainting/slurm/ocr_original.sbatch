#!/bin/bash
# OCR study, stage 0: full-resolution tiled OCR of all 56 plans + token features.
#SBATCH --job-name=fp-ocr-original
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/ocr-original-%j.out
#SBATCH --partition=RTXA6000,L40S,L40S-DSA,RTX3090,A100-40GB,A100-PCI,A100-80GB,A100-RP,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-/netscratch/espandar/easyocr_models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/netscratch/espandar/miniconda3/envs/ocr/bin/python
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
LIMIT_ARGS=()
if [[ -n "${OCR_LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${OCR_LIMIT}"); fi
OCR_ROOT="${OCR_ROOT:-data/ocr}"
"$PY" scripts/ocr_cache.py --coco data/processed/instances_deduplicated.json \
  --image-dir ../area_segmentation_coco/images --out "${OCR_ROOT}/original_tokens.json" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
"$PY" scripts/ocr_embed.py --cache "${OCR_ROOT}/original_tokens.json" --out "${OCR_ROOT}/original_features.npz"
