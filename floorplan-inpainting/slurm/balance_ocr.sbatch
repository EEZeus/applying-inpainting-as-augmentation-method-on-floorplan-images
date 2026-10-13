#!/bin/bash
# Follow-up X2: OCR + embed the synthetic plans of the largest dose (b100).
#SBATCH --job-name=fp-balance-ocr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/balance-ocr-%j.out
#SBATCH --partition=RTXA6000,L40S,L40S-DSA,RTX3090,A100-40GB,A100-PCI,A100-80GB,A100-RP,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-/netscratch/espandar/easyocr_models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/netscratch/espandar/miniconda3/envs/ocr/bin/python
B=data/augmented_c_balance/b100/copy_paste_multi
"$PY" scripts/ocr_cache.py --coco "$B/instances_train_augmented.json" --image-dir "$B" --out "$B/ocr_tokens.json" --synthetic-only
"$PY" scripts/ocr_embed.py --cache "$B/ocr_tokens.json" --out "$B/ocr_features.npz"
