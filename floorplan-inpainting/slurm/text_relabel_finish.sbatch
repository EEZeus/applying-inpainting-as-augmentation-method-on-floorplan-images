#!/bin/bash
# OCR study, stage 4 data: OCR-verify each edit, compose the synthetic plans,
# then OCR + embed the synthetic plans for the text-aware model.
#SBATCH --job-name=fp-text-finish
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/text-relabel-finish-%j.out
#SBATCH --partition=RTXA6000,L40S,L40S-DSA,RTX3090,A100-40GB,A100-PCI,A100-80GB,A100-RP,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EASYOCR_MODEL_DIR="${EASYOCR_MODEL_DIR:-/netscratch/espandar/easyocr_models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
PY=/netscratch/espandar/miniconda3/envs/ocr/bin/python
CONFIG="${TEXT_CONFIG:-configs/augmentation_text.yaml}"
DATA=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output']['dataset_dir'])" "$CONFIG")
"$PY" scripts/text_relabel_finish.py --config "$CONFIG"
"$PY" scripts/ocr_cache.py --coco "${DATA}/instances_train_augmented.json" --image-dir "${DATA}" \
  --out "${DATA}/ocr_tokens.json" --synthetic-only
"$PY" scripts/ocr_embed.py --cache "${DATA}/ocr_tokens.json" --out "${DATA}/ocr_features.npz"
/netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/validate_augmented_dataset.py \
  "${DATA}/instances_train_augmented.json"
