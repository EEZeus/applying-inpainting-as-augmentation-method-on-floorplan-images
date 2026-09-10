#!/bin/bash
# Experiment 3, stage 1: extract frozen DINOv2 features for all crop sets.
# Frozen dinov2-base at 224 px, batch 16 -> < 3 GB GPU; any GPU is fine.
#SBATCH --job-name=fp-dino-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/dino-extract-%j.out
#SBATCH --partition=RTX3090,L40S,L40S-DSA,V100-32GB,A100-40GB,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

REAL_IMGS="../area_segmentation_coco/images"
DEDUP="data/processed/instances_deduplicated.json"
PROP_JSON="outputs/sam3_baseline_agnostic/dumps/test/mask/coco_predictions_segm.json"
TEST_COCO="data/sam3_floorplan/test/_annotations.coco.json"

test -f "$PROP_JSON" || { echo "missing SAM3 agnostic test proposals: $PROP_JSON" >&2; exit 2; }

if [[ -n "${DINO_SMOKE:-}" ]]; then
  OUT=outputs/dino_smoke
  echo "SMOKE: limited extraction -> $OUT"
  "$PY" scripts/dino_extract_crops.py --task gt --coco "$DEDUP" --image-dir "$REAL_IMGS" \
    --split-file data/splits/test.txt --out "$OUT/gt_test.npz" --limit 30
  "$PY" scripts/dino_extract_crops.py --task proposals --pred-json "$PROP_JSON" \
    --coco "$TEST_COCO" --image-dir "$REAL_IMGS" \
    --out "$OUT/test_proposals.npz" --dets-out "$OUT/test_proposals.dets.json" --limit 30
  echo "smoke extraction OK"
  exit 0
fi

OUT=outputs/dino
"$PY" scripts/dino_extract_crops.py --task gt --coco "$DEDUP" --image-dir "$REAL_IMGS" \
  --split-file data/splits/train.txt --out "$OUT/gt_train.npz"
"$PY" scripts/dino_extract_crops.py --task gt --coco "$DEDUP" --image-dir "$REAL_IMGS" \
  --split-file data/splits/val.txt --out "$OUT/gt_val.npz"
"$PY" scripts/dino_extract_crops.py --task gt --coco "$DEDUP" --image-dir "$REAL_IMGS" \
  --split-file data/splits/test.txt --out "$OUT/gt_test.npz"
"$PY" scripts/dino_extract_crops.py --task gt \
  --coco data/augmented_a/crops/instances_train_aug_a_crops.json \
  --image-dir data/augmented_a/crops --only-cats 11,12,13,14,15,16 \
  --out "$OUT/synth_rare.npz"
"$PY" scripts/dino_extract_crops.py --task proposals --pred-json "$PROP_JSON" \
  --coco "$TEST_COCO" --image-dir "$REAL_IMGS" \
  --out "$OUT/test_proposals.npz" --dets-out "$OUT/test_proposals.dets.json"
echo "full extraction OK -> $OUT"
