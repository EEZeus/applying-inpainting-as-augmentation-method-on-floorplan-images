#!/bin/bash
# Follow-up X2 + X3 (descriptive): balanced two-stage text head; SAM 3 proposals + text head.
#SBATCH --job-name=fp-followup-diag
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/followup-diag-%j.out
#SBATCH --partition=batch,RTXA6000,L40S,L40S-DSA,RTX3090,V100-32GB,A100-40GB,A100-PCI
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg OMP_NUM_THREADS=8
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
B=data/augmented_c_balance/b100/copy_paste_multi
"$PY" scripts/ocr_diagnostics.py --out "${FINAL_ROOT:-outputs/final/followup}/diagnostics_balanced" \
  --extra-train-coco "$B/instances_train_augmented.json" --extra-ocr-cache "$B/ocr_tokens.json" \
  --extra-ocr-features "$B/ocr_features.npz" \
  --sam3-agnostic-dump outputs/sam3_baseline_agnostic/dumps/test/mask/coco_predictions_segm.json \
  --sam3-multiclass-dump outputs/sam3_baseline/dumps/test/mask/coco_predictions_segm.json
