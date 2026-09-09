#!/bin/bash
# Submit train-only crop building, LoRA training, validation, and automatic conversion integration.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
CROPS=$(sbatch --parsable slurm/build_lora_crops.sbatch)
TRAIN=$(sbatch --parsable --dependency="afterok:$CROPS" slurm/train_floorplan_lora.sbatch)
VALIDATE=$(sbatch --parsable --dependency="afterok:$TRAIN" slurm/validate_floorplan_lora.sbatch)
CONVERT=$(sbatch --parsable --dependency="afterok:$TRAIN" slurm/generate_lora_exemplar_v2.sbatch)
EXPORTS="ALL,REPLACEMENT_PERCENTAGE=5,PERCENTAGE_JITTER=2"
MULTI=$(sbatch --parsable --dependency="afterok:$CONVERT" --export="$EXPORTS" slurm/build_lora_multiroom.sbatch)
EXPERIMENTS=$(sbatch --parsable --dependency="afterok:$MULTI" --export="$EXPORTS" slurm/run_lora_multiroom_experiments.sbatch)
REPORT=$(sbatch --parsable --dependency="afterany:$EXPERIMENTS" --export="$EXPORTS" slurm/report_lora_multiroom.sbatch)
echo "LoRA crops: $CROPS"
echo "LoRA training: $TRAIN"
echo "Base-vs-LoRA validation: $VALIDATE"
echo "Rare-exemplar + LoRA conversions: $CONVERT"
echo "Semantic multi-room dataset: $MULTI"
echo "Semantic E14/E15 experiments: $EXPERIMENTS"
echo "Semantic multi-room report: $REPORT"
