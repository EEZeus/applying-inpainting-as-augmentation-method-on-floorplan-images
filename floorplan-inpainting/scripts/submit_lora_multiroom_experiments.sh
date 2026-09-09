#!/bin/bash
# Usage: bash scripts/submit_lora_multiroom_experiments.sh [replacement_percentage] [percentage_jitter]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PERCENTAGE="${1:-5}"
JITTER="${2:-2}"
EXPORTS="ALL,REPLACEMENT_PERCENTAGE=$PERCENTAGE,PERCENTAGE_JITTER=$JITTER"
BUILD=$(sbatch --parsable --export="$EXPORTS" slurm/build_lora_multiroom.sbatch)
EXPERIMENTS=$(sbatch --parsable --dependency="afterok:$BUILD" --export="$EXPORTS" slurm/run_lora_multiroom_experiments.sbatch)
REPORT=$(sbatch --parsable --dependency="afterany:$EXPERIMENTS" --export="$EXPORTS" slurm/report_lora_multiroom.sbatch)
echo "Multi-room dataset ($PERCENTAGE% ± $JITTER%): $BUILD"
echo "E14/E15 detector experiment array: $EXPERIMENTS"
echo "Multi-room isolated report: $REPORT"
