#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
EXPERIMENTS=$(sbatch --parsable slurm/run_lora_experiments.sbatch)
REPORT=$(sbatch --parsable --dependency="afterany:$EXPERIMENTS" slurm/report_lora_experiments.sbatch)
echo "LoRA detector experiment array: $EXPERIMENTS"
echo "LoRA isolated report: $REPORT"
