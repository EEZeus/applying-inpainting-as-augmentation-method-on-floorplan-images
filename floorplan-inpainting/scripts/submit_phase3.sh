#!/bin/bash
# Submit all single-seed multiclass+agnostic experiments and automatic reports.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAIN=$(sbatch --parsable slurm/run_phase3_array.sbatch)
REPORT=$(sbatch --parsable --dependency="afterany:$MAIN" slurm/report_phase3.sbatch)
echo "Phase 3 array: $MAIN"
echo "Automatic report: $REPORT"
if [[ "${1:-}" == "--with-extra-seeds" ]]; then
  EXTRA=$(sbatch --parsable --dependency="afterok:$REPORT" slurm/run_phase3_extra_seeds.sbatch)
  FINAL=$(sbatch --parsable --dependency="afterany:$EXTRA" slurm/report_phase3.sbatch)
  echo "Automatic selected-method extra seeds: $EXTRA"
  echo "Final report: $FINAL"
else
  echo "Extra seeds were not submitted. Use --with-extra-seeds for the fully unattended chain."
fi
