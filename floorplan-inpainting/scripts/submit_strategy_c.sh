#!/bin/bash
# Submit Strategy C generation, its isolated experiment matrix, and its report.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GENERATION=$(sbatch --parsable slurm/generate_strategy_c.sbatch)
EXPERIMENTS=$(sbatch --parsable --dependency="afterok:$GENERATION" slurm/run_strategy_c_experiments.sbatch)
REPORT=$(sbatch --parsable --dependency="afterany:$EXPERIMENTS" slurm/report_strategy_c.sbatch)
echo "Strategy C generation: $GENERATION"
echo "Strategy C experiments: $EXPERIMENTS"
echo "Strategy C report: $REPORT"

