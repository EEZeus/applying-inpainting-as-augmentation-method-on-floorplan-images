#!/bin/bash
# Round-1 grid as a Slurm dependency chain, submitted from an extracted release.
#
#   smoke  : 1 task (e0/multiclass/seed123, a fresh from-scratch run)
#            -> catches a broken training/eval path, env or partition problem early
#   full   : the 56-task grid (seeds 123,2026), starts only afterok:smoke
#   analyze: scripts/analyze_grid.py, runs afterany:full (also reads seed 42 from disk)
#
# Run this from the extracted release directory (so $SLURM_SUBMIT_DIR is the
# release and the sbatch --chdir matches it). Usage: bash scripts/submit_grid_r1.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"

SBATCH_GRID="${HERE}/slurm/grid_r1.sbatch"
SBATCH_ANALYZE="${HERE}/slurm/analyze_grid_r1.sbatch"

bash -n "${SBATCH_GRID}"
bash -n "${SBATCH_ANALYZE}"

SMOKE=$(sbatch --parsable --chdir="${HERE}" --array=0-0 "${SBATCH_GRID}")
echo "smoke     : ${SMOKE}  (task 0 = e0/multiclass/seed123)"

FULL=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${SMOKE}" \
  --array=0-55%16 "${SBATCH_GRID}")
echo "full grid : ${FULL}  (tasks 0-55, afterok:${SMOKE})"

ANALYZE=$(sbatch --parsable --chdir="${HERE}" --dependency="afterany:${FULL}" \
  "${SBATCH_ANALYZE}")
echo "analyze   : ${ANALYZE}  (afterany:${FULL})"

echo
echo "chain: ${SMOKE} -> ${FULL} -> ${ANALYZE}"
echo "logs : ${HERE}/outputs/logs/grid-r1-*.out"
