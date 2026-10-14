#!/bin/bash
# Round-1 grid chain: 1-task smoke -> 56-task array (seeds 123, 2026) -> scripts/analyze_grid.py.
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
