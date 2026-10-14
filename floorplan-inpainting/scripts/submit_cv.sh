#!/bin/bash
# Rotated held-out CV chain: build splits -> 1-task smoke -> 60-task eval array -> analyze.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python

bash -n slurm/cv_eval.sbatch
bash -n slurm/analyze_cv.sbatch

"${PY}" scripts/make_cv_splits.py --folds 5 >/dev/null
echo "cv splits built: $(ls data/splits_cv)"

SMOKE=$(sbatch --parsable --chdir="${HERE}" --array=0-0 slurm/cv_eval.sbatch)
echo "smoke   : ${SMOKE}  (task 0 = e0/multiclass/fold0)"

FULL=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${SMOKE}" \
  --array=0-59%20 slurm/cv_eval.sbatch)
echo "cv eval : ${FULL}  (tasks 0-59, afterok:${SMOKE})"

ANALYZE=$(sbatch --parsable --chdir="${HERE}" --dependency="afterany:${FULL}" slurm/analyze_cv.sbatch)
echo "analyze : ${ANALYZE}  (afterany:${FULL})"

echo
echo "chain: ${SMOKE} -> ${FULL} -> ${ANALYZE}"
echo "out  : outputs/final/cv/summary.{json,csv}"
