#!/bin/bash
# SAM 3 x augmentation chain: build the 6 datasets (CPU) -> smoke -> 6-task array -> collect.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
SAM3_PY="${SAM3_ENV:-/netscratch/espandar/miniconda3/envs/sam3}/bin/python"

bash -n slurm/sam3_aug.sbatch
bash -n slurm/collect_sam3_results.sbatch

for aug in strategy_c_inpaint lora_b copy_paste; do
  "${SAM3_PY}" scripts/prepare_sam3_aug_dataset.py --aug "${aug}" >/dev/null
  "${SAM3_PY}" scripts/prepare_sam3_aug_dataset.py --aug "${aug}" --class-agnostic >/dev/null
  echo "built data/sam3_${aug} and data/sam3_${aug}_agnostic"
done

SMOKE=$(sbatch --parsable --chdir="${HERE}" --array=0-0 \
  --export=ALL,SAM3_LOG_SUFFIX=_smoke,SAM3_EXTRA_OVERRIDES=trainer.max_epochs=2 \
  slurm/sam3_aug.sbatch)
echo "smoke   : ${SMOKE}  (task 0 = strategy_c_inpaint / multiclass, 2 epochs -> outputs/sam3_strategy_c_inpaint_smoke)"

FULL=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${SMOKE}" \
  --array=0-5%3 slurm/sam3_aug.sbatch)
echo "sam3 aug: ${FULL}  (tasks 0-5, afterok:${SMOKE})"

COLLECT=$(sbatch --parsable --chdir="${HERE}" --dependency="afterany:${FULL}" slurm/collect_sam3_results.sbatch)
echo "collect : ${COLLECT}  (afterany:${FULL})"

echo
echo "chain: ${SMOKE} -> ${FULL} -> ${COLLECT}"
echo "out  : outputs/final/sam3/summary.{json,csv}"
