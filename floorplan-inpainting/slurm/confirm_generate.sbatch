#!/bin/bash
# Generate C2-wide at each piloted strength on all planned candidates, pick the
# strength by the pre-registered generation-only rule, and build the matched
# e16/e17 multi-room datasets from the pairs accepted in both modes.
#SBATCH --job-name=fp-confirm-generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/confirm-generate-%j.out
#SBATCH --partition=L40S,L40S-DSA,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-confirm-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
WIDE_ROOT="${WIDE_ROOT:-data/augmented_c_wide}"
QA_ROOT="${QA_ROOT:-outputs/augmentation_c_wide}"
CONFIRM_ROOT="${CONFIRM_ROOT:-outputs/confirm}"
MATCHED_ROOT="${MATCHED_ROOT:-data/augmented_c_matched}"
STRENGTHS="${STRENGTHS:-0.50 0.75}"
MIN_PAIRS="${MIN_PAIRS:-30}"
MODEL_CACHE="${MODEL_CACHE:-$PWD/data/cache/huggingface}"
LIMIT_ARGS=()
if [[ -n "${GEN_LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${GEN_LIMIT}"); fi
"$PY" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
tag() { "$PY" -c "import sys; print(f's{int(round(float(sys.argv[1])*100)):03d}')" "$1"; }

for S in ${STRENGTHS}; do
  T=$(tag "$S")
  echo "== generating C2-wide at strength ${S} -> ${WIDE_ROOT}/${T}"
  "$PY" scripts/generate_strategy_c.py --config configs/augmentation_c_wide.yaml \
    --mode diffusion_inpaint --strength "$S" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} --num-workers 2 \
    --model-cache "$MODEL_CACHE" --dataset-root "${WIDE_ROOT}/${T}" --qa-root "${QA_ROOT}/${T}" \
    --retry-errors
done

FORCE_ARGS=()
if [[ -n "${FORCE_STRENGTH:-}" ]]; then FORCE_ARGS=(--force-strength "${FORCE_STRENGTH}"); fi
"$PY" scripts/select_c_wide_strength.py --pilot-root "${WIDE_ROOT}" --strengths ${STRENGTHS} \
  --out "${CONFIRM_ROOT}/strength_selection.json" ${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}
CHOSEN=$("$PY" -c "import json,sys; v=json.load(open(sys.argv[1]))['chosen_strength']; print('' if v is None else v)" \
  "${CONFIRM_ROOT}/strength_selection.json")
if [[ -n "${CHOSEN}" ]]; then
  "$PY" scripts/validate_augmented_dataset.py \
    "${WIDE_ROOT}/$(tag "${CHOSEN}")/diffusion_inpaint/instances_train_aug_c.json"
fi
"$PY" scripts/build_matched_c_datasets.py --selection "${CONFIRM_ROOT}/strength_selection.json" \
  --output-root "${MATCHED_ROOT}" --min-pairs "${MIN_PAIRS}"
STATUS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "${MATCHED_ROOT}/matched_manifest.json")
echo "matched datasets: ${STATUS}"
if [[ "${STATUS}" == built ]]; then
  for D in copy_paste_multi inpaint_wide_multi; do
    "$PY" scripts/validate_augmented_dataset.py "${MATCHED_ROOT}/${D}/instances_train_augmented.json"
  done
fi
