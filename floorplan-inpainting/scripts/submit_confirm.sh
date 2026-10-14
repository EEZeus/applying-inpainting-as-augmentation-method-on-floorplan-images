#!/bin/bash
# Confirm chain (docs/preregistration_confirm.md): rescore + generate -> gate -> train -> analyze.
#   bash scripts/submit_confirm.sh smoke|full
set -euo pipefail
MODE=${1:?usage: submit_confirm.sh smoke|full}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
mkdir -p outputs/logs

case "${MODE}" in
  smoke)
    export RESCORE_ROOT=outputs/confirm_smoke/rescore
    export WIDE_ROOT=data/confirm_smoke/augmented_c_wide
    export QA_ROOT=outputs/confirm_smoke/augmentation_c_wide
    export MATCHED_ROOT=data/confirm_smoke/augmented_c_matched
    export CONFIRM_ROOT=outputs/confirm_smoke/confirm
    export FINAL_ROOT=outputs/confirm_smoke/final
    export GEN_LIMIT=3 FORCE_STRENGTH=0.5 MIN_PAIRS=1 FORCE_ALL=1
    export SEEDS="1001 1002" EPOCHS=1
    export BOOT=50 RESCORE_BOOT=20 PERMUTATIONS=2000
    # e0/multiclass/42, e0/agnostic/42, e8/agnostic/42
    RESCORE_ARRAY="0,3,51"
    TRAIN_ARRAY="0-7"
    ;;
  full)
    export RESCORE_ROOT=outputs/rescore
    export WIDE_ROOT=data/augmented_c_wide
    export QA_ROOT=outputs/augmentation_c_wide
    export MATCHED_ROOT=data/augmented_c_matched
    export CONFIRM_ROOT=outputs/confirm
    export FINAL_ROOT=outputs/final/confirm
    unset GEN_LIMIT FORCE_STRENGTH FORCE_ALL EPOCHS SEEDS || true
    export MIN_PAIRS=30
    RESCORE_ARRAY="0-83%24"
    TRAIN_ARRAY="0-47%16"
    if [[ -e "${CONFIRM_ROOT}/decisions.json" ]]; then
      echo "refusing: ${CONFIRM_ROOT} already holds a run; move it away first" >&2; exit 1
    fi
    ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac

for f in slurm/confirm_rescore.sbatch slurm/confirm_generate.sbatch slurm/confirm_gate.sbatch \
         slurm/confirm_train.sbatch slurm/confirm_analyze.sbatch; do
  bash -n "$f"
done
"${PY}" scripts/make_heldout21_split.py
mkdir -p "${CONFIRM_ROOT}"
{
  echo "mode: ${MODE}"
  echo "submitted_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git -C "${HERE}" rev-parse HEAD)"
  echo "git_dirty_tracked: $(git -C "${HERE}" status --porcelain --untracked-files=no | wc -l)"
  echo "preregistration_sha256: $(sha256sum docs/preregistration_confirm.md | cut -d' ' -f1)"
} > "${CONFIRM_ROOT}/preregistration_record.txt"
cat "${CONFIRM_ROOT}/preregistration_record.txt"

R=$(sbatch --parsable --chdir="${HERE}" --array="${RESCORE_ARRAY}" slurm/confirm_rescore.sbatch)
G=$(sbatch --parsable --chdir="${HERE}" slurm/confirm_generate.sbatch)
GATE=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${R}:${G}" slurm/confirm_gate.sbatch)
T=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${GATE}" --array="${TRAIN_ARRAY}" slurm/confirm_train.sbatch)
A=$(sbatch --parsable --chdir="${HERE}" --dependency="afterany:${T}" slurm/confirm_analyze.sbatch)
{
  echo "rescore: ${R}"; echo "generate: ${G}"; echo "gate: ${GATE}"; echo "train: ${T}"; echo "analyze: ${A}"
} | tee "${CONFIRM_ROOT}/job_ids.txt"
echo "logs: ${HERE}/outputs/logs/confirm-*"
