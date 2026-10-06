#!/bin/bash
# Multiclass follow-up (docs/preregistration_confirm_mc.md): e0/e1/e2 x 12 fresh seeds.
#   bash scripts/submit_confirm_mc.sh smoke|full
set -euo pipefail
MODE=${1:?usage: submit_confirm_mc.sh smoke|full}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
mkdir -p outputs/logs
export TRACK=multiclass ARMS="e0 e1 e2" SKIP_GATE=1
case "${MODE}" in
  smoke) export CONFIRM_ROOT=outputs/confirm_mc_smoke FINAL_ROOT=outputs/confirm_mc_smoke/final
         export SEEDS="1001 1002" EPOCHS=1 BOOT=50 PERMUTATIONS=2000; TRAIN_ARRAY="0-5" ;;
  full)  export CONFIRM_ROOT=outputs/confirm_mc FINAL_ROOT=outputs/final/confirm_mc
         unset SEEDS EPOCHS || true; TRAIN_ARRAY="0-35%16"
         if [[ -e "${CONFIRM_ROOT}/preregistration_record.txt" ]]; then
           echo "refusing: ${CONFIRM_ROOT} already holds a run" >&2; exit 1; fi ;;
  *) echo "unknown mode" >&2; exit 1 ;;
esac
bash -n slurm/confirm_train.sbatch; bash -n slurm/confirm_mc_analyze.sbatch
test -f data/splits_heldout21/test.txt || /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/make_heldout21_split.py
mkdir -p "${CONFIRM_ROOT}"
{
  echo "mode: ${MODE}"
  echo "submitted_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git -C "${HERE}" rev-parse HEAD)"
  echo "git_dirty_tracked: $(git -C "${HERE}" status --porcelain --untracked-files=no | wc -l)"
  echo "preregistration_sha256: $(sha256sum docs/preregistration_confirm_mc.md | cut -d' ' -f1)"
} > "${CONFIRM_ROOT}/preregistration_record.txt"
cat "${CONFIRM_ROOT}/preregistration_record.txt"
T=$(sbatch --parsable --chdir="${HERE}" --array="${TRAIN_ARRAY}" \
  --partition=L40S,L40S-DSA,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch,H200,H200-PCI,V100-32GB \
  slurm/confirm_train.sbatch)
A=$(sbatch --parsable --chdir="${HERE}" --dependency="afterany:${T}" slurm/confirm_mc_analyze.sbatch)
{ echo "train: ${T}"; echo "analyze: ${A}"; } | tee "${CONFIRM_ROOT}/job_ids.txt"
