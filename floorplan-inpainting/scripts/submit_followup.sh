#!/bin/bash
# Follow-up chain (docs/preregistration_followup.md):
#   balance_generate[b20,b50,b100] -> { balance_ocr -> followup_diag,
#                                       train smoke (e21-e24, 1 epoch) -> train full (4 arms x 12 seeds) } -> analyze
#   bash scripts/submit_followup.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
mkdir -p outputs/logs outputs/followup
STUDY=outputs/followup
if [[ -e "${STUDY}/preregistration_record.txt" ]]; then echo "refusing: ${STUDY} already holds a run" >&2; exit 1; fi
for f in slurm/balance_generate.sbatch slurm/balance_ocr.sbatch slurm/followup_diag.sbatch \
         slurm/confirm_train.sbatch slurm/followup_analyze.sbatch; do bash -n "$f"; done
test -f outputs/final/ocr_study/diagnostics/diagnostics.json || { echo "OCR study diagnostics missing" >&2; exit 1; }
{
  echo "submitted_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git -C "${HERE}" rev-parse HEAD)"
  echo "git_dirty_tracked: $(git -C "${HERE}" status --porcelain --untracked-files=no | wc -l)"
  echo "preregistration_sha256: $(sha256sum docs/preregistration_followup.md | cut -d' ' -f1)"
} > "${STUDY}/preregistration_record.txt"
cat "${STUDY}/preregistration_record.txt"
GPU_PARTS=L40S,L40S-DSA,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch,H200,H200-PCI,V100-32GB
submit() { sbatch --parsable --chdir="${HERE}" "$@"; }
G=$(submit slurm/balance_generate.sbatch)
O=$(submit --dependency="afterok:${G}" slurm/balance_ocr.sbatch)
D=$(submit --dependency="afterok:${O}" slurm/followup_diag.sbatch)
S=$(export TRACK=multiclass ARMS="e21 e22 e23 e24" SEEDS=1001 EPOCHS=1 SKIP_GATE=1 CONFIRM_ROOT=outputs/followup_smoke
     submit --dependency="afterok:${G}" --array=0-3 --partition="${GPU_PARTS}" slurm/confirm_train.sbatch)
T=$(export TRACK=multiclass ARMS="e21 e22 e23 e24" SKIP_GATE=1 CONFIRM_ROOT="${STUDY}"
     submit --dependency="afterok:${S}" --array=0-47%24 --partition="${GPU_PARTS}" slurm/confirm_train.sbatch)
A=$(export CONFIRM_ROOT="${STUDY}" FINAL_ROOT=outputs/final/followup
     submit --dependency="afterany:${T}:${D}" slurm/followup_analyze.sbatch)
{ echo "balance_generate: ${G}"; echo "balance_ocr: ${O}"; echo "followup_diag: ${D}"; echo "train_smoke: ${S}"
  echo "train_full: ${T}"; echo "analyze: ${A}"; } | tee "${STUDY}/job_ids.txt"
