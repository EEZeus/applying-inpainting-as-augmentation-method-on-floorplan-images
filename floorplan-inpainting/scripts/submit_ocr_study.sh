#!/bin/bash
# OCR study (docs/preregistration_ocr.md): full dependency chain.
#   ocr_original -> { ocr_diag, text_relabel_generate -> text_relabel_finish
#                     -> train smoke (e18/e19/e20, 1 epoch) -> train full (3 arms x 12 seeds) -> analyze }
# Run from the cluster project dir after scripts/setup_ocr_env.sh:
#   bash scripts/submit_ocr_study.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
mkdir -p outputs/logs outputs/ocr_study
STUDY=outputs/ocr_study
if [[ -e "${STUDY}/preregistration_record.txt" ]]; then
  echo "refusing: ${STUDY} already holds a run" >&2; exit 1
fi
for f in slurm/ocr_original.sbatch slurm/ocr_diag.sbatch slurm/text_relabel_generate.sbatch \
         slurm/text_relabel_finish.sbatch slurm/confirm_train.sbatch slurm/ocr_analyze.sbatch; do bash -n "$f"; done
test -x /netscratch/espandar/miniconda3/envs/ocr/bin/python || { echo "run scripts/setup_ocr_env.sh first" >&2; exit 1; }
test -f data/splits_heldout21/test.txt || /netscratch/espandar/miniconda3/envs/inpainting/bin/python scripts/make_heldout21_split.py
{
  echo "submitted_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit: $(git -C "${HERE}" rev-parse HEAD)"
  echo "git_dirty_tracked: $(git -C "${HERE}" status --porcelain --untracked-files=no | wc -l)"
  echo "preregistration_sha256: $(sha256sum docs/preregistration_ocr.md | cut -d' ' -f1)"
} > "${STUDY}/preregistration_record.txt"
cat "${STUDY}/preregistration_record.txt"
GPU_PARTS=L40S,L40S-DSA,RTX3090,A100-40GB,A100-80GB,A100-RP,A100-PCI,RTXA6000,H100,H100-PCI,H100-RP,batch,H200,H200-PCI,V100-32GB
submit() { sbatch --parsable --chdir="${HERE}" "$@"; }
O=$(submit slurm/ocr_original.sbatch)
D=$(submit --dependency="afterok:${O}" slurm/ocr_diag.sbatch)
G=$(submit --dependency="afterok:${O}" slurm/text_relabel_generate.sbatch)
F=$(submit --dependency="afterok:${G}" slurm/text_relabel_finish.sbatch)
S=$(export TRACK=multiclass ARMS="e18 e19 e20" SEEDS=1001 EPOCHS=1 SKIP_GATE=1 CONFIRM_ROOT=outputs/ocr_smoke
     submit --dependency="afterok:${F}" --array=0-2 --partition="${GPU_PARTS}" slurm/confirm_train.sbatch)
T=$(export TRACK=multiclass ARMS="e18 e19 e20" SKIP_GATE=1 CONFIRM_ROOT="${STUDY}"
     submit --dependency="afterok:${S}" --array=0-35%18 --partition="${GPU_PARTS}" slurm/confirm_train.sbatch)
A=$(export CONFIRM_ROOT="${STUDY}" FINAL_ROOT=outputs/final/ocr_study
     submit --dependency="afterany:${T}" slurm/ocr_analyze.sbatch)
{ echo "ocr_original: ${O}"; echo "ocr_diag: ${D}"; echo "text_generate: ${G}"; echo "text_finish: ${F}"
  echo "train_smoke: ${S}"; echo "train_full: ${T}"; echo "analyze: ${A}"; } | tee "${STUDY}/job_ids.txt"
