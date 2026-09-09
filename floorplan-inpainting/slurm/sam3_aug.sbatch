#!/bin/bash
# SAM 3 trained on the best augmentation strategies, both label tracks.
# 3 strategies x {multiclass, agnostic} = 6 tasks. Each: build the SAM3 dataset
# (augmented train + frozen real val/test) -> train 20 epochs -> test.
# The SAM3 baselines peaked at ~33 GB torch.cuda.max_memory_allocated in
# training on A100-80GB; with caching-allocator reservation/fragmentation the
# real footprint is ~40-45 GB, and the augmented sets have more instances per
# image. Stay on the 80 GB tier the baseline used.
#SBATCH --job-name=fp-sam3-aug
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --partition=A100-80GB,A100-RP,H100,H100-PCI,H100-RP,H200
#SBATCH --output=outputs/logs/sam3-aug-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

SAM3_ROOT="${SAM3_ROOT:-/netscratch/espandar/sam3}"
SAM3_ENV="${SAM3_ENV:-/netscratch/espandar/miniconda3/envs/sam3}"
export SAM3_ROOT
export PYTHONPATH="$PWD:$SAM3_ROOT:${PYTHONPATH:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "$SAM3_ROOT/sam3/train/train.py" ]] || { echo "SAM 3 source missing at $SAM3_ROOT" >&2; exit 2; }
[[ -x "$SAM3_ENV/bin/python" ]] || { echo "SAM 3 env missing at $SAM3_ENV" >&2; exit 2; }
PY="$SAM3_ENV/bin/python"

"$PY" -c "import sys, torch, sam3; from packaging.version import Version; assert sys.version_info >= (3, 12); assert Version(torch.__version__.split('+')[0]) >= Version('2.7'); assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

AUGS=(strategy_c_inpaint lora_b copy_paste)
TRACKS=(multiclass agnostic)
TID=${SLURM_ARRAY_TASK_ID}
AUG=${AUGS[$(( TID % 3 ))]}
TRACK=${TRACKS[$(( TID / 3 ))]}

if [[ "$TRACK" == "agnostic" ]]; then AGN=(--class-agnostic); SUFFIX=_agnostic; else AGN=(); SUFFIX=""; fi
# SAM3_LOG_SUFFIX / SAM3_EXTRA_OVERRIDES let the smoke run write to an isolated
# dir with fewer epochs; unset for the real runs.
DS="$PWD/data/sam3_${AUG}${SUFFIX}"
LOG="$PWD/outputs/sam3_${AUG}${SUFFIX}${SAM3_LOG_SUFFIX:-}"
echo "task ${TID}: aug=${AUG} track=${TRACK} dataset=${DS} logdir=${LOG} extra=[${SAM3_EXTRA_OVERRIDES:-}]"

"$PY" scripts/prepare_sam3_aug_dataset.py --aug "$AUG" "${AGN[@]}"

OVR=(paths.dataset_root="$DS" paths.experiment_log_dir="$LOG")
# shellcheck disable=SC2206
[[ -n "${SAM3_EXTRA_OVERRIDES:-}" ]] && OVR+=(${SAM3_EXTRA_OVERRIDES})
"$PY" scripts/run_sam3_baseline.py --mode train --config configs/sam3_floorplan.yaml --overrides "${OVR[@]}"
"$PY" scripts/run_sam3_baseline.py --mode test  --config configs/sam3_floorplan.yaml --overrides "${OVR[@]}"
