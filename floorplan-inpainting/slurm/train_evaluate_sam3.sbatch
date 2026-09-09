#!/bin/bash
#SBATCH --job-name=floorplan-sam3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=A100-80GB,A100-RP,H100,H100-PCI,H100-RP
#SBATCH --output=outputs/logs/sam3-baseline-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

SAM3_ROOT="${SAM3_ROOT:-/netscratch/espandar/sam3}"
SAM3_ENV="${SAM3_ENV:-/netscratch/espandar/miniconda3/envs/sam3}"
export SAM3_ROOT
export PYTHONPATH="$PWD:$SAM3_ROOT:${PYTHONPATH:-}"
# Keep credentials in Hugging Face's default user location while placing the
# large gated model download on netscratch.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"
export TORCH_HOME="${TORCH_HOME:-/netscratch/espandar/torch_cache}"
export OMP_NUM_THREADS=2

if [[ ! -f "$SAM3_ROOT/sam3/train/train.py" ]]; then
  echo "SAM 3 source not found at $SAM3_ROOT. Clone https://github.com/facebookresearch/sam3 there." >&2
  exit 2
fi
if [[ ! -x "$SAM3_ENV/bin/python" ]]; then
  echo "SAM 3 environment not found at $SAM3_ENV. Install the official package with its [train] extra." >&2
  exit 2
fi

"$SAM3_ENV/bin/python" -c \
  "import sys, torch, sam3; from packaging.version import Version; assert sys.version_info >= (3, 12); assert Version(torch.__version__.split('+')[0]) >= Version('2.7'); assert torch.cuda.is_available(); print(sys.version); print(torch.__version__, torch.cuda.get_device_name(0))"

"$SAM3_ENV/bin/python" scripts/prepare_sam3_dataset.py
"$SAM3_ENV/bin/python" scripts/run_sam3_baseline.py --mode train
"$SAM3_ENV/bin/python" scripts/run_sam3_baseline.py --mode test
