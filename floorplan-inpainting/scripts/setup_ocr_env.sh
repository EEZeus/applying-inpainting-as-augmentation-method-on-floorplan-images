#!/bin/bash
# One-time setup on the cluster LOGIN node (has internet): a separate `ocr`
# conda env (EasyOCR + transformers, torch pinned to the inpainting env's
# 2.1.2+cu121) and offline copies of the EasyOCR and multilingual-e5 weights.
#   bash scripts/setup_ocr_env.sh
set -euo pipefail
CONDA=/netscratch/espandar/miniconda3
ENV=${CONDA}/envs/ocr
export EASYOCR_MODEL_DIR=${EASYOCR_MODEL_DIR:-/netscratch/espandar/easyocr_models}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}
if [[ ! -x ${ENV}/bin/python ]]; then
  "${CONDA}/bin/conda" create -y -p "${ENV}" python=3.10
fi
PIP="${ENV}/bin/python -m pip"
$PIP install --quiet torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
$PIP install --quiet "numpy<2" easyocr==1.7.2 transformers==4.44.2 sentencepiece pycocotools \
  "opencv-python-headless>=4.8,<4.12" shapely==2.0.6 PyYAML matplotlib Pillow tqdm pandas
mkdir -p "${EASYOCR_MODEL_DIR}"
"${ENV}/bin/python" - <<'PY'
import os
import easyocr
easyocr.Reader(["de", "en"], gpu=False, model_storage_directory=os.environ["EASYOCR_MODEL_DIR"],
               download_enabled=True, verbose=False)
from huggingface_hub import snapshot_download
print(snapshot_download("intfloat/multilingual-e5-small", cache_dir=os.environ["HF_HUB_CACHE"], token=False))
import torch, transformers
print("ocr env ready:", torch.__version__, torch.version.cuda, easyocr.__version__, transformers.__version__)
PY
