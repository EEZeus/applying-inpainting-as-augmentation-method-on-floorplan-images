#!/bin/bash
# DINOv2 two-stage chain: cache dinov2-base on the login node -> smoke extraction -> full
# extraction -> analyze.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "${HERE}"
PY=/netscratch/espandar/miniconda3/envs/inpainting/bin/python
export HF_HUB_CACHE="${HF_HUB_CACHE:-/netscratch/espandar/huggingface_cache/hub}"

bash -n slurm/dino_extract.sbatch
bash -n slurm/dino_analyze.sbatch

echo "pre-caching facebook/dinov2-base into ${HF_HUB_CACHE} ..."
"${PY}" - <<'PYEOF'
import os
from transformers import AutoModel, AutoImageProcessor
m = AutoModel.from_pretrained("facebook/dinov2-base")
p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
import torch
x = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = m(pixel_values=x)
hs = out.last_hidden_state
assert hs.ndim == 3 and hs.shape[-1] == 768, hs.shape
print("dinov2-base OK  last_hidden_state", tuple(hs.shape), " cache:", os.environ["HF_HUB_CACHE"])
PYEOF
ls -d "${HF_HUB_CACHE}/models--facebook--dinov2-base" >/dev/null

SMOKE=$(sbatch --parsable --chdir="${HERE}" --export=ALL,DINO_SMOKE=1 slurm/dino_extract.sbatch)
echo "smoke   : ${SMOKE}"
EXTRACT=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${SMOKE}" slurm/dino_extract.sbatch)
echo "extract : ${EXTRACT}  (afterok:${SMOKE})"
ANALYZE=$(sbatch --parsable --chdir="${HERE}" --dependency="afterok:${EXTRACT}" slurm/dino_analyze.sbatch)
echo "analyze : ${ANALYZE}  (afterok:${EXTRACT})"

echo
echo "chain: ${SMOKE} -> ${EXTRACT} -> ${ANALYZE}"
echo "out  : outputs/final/dino/summary.{json,csv} + report.md"
