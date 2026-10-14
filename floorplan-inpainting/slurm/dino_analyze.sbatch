#!/bin/bash
# DINOv2 stage 2: train the linear and kNN heads and evaluate. CPU only.
#SBATCH --job-name=fp-dino-analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/dino-analyze-%j.out
#SBATCH --partition=RTX3090,L40S,L40S-DSA,V100-32GB,A100-40GB,A100-PCI,RTXA6000,batch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
exec /netscratch/espandar/miniconda3/envs/inpainting/bin/python \
  scripts/dino_train_eval.py --dir outputs/dino --out outputs/final/dino
