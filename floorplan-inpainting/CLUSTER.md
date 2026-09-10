# Cluster layout & workflow

Everything for this project lives in **one directory** on the cluster:

```
/netscratch/espandar/inpainting/floorplan-inpainting/
├── src/  scripts/  slurm/  configs/  docs/  tests/   # code (tracked in git)
├── data/                                             # datasets + derived views (git-ignored)
├── models/                                           # trained LoRA / adapters       (git-ignored)
└── outputs/                                          # ALL results, checkpoints, logs (git-ignored)
    ├── logs/                 slurm .out files
    ├── experiments*/         Mask R-CNN grid runs (per track / id / seed)
    ├── cv/                   rotated held-out CV eval (Experiment 1)
    ├── sam3_*/               SAM3 baseline + SAM3 x augmentation runs
    ├── dino/                 DINOv2 two-stage features (Experiment 3)
    └── final/                consolidated summaries: grid_r1/ cv/ sam3/ dino/
```

The git checkout root is the parent (`/netscratch/espandar/inpainting/`, holding
`.git`, `.gitignore`, this project as a subtree, and the read-only source
dataset `area_segmentation_coco/`). Nothing else should live there.

## Running an experiment

Submit from the project directory (the sbatch files `cd "$SLURM_SUBMIT_DIR"`):

```bash
cd /netscratch/espandar/inpainting/floorplan-inpainting
bash scripts/submit_grid_r1.sh      # Mask R-CNN grid  -> outputs/final/grid_r1/
bash scripts/submit_cv.sh           # Experiment 1     -> outputs/final/cv/
bash scripts/submit_sam3_aug.sh     # SAM3 x aug       -> outputs/final/sam3/
bash scripts/submit_dino.sh         # Experiment 3     -> outputs/final/dino/
```

`data/`, `models/`, `outputs/` are never committed; the submit scripts recreate
any derived dataset they need.

## Code sync

Develop locally, commit, then update the cluster checkout **in place**:

- Interim (no GitHub credentials on the cluster): `scripts/deploy_cluster.sh`
  ships local `main` as a git bundle and fast-forwards the checkout.
- Once GitHub access is set up on the cluster: `git pull` in
  `/netscratch/espandar/inpainting/` instead, and `deploy_cluster.sh` /
  `.deploy/` can be dropped.

Never edit code directly on the cluster — it will conflict with the next sync.
