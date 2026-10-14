# Cluster layout and workflow

The project lives in one directory on the cluster:

```
/netscratch/espandar/inpainting/floorplan-inpainting/
├── src/ scripts/ slurm/ configs/ docs/ tests/   code (tracked)
├── data/                                        datasets and derived views (ignored)
├── models/                                      trained LoRA adapters (ignored)
└── outputs/                                     results, checkpoints, logs (ignored)
    ├── logs/          Slurm .out files
    ├── experiments*/  Mask R-CNN runs (<track>/<id>/seed_<seed>)
    ├── cv/            rotated held-out CV
    ├── sam3_*/        SAM 3 baselines and SAM 3 on augmented data
    ├── dino/          DINOv2 features
    └── final/         summaries (grid_r1/, cv/, sam3/, dino/, confirm/, ocr_study/, ...)
```

The git checkout root is the parent directory `/netscratch/espandar/inpainting/`, which holds `.git`,
this project and the read-only source dataset `area_segmentation_coco/`.

## Running experiments

Submit from the project directory; the sbatch files `cd "$SLURM_SUBMIT_DIR"`.

```bash
cd /netscratch/espandar/inpainting/floorplan-inpainting
bash scripts/submit_grid_r1.sh      # Mask R-CNN grid   -> outputs/final/grid_r1/
bash scripts/submit_cv.sh           # rotated CV        -> outputs/final/cv/
bash scripts/submit_sam3_aug.sh     # SAM 3 x aug       -> outputs/final/sam3/
bash scripts/submit_dino.sh         # DINOv2 two-stage  -> outputs/final/dino/
```

The submit scripts rebuild any derived dataset they need. The other chains are listed in the
[README](README.md).

## Code sync

Develop locally, commit, then update the cluster checkout in place with `scripts/deploy_cluster.sh`.
It ships local `main` as a git bundle (the cluster has no GitHub credentials) and fast-forwards the
checkout. Once GitHub access works on the cluster, `git pull` in `/netscratch/espandar/inpainting/`
replaces it and `.deploy/` can be removed.

Do not edit code on the cluster; it will conflict with the next sync.
