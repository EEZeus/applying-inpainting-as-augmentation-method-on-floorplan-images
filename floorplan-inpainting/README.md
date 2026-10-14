# Floor-plan inpainting augmentation

This project tests whether diffusion inpainting is a useful augmentation for room instance
segmentation on floor plans. The data is a small COCO dataset of German floor plans (56 images after
removing exact duplicates, 1,633 room polygons, 19 categories, six of them rare). The detector is a
torchvision Mask R-CNN (`maskrcnn_resnet50_fpn_v2`), trained in a multiclass and a class-agnostic
("room") track on a fixed project-aware 35/10/11 train/val/test split. Augmentations only ever read
training images. The project compares several generation strategies (A, B, C, LoRA) against plain
and oversampled baselines, and adds a pre-registered confirmatory study, an OCR study, a SAM 3
baseline and a DINOv2 two-stage classifier.

## Repository layout

```
configs/      dataset, model, augmentation and experiment YAML files
  experiments/          E0-E7   (baseline, oversampling, Strategy A/B)
  experiments_c/        E8-E11  (Strategy C)
  experiments_lora/     E12-E15 (LoRA)
  experiments_confirm/  E16-E17 (confirmatory study)
  experiments_ocr/      E18-E20 (OCR study)
  experiments_followup/ E21-E24 (follow-up)
src/          library code (data, augmentation, lora_inpainting, models, training, evaluation, ocr)
scripts/      command-line entry points and Slurm submit scripts (submit_*.sh)
slurm/        sbatch files
docs/         design notes and pre-registrations
tests/        pytest suite
```

`data/`, `models/`, `outputs/` and `runs/` are local artifacts and are not tracked. The source dataset
is expected at `../area_segmentation_coco/images` (set in `configs/dataset.yaml`). See
[`CLUSTER.md`](CLUSTER.md) for the cluster layout and how code is synced.

## Setup

The Slurm scripts use three conda environments under `/netscratch/espandar/miniconda3`:

- `inpainting` (Python 3.10+): Mask R-CNN, inpainting and analysis. Install with
  `python -m pip install -r requirements.txt`.
- `sam3` (Python 3.12, torch >= 2.7): the official [SAM 3](https://github.com/facebookresearch/sam3)
  repository with its `[train]` extra, cloned to `/netscratch/espandar/sam3` (override with
  `SAM3_ROOT` / `SAM3_ENV`). It also needs `setuptools<82`, `einops`, `pycocotools` and `psutil`,
  and access to the gated `facebook/sam3` checkpoint (`hf auth login`).
- `ocr`: EasyOCR and multilingual-e5, created on the login node by `bash scripts/setup_ocr_env.sh`.

```bash
source /netscratch/espandar/miniconda3/etc/profile.d/conda.sh
conda activate inpainting
python scripts/cache_inpainting_model.py   # SD 1.5 inpainting weights -> data/cache/huggingface
```

## Data preparation

```bash
python scripts/prepare_dataset.py                 # audit + deduplication -> data/processed/instances_deduplicated.json
python scripts/create_group_split.py              # project-aware split -> data/splits/{train,val,test}.txt
python scripts/create_class_agnostic_dataset.py   # single "room" category view
python scripts/visualize_dataset.py --split train --limit 20
```

Split files contain original COCO image IDs. `scripts/make_heldout21_split.py` (test = all 21
val + test plans, for evaluating the epoch-25 checkpoint) and `scripts/make_cv_splits.py --folds 5` are run by the submit scripts that
need them.

## Baseline

```bash
python scripts/train_baseline.py --epochs 1        # smoke run
python scripts/train_baseline.py                   # 25 epochs
python scripts/evaluate_baseline.py --checkpoint outputs/checkpoints/best.pth --split test
```

On the cluster use `slurm/train_baseline.sbatch` and `slurm/evaluate_baseline.sbatch` (edit
`TRAIN_EPOCHS`, `CHECKPOINT` and `EVAL_SPLIT` at the top). The class-agnostic baseline runs with
`sbatch slurm/train_evaluate_baseline_agnostic.sbatch` and writes to `outputs/*_agnostic`.

## Augmentation generation

| Strategy | Idea | Command |
|---|---|---|
| A | Inpaint the interior of rare rooms (same label), plus rare-centred crops | `sbatch slurm/run_phase2.sbatch` (A, crops and B) |
| B | Inpaint geometry-matched frequent rooms and relabel them as rare | included in `run_phase2` |
| C | Paste a real rare-room exemplar into a frequent room (C0 copy-paste, C1 feather, C2 diffusion) | `bash scripts/submit_strategy_c.sh` |
| LoRA | Room-crop LoRA on SD 1.5 inpainting, then exemplar-guided conversion | `bash scripts/submit_lora_branch.sh` |

Individual steps, e.g. for a quick check:

```bash
python scripts/inspect_safe_masks.py --limit 30
python scripts/generate_strategy_a.py --limit 1
python scripts/generate_rare_crops.py
python scripts/generate_strategy_b.py --plan-only
python scripts/generate_strategy_c.py --plan-only --mode all
python scripts/generate_lora_strategy_b.py --plan-only
python scripts/validate_augmented_dataset.py data/augmented_a/instances_train_aug_a.json
python scripts/report_augmentation.py
```

Generation is resumable; only pixels inside the safe (eroded, stroke-protected) room mask may change,
and QC rejects samples that change anything outside it. All candidates pass or fail automatically;
there is no manual review. Mask and QC thresholds are in the `configs/augmentation_*.yaml` files.
Datasets are written to `data/augmented_*`, QA sheets and counts to `outputs/augmentation_*`. Design
notes: [`docs/strategy_c.md`](docs/strategy_c.md) and
[`docs/lora_inpainting_branch.md`](docs/lora_inpainting_branch.md).

## Detector experiments

A single run (any config, either track):

```bash
python scripts/run_experiment.py --config configs/experiments/e0_original.yaml \
  --track multiclass --seed 42 --epochs 1
```

Every run uses 35 optimizer steps per epoch and can be continued with `--resume`. Outputs go to
`<output_root>/<track>/<id>/seed_<seed>/` (checkpoints, history, test metrics, class exposure).
Rare-aware sampling is implemented in `src/augmentation/rare_sampling.py`.

| Experiments | Submit | Results |
|---|---|---|
| E0-E7, seed 42 (+ `--with-extra-seeds`) | `bash scripts/submit_phase3.sh` | `outputs/final/` |
| E8-E11 | `bash scripts/submit_strategy_c.sh` | `outputs/final_c_multi/` |
| E12-E13 | `bash scripts/submit_lora_experiments.sh` | `outputs/final_lora/` |
| E14-E15 | `bash scripts/submit_lora_multiroom_experiments.sh 5 2` | `outputs/final_lora_multi/` |
| Round-1 grid: E0-E13, seeds 123 and 2026 | `bash scripts/submit_grid_r1.sh` | `outputs/final/grid_r1/` |
| Rotated held-out CV of six grid arms | `bash scripts/submit_cv.sh` | `outputs/final/cv/` |

## Pre-registered studies

Each submit script records the commit and the SHA-256 of its pre-registration in
`<study root>/preregistration_record.txt` and refuses to overwrite an existing run. Each chain runs
a one-epoch smoke stage (or a `smoke` mode) before the full array.

| Study | Pre-registration | Submit | Results |
|---|---|---|---|
| Confirmatory (E0, E8, E16, E17, agnostic, 12 seeds) | `docs/preregistration_confirm.md` | `bash scripts/submit_confirm.sh smoke\|full` | `outputs/final/confirm/` |
| Multiclass follow-up (E0, E1, E2) | `docs/preregistration_confirm_mc.md` | `bash scripts/submit_confirm_mc.sh smoke\|full` | `outputs/final/confirm_mc/` |
| OCR study (E18-E20) | `docs/preregistration_ocr.md` | `bash scripts/submit_ocr_study.sh` | `outputs/final/ocr_study/` |
| Follow-up (E21-E24, exploratory) | `docs/preregistration_followup.md` | `bash scripts/submit_followup.sh` | `outputs/final/followup/` |

The OCR study needs the `ocr` environment; the follow-up needs the OCR study diagnostics.

## SAM 3 and DINOv2

```bash
sbatch slurm/train_evaluate_sam3.sbatch            # multiclass baseline -> outputs/sam3_baseline/
sbatch slurm/train_evaluate_sam3_agnostic.sbatch   # agnostic baseline   -> outputs/sam3_baseline_agnostic/
bash scripts/submit_sam3_aug.sh                    # SAM 3 on augmented data -> outputs/final/sam3/
bash scripts/submit_dino.sh                        # DINOv2 two-stage   -> outputs/final/dino/
```

SAM 3 uses text prompts from the category names (`configs/sam3_floorplan*.yaml`). The DINOv2
experiment types rooms with linear and kNN heads on frozen `dinov2-base` crop features and needs the
agnostic SAM 3 test predictions.

## Tests

```bash
pytest -q
pytest -q -m "not slow"   # skip model-level tests
```

The tests use a fake inpainting backend in temporary directories and need no GPU.
