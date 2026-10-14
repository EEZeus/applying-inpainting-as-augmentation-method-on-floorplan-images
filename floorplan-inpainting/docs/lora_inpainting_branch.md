# LoRA inpainting branch

A separate augmentation branch that fine-tunes a LoRA adapter for SD 1.5 inpainting on floor-plan
room crops and uses it for common-to-rare conversion. It does not change the COCO file, the frozen
split, Strategy A/B/C outputs or E0-E11. `load_inputs()` restricts all diffusion inputs to training
images, and jobs run Hugging Face in offline mode from the local cache.

## 1. Training crops

`scripts/build_lora_training_crops.py` crops every usable training annotation with one bbox
width/height of context, clamps to the image, pads to a white square and resizes to 512x512. The
inpainting mask is the room interior after a 2% erosion (2-12 source pixels), minus overlapping
rooms and dilated dark strokes, so walls, labels and symbols are protected where detected.

Every class contributes its real crop. Classes 11-16 get 12 classical variants per crop and classes
17-18 get six (brightness/contrast, optional grayscale, mild blur, JPEG/scan noise, one-pixel line
thickness change, mild affine applied to image and mask). Each record keeps `source_annotation_id`
and `geometry_identity` and sets `independent_geometry=false`.

The manifest and counts go to `data/lora_training_semantic_v2/`. The fixed validation case for each
rare class uses the common-room geometry closest to the median real geometry of that class.

## 2. LoRA training

`scripts/train_floorplan_lora.py` freezes the VAE, CLIP text encoder and UNet and trains rank-8 LoRA
adapters on the UNet attention projections with the prompt `architectural technical floor plan,
{category_name}, 2D line drawing`. The UNet input is the usual nine-channel inpainting input (noisy
latent, mask, masked-image latent). Rare records are sampled three times as often (classes 17-18
twice). Defaults: fp16 mixed precision, effective batch 4, 5,000 steps, checkpoints every 500 steps.

`models/floorplan_lora_semantic_v2/` holds the final safetensors weights, checkpoints, the training
config, the loss curve and a summary. It is one shared adapter for all classes.

## 3. Generator validation

`scripts/validate_lora_generator.py` runs the same prompts, masks and seeds through pretrained SD and
the LoRA, composites only inside the safe mask, and writes contact sheets, metadata, outside-change
checks and `outputs/lora_validation_semantic_v2/automatic_comparison.csv`. The sheets are diagnostics
and do not gate the pipeline.

Lower loss does not show that the model learned room semantics, and protecting dark strokes removes
some of the strongest category cues from the editable region. Semantic correctness is therefore
reported as unverified; the detector experiments are the actual test.

## 4. Common-to-rare conversion

`scripts/generate_lora_strategy_b.py` takes source rooms from classes 1-10 and targets classes
11-16. Per rare class it computes real training ranges of area, aspect ratio, width, height and
solidity (p05-p95, widened by the configured tolerance), rejects incompatible sources, never reuses a
source annotation, and caps the class total at `min(3 * real_train_count, 20)`.

Only the safe interior is generated and composited. QC requires `outside_change_ratio=0` and
`protected_boundary_change_ratio=0`, plus the configured pixel-quality thresholds. In the output COCO
only the selected annotation's category changes.

Acceptance is automatic. `outputs/augmentation_lora_semantic_v2/decision_audit.csv` is generated
from the metadata and cannot be edited to admit a failed sample. Source hashes, output hashes and the
LoRA weight hash are rechecked before `instances_train_augmented.json` is written.

## 5. Counting

Report three numbers separately:

1. real rare geometries (unique training annotation IDs);
2. classical LoRA crop variants, which reuse those geometries;
3. accepted synthetic rare rooms: common polygons relabelled after QC. The polygon is new to the
   rare class but is not new geometry.

## Multi-room semantic v2 (E14/E15)

The first prompt-only conversion (E12/E13) was invalidated: QA showed retained common-room labels
such as `Büro`, so its results are negative evidence only. Semantic v2 masks compact interior labels
and symbols while keeping boundaries and long structural lines, and `scripts/generate_lora_exemplar_v2.py`
seeds each conversion with a real training rare-room exemplar whose dark strokes must survive QC.

E14/E15 convert several common rooms in the same plan. For each source plan a seeded uniform jitter
is added to the requested percentage and the result is stochastically rounded to a number of rooms.
Geometry matching, the per-class cap and pixel QC take priority, so the manifest reports requested
and achieved percentages. The planner asserts that sources are in classes 1-10 and targets in 11-16.
Each safe mask is composed independently and the union is checked for outside and boundary changes.

Datasets: `data/augmented_lora_multi_semantic_v2/pct_<P>_jitter_<J>/`; runs:
`outputs/experiments_lora_multi_semantic_v2/pct_<P>_jitter_<J>/`.

## Reproduction

```bash
bash scripts/submit_lora_branch.sh                    # crops -> LoRA -> validation -> exemplar conversion -> E14/E15 -> report
bash scripts/submit_lora_multiroom_experiments.sh 5 2 # rebuild and rerun E14/E15 for 5% +-2 points
python scripts/generate_lora_strategy_b.py --plan-only   # geometry preview, CPU only
```

If 5,000 steps do not fit the time budget, `training.max_train_steps` can be reduced to 3,000. The
unannotated Feuerwehr plans are not used; they would need a separate self-supervised design.
