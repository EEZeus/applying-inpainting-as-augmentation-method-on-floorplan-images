# Strategy C: exemplar-guided rare-room inpainting

Strategy C asks whether inpainting helps when the rare-room content comes from a real training
exemplar instead of a text prompt. It leaves Strategy A, Strategy B and E0-E7 unchanged.

Hypotheses:

1. Transferring a real rare donor into a frequent-room geometry is more label-faithful than
   prompt-only common-to-rare generation.
2. Diffusion inpainting of seams and uncovered interior improves on direct copy-paste and feathering.
3. The class-agnostic track separates localization gains from rare-class recognition gains.
4. E11 tests whether inpainting adds to rare-aware sampling.

## Inputs and outputs

- COCO source: `data/processed/instances_deduplicated.json`; images: `../area_segmentation_coco/images`
- Donors and targets come from the training split only.
- Datasets: `data/augmented_c/<mode>/`; E8 uses the composed `data/augmented_c/copy_paste_multi/`.
- QA: `outputs/augmentation_c/<mode>/`
- Experiments: `outputs/experiments_c_multi/`; report: `outputs/final_c_multi/`

The earlier single-replacement runs in `outputs/experiments_c/` are kept as they are; the multi-room
E8-E11 runs use a new directory so they cannot resume from incompatible checkpoints.

Input hashes, pairings, seeds, affine transforms, mask settings, model revision, output hashes and QC
results are recorded per sample. Interrupted runs resume by stable sample ID; a changed source or
config needs a new output directory.

## Pairing

For each primary rare category 11-16:

1. Compute area, width, height, aspect ratio and solidity of every real training donor.
2. Take the 5th-95th percentile ranges, widened by the configured tolerance.
3. Screen frequent classes 1-10 for compatible target rooms.
4. Allocate the scarcest rare categories first.
5. Rank targets by log-geometry distance to the nearest donor plus solidity distance.
6. Use each target annotation at most once and each donor at most `max_donor_reuse` times.
7. Aim for `min(3 * real_train_count, 20)` occurrences per rare class.

The default plan has 52 candidates from all 30 primary-rare training annotations. Exact counts are
in `pairing_diagnostics.json`.

## Image construction

For each donor-target pair:

1. Rasterize both polygons at original resolution.
2. Erode the target polygon by a size-relative radius to protect its walls.
3. Subtract every overlapping non-target annotation.
4. Disable dark-stroke protection inside the target, so the old room label and symbols get erased.
5. Map the donor bounding box and mask affinely onto the target bounding box.
6. Whiten the editable interior and insert only pixels from the donor polygon.
7. Keep the target polygon and all other annotations; only the target category changes.

- C0 (copy-paste): the warped donor is inserted directly; no generated pixels. For E8, accepted C0
  edits from the same plan are composed with the same 5% +-2 percentage-point replacement policy as
  the multi-room LoRA experiment. Overlapping edits and any change outside the edit regions are
  rejected.
- C1 (feather blend): the donor boundary is alpha-feathered towards white using a distance
  transform. No diffusion.
- C2 (diffusion inpainting): starts from the C1 composite. The inpainting mask covers only a narrow
  inner seam and the target pixels the donor does not cover; the donor core is outside it. Strength
  0.30 limits the model to harmonization. The result is copied back only through the inpainting mask.

## QC

A sample is rejected if any of these hold:

- editable target fraction below the configured minimum;
- donor covers less than 50% of the editable target;
- fewer than 10 dark donor pixels, or less than 95% of evaluated donor strokes retained;
- any changed pixel outside the editable interior or on the protected boundary;
- blank, uniform, nearly unchanged, too dark or saturated output;
- invalid size, safety-checker rejection, generation error, corrupt PNG, hash mismatch, invalid
  polygon or invalid COCO reference.

The semantic claim is limited to `real_train_exemplar_preserved`; it does not assume that the
diffusion model understands German room names. QA rows show target plan and crop, donor, editable
mask, exemplar seed, inpainting mask, generated crop, difference and final plan.

## Experiments

| ID | Training data | Sampling | Role |
|---|---|---|---|
| E0 | original | uniform | existing baseline |
| E8 | C0 multi-room copy-paste | uniform | transplant control |
| E9 | C1 feather blend | uniform | deterministic blending control |
| E10 | C2 diffusion inpainting | uniform | main treatment |
| E11 | C2 diffusion inpainting | rare-aware | sampling interaction |

E8-E11 run both tracks with the Phase 3 model, 25 epochs, 35 optimizer steps per epoch and the
original test set. Main comparisons: E10 vs E8 and E10 vs E9; E11 vs E10 isolates sampling and
E10 vs E0 gives the overall change.

## Commands

```bash
python scripts/generate_strategy_c.py --plan-only --mode all   # pairings only, no model
python scripts/generate_strategy_c.py --mode all --limit 1     # one preflight sample per mode
python scripts/generate_strategy_c.py --mode all --retry-errors   # CUDA needed for C2
python scripts/build_strategy_c_multiroom_dataset.py \
  --replacement-percentage 5 --percentage-jitter 2
for mode in copy_paste feather_blend diffusion_inpaint; do
  python scripts/validate_augmented_dataset.py \
    "data/augmented_c/$mode/instances_train_aug_c.json"
done
python scripts/validate_augmented_dataset.py \
  data/augmented_c/copy_paste_multi/instances_train_augmented.json

bash scripts/submit_strategy_c.sh       # generation, E8-E11 (seed 42, both tracks), report
python scripts/report_strategy_c.py     # report only
```

## Caveats

- Synthetic occurrences reuse warped donor pixels and are not independent observations; report
  donor reuse.
- Only the target geometry and context are new to the rare label.
- Test support is very small, so single-seed differences are exploratory.
- A class-agnostic gain with zero rare AP means localization improved, not rare-class recognition.
