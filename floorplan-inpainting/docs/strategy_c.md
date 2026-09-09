# Strategy C: exemplar-guided rare-room inpainting

## Purpose and hypotheses

Strategy C tests whether inpainting adds value when rare-room semantics are
supplied by a real training exemplar rather than inferred from a text prompt. It
does not replace or rewrite Strategy A, Strategy B, E0-E7, or existing reports.

The controlled hypotheses are:

1. A real rare donor transferred into a frequent-room geometry is more
   label-faithful than prompt-only common-to-rare generation.
2. Diffusion inpainting of seams and uncovered interior improves that transfer
   beyond direct copy-paste and deterministic feathering.
3. The matched class-agnostic track separates generic localization gains from
   rare semantic-classification gains.
4. E11 tests whether inpainting is additive with bounded rare-aware sampling.

## Frozen inputs and isolated outputs

- COCO source: `data/processed/instances_deduplicated.json`
- Images: `../area_segmentation_coco/images`
- Split authority: frozen split files and the Phase 1 split manifest
- Donors and targets: training split only
- New data: `data/augmented_c/<mode>/`; E8 uses the composed
  `data/augmented_c/copy_paste_multi/` view.
- New QA: `outputs/augmentation_c/<mode>/`
- New experiments: `outputs/experiments_c_multi/`
- New report: `outputs/final_c_multi/`

The earlier single-replacement runs under `outputs/experiments_c/` are retained
as immutable historical results. Updated E8-E11 runs use a fresh namespace so
their checkpoints cannot be resumed with incompatible configurations or
optimizer parameter groups.

Existing outputs are never opened for writing. Every input hash, pairing, seed,
affine transform, mask setting, model revision, output hash, and QC result is
recorded. A changed source/configuration requires a new output directory.
Interrupted runs resume by stable sample ID.

## Pair construction

For each primary rare category 11-16:

1. Calculate area, width, height, aspect ratio, and polygon solidity for every
   real training donor.
2. Construct its 5th-95th percentile geometry ranges, expanded by the configured
   tolerance.
3. Screen frequent classes 1-10 for compatible target rooms.
4. Allocate the scarcest rare categories first.
5. Rank targets by nearest-donor log geometry distance plus solidity distance.
6. Use each target annotation at most once across all categories.
7. Limit each rare donor to `max_donor_reuse` uses.
8. Target `min(3 * real_train_count, 20)` total occurrences per rare class.

The default plan has 52 candidates drawn from all 30 primary-rare training
annotations. Exact counts are written to `pairing_diagnostics.json`.

## Image construction

For every donor-target pair, the implementation:

1. Rasterizes both polygons at original resolution.
2. Erodes the target polygon by a size-relative radius to protect its walls.
3. Subtracts every overlapping non-target annotation.
4. Disables dark-stroke protection inside the editable target interior. This is
   intentional: the old frequent-room label/symbols must be erased.
5. Affinely maps the donor bounding box and mask to the target bounding box.
6. Whites the old editable interior, then inserts only pixels belonging to the
   real donor polygon.
7. Leaves the target polygon and all other annotations unchanged, changing only
   the selected target category to the donor category.

### C0: direct copy-paste

The warped donor is inserted directly. This is the lower control for semantic
transplantation and contains no generated pixels. For E8, accepted C0 edits
from the same source image are composed into a plan using the same 5% ±2
percentage-point replacement policy as the multi-room LoRA experiment. Each
converted annotation is tracked and relabeled independently; overlapping edit
regions and any outside-pixel changes are rejected.

### C1: feather blend

The donor boundary is alpha-feathered toward white using an interior distance
transform. It controls for deterministic seam smoothing and uses no diffusion.

### C2: exemplar-seeded diffusion inpainting

C2 starts from the C1 composite. Its inpainting mask contains only a narrow inner
seam around the donor and editable target pixels not covered by the warped donor.
Core donor pixels are outside the diffusion mask. The configured low strength
(`0.30`) further restricts the model to harmonization/filling. The inverse-mapped
result is copied only through the original-resolution inpainting mask.

## Automated QC

A sample is rejected for any of these conditions:

- editable target fraction below the configured minimum;
- donor coverage below 50% of editable target pixels;
- fewer than 10 retained dark donor pixels;
- less than 95% retention of evaluated donor dark strokes;
- any changed pixel outside the editable interior or on the protected boundary;
- blank, uniform, almost-identical, excessively dark, or saturated output;
- invalid dimensions, safety-checker rejection, generation error, corrupt PNG,
  hash mismatch, invalid polygon, or invalid COCO reference.

The semantic claim is intentionally limited to
`real_train_exemplar_preserved`. It does not claim that diffusion independently
understands a German class prompt. QA rows show target plan/crop, real donor,
editable mask, exemplar seed, inpainting mask, generated crop, difference, and
final plan.

## Controlled experiment matrix

| ID | Training data | Sampling | Role |
|---|---|---|---|
| E0 | Existing original result | Uniform | Frozen external baseline |
| E8 | C0 multi-room copy-paste | Uniform | Multi-replacement transplant control |
| E9 | C1 feather blend | Uniform | Deterministic integration control |
| E10 | C2 diffusion inpainting | Uniform | Primary inpainting treatment |
| E11 | C2 diffusion inpainting | Rare-aware | Sampling interaction |

E8-E11 each run multiclass and class-agnostic tracks with the Phase 3 model,
initialization, 25 epochs, 35 optimizer steps per epoch, validation procedure,
and untouched original test set. The primary comparisons are E10-E8 and E10-E9.
E10-E11 isolates sampling; E10-E0 measures the end-to-end baseline change.

## Commands

Inspect pairings without loading a model:

```bash
python scripts/generate_strategy_c.py --plan-only --mode all
```

Generate one preflight sample per mode:

```bash
python scripts/generate_strategy_c.py --mode all --limit 1
```

Generate and validate all datasets (CUDA is required only for C2):

```bash
python scripts/generate_strategy_c.py --mode all --retry-errors
python scripts/build_strategy_c_multiroom_dataset.py \
  --replacement-percentage 5 --percentage-jitter 2
for mode in copy_paste feather_blend diffusion_inpaint; do
  python scripts/validate_augmented_dataset.py \
    "data/augmented_c/$mode/instances_train_aug_c.json"
done
python scripts/validate_augmented_dataset.py \
  data/augmented_c/copy_paste_multi/instances_train_augmented.json
```

Submit generation, eight seed-42 training jobs, and the isolated report:

```bash
bash scripts/submit_strategy_c.sh
```

Regenerate only the report:

```bash
python scripts/report_strategy_c.py
```

## Interpretation constraints

- Synthetic occurrences are not independent real observations.
- Donor pixels are reused and warped; donor reuse must be reported.
- Only target geometry/context is new to the rare label.
- Test support remains extremely small. Single-seed changes are exploratory;
  extra seeds should follow for E0, E8, E9, and E10.
- Overall AP, rare macro AP, and class-agnostic AP answer different questions.
- A class-agnostic gain with zero rare AP means localization improved without
  improving rare semantic recognition.
