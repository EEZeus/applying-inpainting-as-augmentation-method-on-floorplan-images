# LoRA inpainting branch: implementation and protocol

This is an additional, isolated augmentation branch for the project **Applying
Inpainting as an augmentation method for floor plans**. It does not alter the
canonical COCO file, frozen train/validation/test split, Strategy A/B/C outputs,
or E0–E11 results. All diffusion inputs are restricted by `load_inputs()` to
the frozen training image IDs; model inference is local and jobs set Hugging
Face offline mode after using the existing local cache.

## 1. Training data construction

`scripts/build_lora_training_crops.py` visits every usable training annotation.
It crops the room bbox with one bbox-width/height of context, clamps to the
image, white-pads to a square, and resizes to 512×512. The binary inpainting
mask is the polygon interior after a 2% relative erosion (clamped to 2–12 source
pixels), subtraction of overlapping room annotations, and subtraction of
dilated dark strokes detected inside the room. Consequently the room boundary,
walls, labels, and symbols are protected where detectable.

Every class contributes its real base crop. Classes 11–16 receive 12
deterministic classical variants per usable crop and optional classes 17–18
receive six. Variants combine bounded brightness/contrast, optional grayscale,
mild blur, JPEG/scan noise, one-pixel line-thickness variation, and a mild
geometry-aligned affine transform. Image and mask receive the same affine.
Every record keeps `source_annotation_id` and `geometry_identity`, and explicitly
sets `independent_geometry=false`.

The manifest and counts are written to `data/lora_training/`. A fixed validation
case for each available rare target uses the nearest common-room geometry to the
median real target geometry, rather than presenting an augmented rare crop as a
new geometry.

## 2. Shared class-conditioned LoRA

`scripts/train_floorplan_lora.py` freezes the SD 1.5 inpainting VAE, CLIP text
encoder, and base UNet, then trains rank-8 LoRA adapters on UNet attention
projections. The prompt is `architectural technical floor plan,
{category_name}, 2D line drawing`. The standard nine-channel inpainting input
is noisy latent + resized mask + masked-image latent. Rare records are sampled
three times as strongly (optional rare records twice) while common real crops
remain in the stream. The default is mixed precision, effective batch four,
5,000 optimizer steps, and checkpoints every 500 steps.

Outputs in `models/floorplan_lora/` include final safetensors LoRA weights,
checkpoint adapters and metadata, exact training config, loss CSV/curve, and a
training summary. This is one shared adapter, not one model per category and
not a full Stable Diffusion fine-tune.

## 3. Generator validation

`scripts/validate_lora_generator.py` runs identical prompts, masks, and fixed
seeds through pretrained SD and the LoRA. Generated pixels are composited only
inside the safe mask. It writes side-by-side source/mask/base/LoRA contact sheets,
generation metadata, hashes, exact outside-change checks, and
`outputs/lora_validation/automatic_comparison.csv`. The contact sheets remain
diagnostic report artifacts; they do not gate or pause the pipeline.

Loss reduction is not evidence that the model learned room semantics. Protecting
dark labels and symbols is structurally conservative but can remove some of the
strongest category cues from the editable region. Therefore downstream detector
ablation is essential, and semantic correctness is reported as unverified.

## 4. Common-to-rare conversion

`scripts/generate_lora_strategy_b.py` uses only common classes 1–10 as source
rooms and targets classes 11–16. For each rare class it estimates real-training
area, aspect ratio, width, height, and solidity ranges (p05–p95 expanded by the
configured tolerance), rejects incompatible source polygons, never reuses a
source annotation across targets, and caps the target total at
`min(3 × real_train_count, 20)`. The original destination polygon is retained.

Only the safe interior is sent to the LoRA and only that mask is composited into
the full plan. Pixel QC requires both `outside_change_ratio=0` and
`protected_boundary_change_ratio=0`. The exterior boundary, walls, doors,
neighbors, and every pixel outside the safe mask remain exact. In the candidate
COCO, only the selected annotation category changes; polygon, bbox, area, and
all other annotations are copied unchanged.

## 5. Fully automatic acceptance

There is no human acceptance step. Geometry screening happens before generation;
after generation, exact structural invariance and configured pixel-quality
thresholds decide acceptance. `outputs/augmentation_lora_b/decision_audit.csv`
is generated from immutable metadata and cannot be edited to admit a failed
sample. Source image hashes, generated PNG hashes, and the exact LoRA weight
hash are rechecked before `instances_train_augmented.json` is created.

The contact sheets are optional diagnostics only. `semantic_correctness` remains
`unverified`, because deterministic raster checks cannot prove that an image
visually represents the requested room category. E12/E13 measure whether these
automatically admitted samples help the detector.

## 6. Scientific accounting

Report three quantities separately:

1. **Real rare geometries**: unique real training annotation IDs.
2. **Classical rare LoRA crop variants**: repeated appearances tied to those
   same geometry identities; they are not independent rooms.
3. **Automatically accepted synthetic rare geometries**: common source polygons
   relabelled after geometry, pixel-quality, and exact structural checks. These
   polygons are new to the target class but are not newly invented geometry.

Do not merge these counts into a claim of independent rare-room samples.

## Reproduction

```bash
# Runs crops -> LoRA -> base/LoRA validation -> automatic conversion integration.
bash scripts/submit_lora_branch.sh

# CPU-only geometry preview (does not load diffusion):
python scripts/generate_lora_strategy_b.py --plan-only
```

If 5,000 steps cannot finish inside the project window, reduce
`training.max_train_steps` to 3,000 while retaining fixed validation and all
checkpoints. The unannotated Feuerwehr plans are not required for this first
experiment. They would need a separate self-supervised/domain-adaptation design;
mixing them into the supervised crop manifest without room masks or reliable
prompts would not meet this protocol.

## Parameterized multi-room flavor

The initial prompt-only conversion was invalidated after QA showed retained
common-room labels such as `Büro`. Its E12/E13 results are negative evidence,
not valid augmentation results. Semantic v2 masks compact interior labels and
symbols while preserving boundaries/long structural lines, and final generation
uses a genuine training-split rare-room exemplar whose dark semantic strokes
must survive automatic QC. This prevents category-only common-to-rare relabeling.

The original LoRA dataset deliberately changes one room per synthetic plan.
The additional E14/E15 flavor can change several common rooms in the same copy:

```bash
# Nominal 5% of common rooms per plan, with a random ±2 percentage-point offset.
bash scripts/submit_lora_multiroom_experiments.sh 5 2
```

For every source plan a seeded uniform jitter is applied to the requested
percentage, followed by stochastic rounding to an integer number of rooms.
Geometry matching and the conservative per-rare-class cap still take priority,
so the manifest reports both requested and achieved percentages. Failed pixel
QC also lowers the achieved percentage; unsafe results are never admitted just
to hit a quota.

The planner asserts that every source category belongs to common IDs 1–10 and
every destination category belongs to rare IDs 11–16. A source room can never
be mapped to another common category. Each safe mask is composed independently,
and the union is checked for exact outside and boundary invariance. Parameterized
datasets are stored under `data/augmented_lora_multi/pct_<P>_jitter_<J>/` and
E14/E15 outputs use the corresponding isolated directory under
`outputs/experiments_lora_multi/`.
