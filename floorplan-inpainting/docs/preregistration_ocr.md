# Pre-registration: OCR study (text-aware detection and label-rewrite inpainting)

Written 2026-10-13, before any OCR study detector run. The SHA-256 of this file
is recorded in `outputs/ocr_study/preregistration_record.txt` at submission.

## Motivation

Room classes in these plans differ mainly in the printed room name ("Büro",
"Spindraum H", "Putzr."), not in a visual pattern. Mask R-CNN sees the plans
at ≤1600 px, where labels are a few pixels high, and every earlier
augmentation (A, B, C, LoRA) kept or moved the printed label instead of making
it agree with the new class. Frozen DINOv2 crop features classify 45% of
ground-truth test rooms correctly and no rare room.

## Interventions

1. **OCR features (no labels used).** EasyOCR (German + English) reads every
   plan at native resolution in 1600 px tiles (256 px overlap). Tokens with at
   least two letters after removing digit runs are label tokens. Each label
   token is represented by its multilingual-e5-small embedding, a 256-d hashed
   character-trigram vector and its cosine similarity to the 19 class names.
2. **Text-aware Mask R-CNN (e19, e20).** For every RoI, label tokens whose box
   centre lies inside the proposal are pooled (confidence-weighted mean,
   max class-name similarity, has-text flag, log count). A 2-layer MLP (256
   hidden, last layer zero-initialised) adds the result to the box feature that
   feeds the classifier only. Token dropout 0.15 and whole-plan dropout 0.15 in
   training. All other settings are those of e0.
3. **Label-rewrite inpainting (e18, e20).** Strategy B geometry screening picks
   frequent rooms (classes 1-10) for the six primary rare classes, with budget
   min(3n, 20) - n per class. For each, the OCR label tokens inside the room
   are erased with SD-1.5 inpainting restricted to the token boxes (Telea
   inpainting is the automatic fallback if the diffusion fill fails the erase
   check). A rare-class name drawn from the OCR vocabulary of *training* rooms
   of that class is rendered at the erased label's size and colour, and the
   room is relabelled. Two edits per room; variant v of a plan applies all
   v-th edits. A room is admitted only if a second OCR pass reads the new name
   (similarity ≥ 0.75) and no longer reads the old one. Rejected rooms stay
   original. Arms e18/e20 use the `paired_variant` sampler (e0's plan order;
   each plan with variants is swapped for one with probability 0.5).

Everything above is fixed in the committed code and configs
(`configs/augmentation_text.yaml`, `configs/experiments_ocr/`). Nothing is tuned
on validation or held-out plans.

## Design

- Track: multiclass. Seeds 1001-1012. Recipe unchanged (25 epochs × 35 steps,
  batch 1, `configs/baseline.yaml`).
- Arms: e0 (plain, original plans: the 12 existing runs of the multiclass
  follow-up, `outputs/confirm_mc`, reused unchanged), e18, e19, e20.
- Endpoint: multiclass mask AP@[.50:.95] of `last.pth` on the 21 held-out
  plans (`data/splits_heldout21`). `best.pth` is not used.

## Hypotheses (one-sided)

- **H6** e19 > e0: OCR features improve multiclass mask AP.
- **H7** e20 > e19: label-rewrite inpainting adds to the text-aware model.
- **H8** e18 > e0: label-rewrite inpainting helps the plain model.

## Test and decision rule

Arms that share a seed share the per-epoch plan order, so seeds are paired.
Each hypothesis: one-sided paired t-test over the 12 seeds. Holm correction
over {H6, H7, H8} at α = 0.05. A hypothesis is **supported** only if Holm
rejects **and** the 95% percentile CI of the mean difference from a paired
bootstrap (seed pairs × the 21 plans, 2000 resamples) lies above 0.
Smallest effect of interest: +0.01 AP.

## Descriptive only

Rare-class macro mask AP and its paired difference; AP50; AP on the 11 original
test plans; per-project differences; a project-cluster bootstrap CI (projects,
then plans within projects); Welch p; the interaction
(e20 − e19) − (e18 − e0); synthetic exposure per run; generation and OCR
verification counts. Stages 1-2 (`scripts/ocr_diagnostics.py`): OCR coverage,
room classification of ground-truth rooms (zero-shot, text, DINOv2, fused) and
two-stage AP from the 12 agnostic e0 runs plus the text head, compared with the
12 multiclass e0 runs.

## Missing runs

A run that fails twice for infrastructure reasons is reported as missing; the
analysis uses the seeds available in both arms of a contrast and states n.

## Known limitations

The 21 held-out plans come from three buildings and were used to evaluate
earlier arms; they are not a fresh test set. The control arm e0 was trained
before this study with code that differs only in the added, unused text path.
GPU types are not balanced across runs.
