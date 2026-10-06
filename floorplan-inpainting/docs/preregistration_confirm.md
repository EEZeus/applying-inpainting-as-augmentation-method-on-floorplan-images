# Pre-registration: leakage-free re-score and confirmatory study

This file is committed, and its SHA-256 and the commit hash are recorded in
`outputs/confirm/preregistration_record.txt`, before any job of the
confirmation chain runs. Nothing below may change after submission.

## Why this study exists

The Round-1 grid (14 arms x 2 tracks x 3 seeds) and the rotated
cross-validation have four defects that bias or blur the comparison:

1. The cross-validation pools the 10 validation plans that selected every
   `best.pth`, and scores only seed 42.
2. The fixed 35-draws-per-epoch budget makes synthetic plans *replace* real
   plans: augmented arms see each real plan 8-14 times instead of 25.
3. e8 (copy-paste) uses a multi-room dataset while e9-e11 use single-edit
   datasets, so "copy-paste vs diffusion" compares two data compositions.
4. The C2 "diffusion inpaint" seam is 6 original pixels at strength 0.30, about
   one Stable Diffusion latent cell, so e10/e11 barely contain generated pixels.

e8 was also the best of 13 arms, so its +0.036 class-agnostic gain is subject
to winner's curse and needs fresh seeds.

## Stage R: leakage-free re-score (exploratory)

Every existing grid checkpoint's `last.pth` (epoch 25, never selected on
validation data) is evaluated on all 21 held-out plans (10 val + 11 test,
projects P02, P04, P07): 14 arms x 2 tracks x 3 seeds = 84 evaluations.
Results are exploratory and Holm-adjusted per track. They feed one gate only.

## Generation: C2-wide

Strategy C2 is regenerated with `configs/augmentation_c_wide.yaml`: same pairing
plan, seed and QC as the original C datasets, but

- the diffusion band is 24 model pixels (3 latent cells), converted per
  candidate to original pixels;
- the warped donor's dark strokes (dilated 3 px) are excluded from the diffusion
  mask and remain under the donor-retention QC.

Strength is chosen from {0.50, 0.75} by a generation-only rule (no detector
metric): the highest strength with acceptance >= 0.80, mean donor-stroke
retention >= 0.95 among accepted samples, and median diffusion-mask area >= 30
latent cells. If none passes, e16/e17 are dropped. Both strengths are run on all
52 planned candidates.

## Matched datasets

e16 (copy-paste) and e17 (C2-wide) are composed by the existing multi-room
builder (5 +/- 2 % of common rooms per plan, config seed 2042) from the
intersection of (target, donor) pairs accepted in both modes, with a canonical
pool order, so both contain identical plans and conversions and differ only in
pixels (asserted). If fewer than 30 pairs survive, e16/e17 are dropped.

## Gates (can only cancel work)

- Futility for e8: if mean(e8 - e0) over the 3 grid seeds, class-agnostic mask AP
  of `last.pth` on the 21 held-out plans, is <= 0, e8 and H1 are dropped. If the
  re-score is incomplete the gate is not evaluable and e8 is kept.
- e0 runs if any other arm runs.

## Confirmatory arms (class-agnostic track only)

12 fresh seeds per arm: 1001-1012. Recipe unchanged (configs/baseline.yaml:
25 epochs x 35 steps, batch 1, SGD 0.005, step decay at 8).

| Arm | Data | Sampler |
|---|---|---|
| e0 | original 35 plans | permutation of the 35 plans per epoch (unchanged) |
| e8 | existing copy_paste_multi (exact replication) | unchanged (35 of 35+N draws) |
| e16 | matched copy-paste multi-room | paired_variant, p = 0.5 |
| e17 | matched C2-wide multi-room | paired_variant, p = 0.5 |

`paired_variant`: each epoch uses e0's exact plan order for the same seed
(`torch.randperm(35)` seeded with seed+epoch); each real plan that has synthetic
variants is swapped for one of them with probability 0.5 (independent RNG).
Every source plan occupies exactly one step per epoch, as in e0; plans with
variants show the original pixels in about half of their steps.

## Endpoint and tests

- Primary endpoint: class-agnostic COCO mask AP@[.50:.95] of `last.pth` on the
  21 held-out plans, one value per run.
- H1: e8 > e0. H2: e17 > e0. One-sided Welch t-tests, Holm at FWER 0.05 over
  the active hypotheses.
- A hypothesis is supported only if Holm rejects AND the two-level bootstrap
  (seeds within arm x shared images, 2000 resamples, percentile) 95% CI of the
  difference excludes 0.
- H3: e17 > e16, tested at alpha 0.05 with the same CI rule only if H2 is
  supported. Supporting H3 is the claim that generated pixels add value beyond
  copy-paste.
- Smallest effect size of interest: +0.02 AP. Reported alongside every test:
  Monte Carlo permutation p (200 000 permutations), Hedges' g, seed-paired
  differences (sensitivity only), per-project deltas, test-11-only deltas, AP50.
- Secondary, descriptive only: e16 vs e0, e8 vs e16 (sampler effect),
  e17 vs e16 when H2 is not supported.
- Missing runs: a task that fails twice is reported as missing, never replaced
  or imputed. No interim look at confirmatory numbers before all tasks finish.

## Out of scope

Multiclass and rare-class claims (17 rare test instances cannot resolve them),
SAM 3 (80 GB queue), LoRA arms, label-rewriting generation.
