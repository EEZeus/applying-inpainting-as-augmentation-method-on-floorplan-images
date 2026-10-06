# Pre-registration addendum: multiclass follow-up (Strategy A vs oversampling)

Committed, hashed and recorded in `outputs/confirm_mc/preregistration_record.txt`
before any job of this study runs. It follows the confirmatory study in
`preregistration_confirm.md`, whose class-agnostic hypotheses were all
unsupported.

## Why

The leakage-free re-score of the grid (exploratory, 3 seeds) found multiclass
gains whose intervals excluded zero for e1 (rare-weighted sampling, +0.010),
e2 (Strategy A inpainting, +0.016) and e7. Because e1 adds no generated pixels,
most of e2's gain may be oversampling of rare-room plans. This study tests both
with fresh seeds.

## Design

- Arms (multiclass track, unchanged grid configs and recipe): e0 original;
  e1 rare-weighted sampling; e2 Strategy A (uniform over originals + 74 accepted
  Strategy A variants).
- Seeds 1001-1012 (fresh; never used for any selection).
- Endpoint: multiclass COCO mask AP@[.50:.95] (all 19 categories) of `last.pth`
  (epoch 25) on the 21 held-out plans.

## Hypotheses (fixed sequence, alpha 0.05 each)

- H4: e2 > e0. One-sided Welch t-test; supported only if p < 0.05 AND the
  two-level (seed x image) bootstrap 95% CI of the difference excludes 0.
- H5: e2 > e1, the gain of inpainted appearance beyond oversampling. Tested with
  the same rule only if H4 is supported.
- Descriptive only: e1 vs e0, rare-class macro mask AP, AP50, per-project and
  11-test-plan deltas, permutation p, Hedges' g.
- Smallest effect of interest: +0.01 multiclass AP (the multiclass seed s.d. on
  the 21 plans is about 0.002-0.007).
- Missing runs are reported, never imputed; no interim look.
