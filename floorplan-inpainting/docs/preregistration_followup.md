# Pre-registration: follow-up experiments (exploratory)

Written 2026-10-13, after the OCR study results (`docs/preregistration_ocr.md`)
were seen and before any follow-up run. All analyses here are **exploratory**:
the designs were chosen in response to earlier results. The SHA-256 of this
file is recorded in `outputs/followup/preregistration_record.txt` at submission.

Common to all detector arms: multiclass track, seeds 1001-1012, unchanged
recipe (`configs/baseline.yaml`, 25 epochs × 35 steps, batch 1), endpoint
multiclass mask AP@[.50:.95] of `last.pth` on the 21 held-out plans. The
control e0 is the 12 existing runs of the multiclass follow-up
(`outputs/confirm_mc`); e19 is reused from the OCR study.

## X1: copy-paste towards class balance (e22-e24)

No earlier strategy converted more than about 20 rooms per rare class, while
the six frequent classes have 87-170 training rooms each. Strategy C
copy-paste (real rare training rooms, with their printed labels, warped into
geometry-compatible frequent rooms; e8/e16 pairing, QC and code) is scaled to
a per-class target of 20 (e22), 50 (e23) and 100 (e24) training instances
(real + converted), with donor reuse capped at 4, 10 and 20. Every accepted
edit of a source plan goes into one synthetic variant of that plan; the
`paired_variant` sampler (swap probability 0.5) keeps e0's plan order and step
budget. Achieved counts are limited by compatible geometry and are reported.

- **F1 (dose trend):** per seed, the OLS slope of AP on the number of
  converted rooms across e0, e22, e23, e24; one-sided one-sample t-test of
  slope > 0, α = 0.05.
- **F2:** e22, e23, e24 each vs e0, one-sided paired t, Holm over the three.
- Per-class AP of the frequent classes and rare macro AP are reported per arm
  to show any cost of shifting the class prior.

## X4: text-aware model with a trainable text branch (e21)

The OCR study's text branch barely left its zero initialisation (last-layer
norm 0.24 vs 3.1 for the classifier). e21 = e19 with (a) a separate AdamW
(lr 1e-3, weight decay 1e-4, no step decay) for the text branch only, the
detector keeping e0's SGD, and (b) pooling only tokens with at least 50% of
their box inside the proposal.

- **F3:** e21 vs e0 and e21 vs e19, one-sided paired t, Holm over the two.

## X2: balanced two-stage OCR classifier (descriptive)

The Stage-2 text head of the OCR study is retrained on the real training
rooms plus the converted rare rooms of the e24 (b100) synthetic plans, read
by the same OCR, and applied to the same 12 agnostic e0 runs. Compared per
seed with the original two-stage result (paired t, reported, no decision).

## X3: SAM 3 proposals + OCR text head (descriptive)

The OCR text head labels the existing SAM 3 class-agnostic test masks; scored
on the 11 test plans against SAM 3 multiclass (single SAM 3 run, no test).

## Decision rule

For F2 and F3 a contrast is supported only if Holm rejects and the paired
seed × image bootstrap 95% CI excludes 0. F1 is supported if p < 0.05.
Smallest effect of interest +0.01 AP. Because these analyses are exploratory,
any supported result is a hypothesis for future confirmation, not a
confirmed effect.
