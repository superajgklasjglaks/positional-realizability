# Pilot analysis — does training-free α_L predict length generalization?

**Setup.** 5 PEs × 5 tasks = 25 cells. 2-layer, 4-head, d=64 transformers, vocab 12, trained at
variable length up to `L0=20` (2500 steps), evaluated on extrapolation rows `[L0, L)` at `L=60`.
`α_L` computed entirely before training. Hardware: M3 Pro / MPS, 7.2 min total.

## Headline numbers
- Spearman (all 25): **ρ = 0.21, p = 0.32** (not significant)
- Spearman (learnable cells, train_acc>0.8): ρ = 0.17, p = 0.57
- Pearson (all 25): **0.54**

## The structure behind the weak rank correlation

| signal | cells | reading |
|---|---|---|
| **α_L ≥ 0.5 ⇒ LG high** | rope/prev 1.0→.91, rope/skip3 1.0→.82, sinus/first 1.0→.96, alibi/first 1.0→1.0 | **4/4 — sufficiency holds** |
| **growing-offset fails for all** | copy & reverse, every PE: α≈0, LG .11–.16 | predictor + models agree |
| **predictor misses (α=0 but LG high)** | alibi/prev 0→.998, learned-abs/first 0→1.0, nope/first 0→1.0, nope/prev 0→.50 | content / absolute-locking rescue |

## Diagnosis (each miss is explainable, not noise)
1. **ALiBi prev/skip3.** The single-head *positional-only* kernel models ALiBi as a monotone bias
   `{1, (i−j)}`, which cannot peak at an interior offset → α=0. Real ALiBi reaches prev-token by
   combining recency bias with the **content/value pathway** — outside the (S3) abstraction.
2. **learned-abs first.** Target = absolute position 0, which is always in the trained range; the model
   locks onto it. Our learned-abs basis only populates the `L0×L0` block, so it scored new query rows as
   uncontrolled (α=0). The predictor is too pessimistic for *absolute* targets that point into the
   trained region.
3. **NoPE prev.** Partial LG via causal/content structure — explicitly out of the positional predictor
   per theory §5.

All three are **(S3) content-neutrality / single-head scope violations** — the pre-registered failure
modes #2 and #3.

## What this means
- **Solid, honest claim:** *high `α_L` ⇒ length generalization* (sufficiency). Matches repaired
  Theorem 1; empirically high-precision (4/4).
- **Not supported:** `α_L` as a symmetric rank-order predictor or a low-side go/no-go. This mirrors the
  theory: Theorem 2b's general impossibility is a **conjecture**, not proven.
- **Theory ⇄ experiment are consistent** about which direction is reliable.

## Options going forward (decision needed)
- **A — Strengthen (v0.2):** make the predictor faithful to each PE's *actual* realizable mechanism
  (ALiBi with self-exclusion; learned-abs absolute-locking; an optional content channel), make empirics
  strictly position-essential, prove Theorem 2b for one fixed-PE family (sinusoidal frequency band),
  re-run. Plausible the correlation sharpens substantially because the residual is structured.
- **B — Reframe (sufficiency paper):** lead with the proven sufficiency result + the taxonomy of when
  content substitutes for position. Honest, cleaner, lower-ambition.
- **C — Reassess venue/idea:** current maturity is workshop / borderline main-track, consistent with
  the original novelty-6 scoring.
