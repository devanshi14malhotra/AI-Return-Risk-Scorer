# Observations

## Why the numbers look mediocre — and why that's expected

Precision 0.49 / recall 0.40 is not a strong result by production fraud-detection
standards. That's intentional, and worth understanding rather than hiding.

The synthetic dataset was generated with a real signal (prior returns, address
mismatch, device flags, etc.) plus **added Gaussian noise** on top of that signal
before labeling. The noise is what caps recall around 40% — the model genuinely
cannot recover a pattern that isn't fully there. This is different from a case
where the model is bad and the data is clean.

## What this does and doesn't prove

**It doesn't prove:** that this exact model, with this exact performance, would
work on real order/return data. We have no real data to test that claim.

**It does prove:** the pipeline is real and complete — training, held-out
evaluation, honest metric reporting (including false-positive cost), SHAP
explainability, and a defense-only routing decision (review vs. process) instead
of an automatic action.

## On "does synthetic noise mimic real-world noise"

Be careful with this claim in a pitch — it's easy to overstate. Gaussian noise
added to a hand-designed logistic function is a reasonable *stand-in* for the
fact that real-world fraud signals are imperfect and probabilistic, not
deterministic. But real fraud data has properties this simulation does not
capture:

- **Class imbalance** far more extreme than the ~31% synthetic base rate here —
  real abusive-return rates are typically much lower single digits.
- **Adversarial adaptation** — real fraud patterns shift over time as bad actors
  learn what gets flagged; synthetic noise is static.
- **Correlated, non-random noise** — real-world "noise" often comes from missing
  or mislabeled data, not a clean random distribution.

The honest claim is: *"the noise demonstrates the model can't achieve perfect
separation on ambiguous cases, which is realistic in spirit"* — not *"this noise
mimics real-world fraud data."* The former is defensible under questioning; the
latter isn't, and a judge with any ML background may push on exactly this point.

## If there were more time

The single highest-leverage next step would be lowering the injected noise
(from `rng.normal(0, 0.5, ...)` to something like `0.15` in
`return_risk_scorer.py`) to see how much of the recall ceiling is noise-driven
versus model-driven — and, ideally, testing against any real (even small,
anonymized) return/chargeback dataset if one becomes available.
