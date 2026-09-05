# Observations

## What the current result means

The current policy reaches 52.9% recall and 8.8% false-positive rate on the
untouched test set. Precision is only 30.5%, so most flagged cases are still
legitimate. This is a useful improvement in customer-impact control, not a
production fraud-detection result.

## Evaluation protocol

The synthetic dataset contains 5,000 rows. The pipeline trains on 60% of the
rows, selects the operating threshold on a separate 20% validation split, and
reports final metrics once on the remaining untouched 20% test split (1,000
orders). The test set contains 68 abusive cases. This prevents threshold
selection from quietly overfitting the final reported result.

The synthetic abusive-return rate is 6.84%, closer to a rare-event setting than
the previous 31% benchmark, but it is still not a measured Razorpay rate.
The test set has only 68 positive cases, so the reported metrics have sampling
uncertainty and should not be treated as production estimates.

## Why synthetic data is still limited

The labels come from a hand-designed probabilistic process with Gaussian noise.
That creates a learnable benchmark, but it does not recreate real return abuse.
Real data would include missing values, label disputes, correlated signals,
seasonality, merchant differences, and adversarial adaptation.

The synthetic feature distribution also has assumptions that need replacement
before deployment. For example, device/IP history and payment risk are treated
as clean numeric signals, while real systems would need identity, privacy,
data-quality, and governance controls around them.

## Policy choice

The policy selects a 15% probability threshold on validation data by maximizing
recall while keeping the validation false-positive rate at or below 10%. On the
untouched test set this produced 52.9% recall, 8.8% false-positive rate, and
30.5% precision. The 10% guardrail is a demonstration operating assumption and
must be replaced with merchant estimates in a real deployment.

The threshold is not universally optimal. It is optimal only for the stated
validation guardrail and this synthetic split. The dashboard intentionally lets
the user move it and inspect the resulting customer-impact tradeoff.

## Model selection and explainability

Calibrated logistic regression, RandomForest, Gradient Boosting, and histogram
Gradient Boosting candidates were compared under the same development protocol.
Histogram Gradient Boosting was retained because it captured nonlinear
interactions better under the recall-at-10%-false-positive-rate objective. The
final test set was not used to choose the model or threshold.

SHAP explanations and feature ablation are included because the retained model
is less directly interpretable than logistic regression. Ablation replaces one
feature at a time with its median and measures the policy F1 change. It is a
dependency check, not proof of causality.

## What this does and does not prove

**It does not prove:** that this model, this threshold, or these metrics would
work on real order/return data.

**It does demonstrate:** a reproducible training pipeline, calibrated scoring,
validation-only policy selection, untouched-test reporting, customer-impact
guardrails, SHAP explanations, counterfactual checks, robustness analysis, and
defense-only human-review routing.

## Next validation step

Before making a production claim, run repeated time-based evaluation on a real
or anonymized return/chargeback dataset. Report confidence intervals, missing
data behavior, calibration by merchant segment, and performance drift over
time. No further synthetic tuning should be described as real-world lift.
