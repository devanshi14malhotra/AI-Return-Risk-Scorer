# AI Return-Risk Scorer

**Trust-aware return abuse detection with explainable, cost-sensitive decisions.**

Flags orders likely to end in an abusive or fraudulent return, using signals
available by the time a return is being evaluated.

## What's here

- `return_risk_core.py` — shared pipeline for synthetic data generation, calibrated nonlinear-model training, threshold analysis, counterfactuals, and robustness checks.
- `return_risk_scorer.py` — command-line evaluation with held-out metrics, Brier score, cost-sensitive policy selection, SHAP plots, and feature-ablation results.
- `risk_dashboard_streamlit.py` — interactive review console with threshold controls, customer-impact tradeoffs, signal robustness, live scoring, SHAP explanations, and counterfactual checks.
- `requirements.txt`

## Run it

```bash
pip install -r requirements.txt
python -m streamlit run risk_dashboard_streamlit.py
```

Or just run the training script directly:

```bash
python return_risk_scorer.py
```

## Current evaluation (untouched test set, 1,000 orders)

The pipeline uses 60% of 5,000 synthetic orders for training, 20% for selecting
the policy threshold, and a final untouched 20% for reporting. The CLI selects
the threshold on validation data by maximizing recall subject to a 10% maximum
false-positive rate, then evaluates once on the untouched test set. The
dashboard lets you move the threshold and see the operational tradeoff.

| Metric | Value |
|---|---|
| Precision | 0.31 |
| Recall | 0.53 |
| F1 | 0.39 |
| False-positive rate (legit customers wrongly flagged) | 8.8% |
| Miss rate (abusive returns missed) | 47.1% |
| Brier score (probability calibration, lower is better) | 0.055 |

Validation selects a 15% threshold under the recall-at-10%-false-positive-rate
policy. The final test metrics above are not used to choose that threshold. The
10% guardrail is a demonstration operating assumption, not a production claim.

## What makes this project different

- **Calibrated probabilities:** the nonlinear scorer is evaluated as a probability, not only as a class label.
- **Cost-sensitive policy:** the dashboard shows how a threshold changes false accusations, missed abuse, and review volume.
- **Counterfactual explanations:** each scored order gets plausible one-feature changes and the resulting risk movement.
- **Robustness analysis:** feature ablation shows how much performance changes when a signal is unavailable.
- **Defense-only workflow:** high-risk cases go to human review; the system never auto-denies a return.

Top drivers (SHAP): `payment_method_risk_score`, `shipping_billing_mismatch`, `device_ip_flagged_before`.

## Honest caveat

Training data is **synthetic** — no real Razorpay order/return data was available for this build. The model itself, and every metric and score reported above, are real: a genuine scikit-learn HistGradientBoostingClassifier trained, calibrated, and evaluated with a validation-selected policy on an untouched test split.

On synthetic data we get precision 0.31 / recall 0.53 at the validation-selected policy threshold. The model stays below the 10% false-positive-rate guardrail on the untouched test set, but precision is only 0.31, so it is not being presented as production-ready. We're demonstrating the pipeline: calibrated nonlinear scoring → validation-selected policy → honest metrics → explainability → human-review routing. With real transaction data, the same architecture would need fresh calibration and validation before use.

See `OBSERVATIONS.md` for more on why the results look the way they do.

## Design choice: defense-only

The model scores and routes borderline cases to human review. It never auto-denies a return or takes unilateral action against a customer — a false accusation costs more trust than a missed return costs money.

