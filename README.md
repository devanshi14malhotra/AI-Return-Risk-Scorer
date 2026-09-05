# AI Return-Risk Scorer

Flags orders likely to end in an abusive or fraudulent return, using only signals available at checkout time — before the return is ever filed.

## What's here

- `return_risk_scorer.py` — standalone training + evaluation script. Trains a RandomForest, prints precision/recall/F1 on a held-out test set, and saves SHAP explainability plots.
- `risk_dashboard_streamlit.py` — interactive dashboard. Trains the same model live and lets you score a hypothetical order in real time via the actual `clf.predict_proba()` call (not an approximation).
- `requirements.txt`

## Run it

```bash
pip install -r requirements.txt
streamlit run risk_dashboard_streamlit.py
```

Or just run the training script directly:

```bash
python return_risk_scorer.py
```

## Results (held-out test set, 200 orders)

| Metric | Value |
|---|---|
| Precision | 0.49 |
| Recall | 0.40 |
| F1 | 0.44 |
| False-positive rate (legit customers wrongly flagged) | 18.8% |
| Miss rate (abusive returns missed) | 59.7% |

Top drivers (SHAP): `device_ip_flagged_before`, `shipping_billing_mismatch`, `payment_method_risk_score`.

## Honest caveat

Training data is **synthetic** — no real Razorpay order/return data was available for this build. The model itself, and every metric and score reported above, are real: a genuine scikit-learn RandomForest trained and evaluated in the standard way.

On synthetic data we get precision 0.49 / recall 0.40. We're not claiming this beats real fraud systems — we're demonstrating the pipeline: detection → honest metrics → explainability → human-review routing. With real transaction data, the same architecture would very likely perform better, since real fraud has structural patterns synthetic noise doesn't fully capture.

See `OBSERVATIONS.md` for more on why the results look the way they do.

## Design choice: defense-only

The model scores and routes borderline cases to human review. It never auto-denies a return or takes unilateral action against a customer — a false accusation costs more trust than a missed return costs money.

--

**Track 02 — AI Risk Manager**
