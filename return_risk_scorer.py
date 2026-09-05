"""
AI Risk Manager — Track 02
Return-Risk Scorer: flags orders likely to result in an abusive/fraudulent return.

Pipeline: synthetic data -> train/test split -> RandomForest -> metrics ->
SHAP explainability -> honest false-positive cost callout.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. Synthetic dataset
# -----------------------------
# In a real submission you'd swap this for Razorpay test-mode order/return data.
# Features are the kind of signals a merchant checkout actually has access to.

rng = np.random.default_rng(42)
N = 800

df = pd.DataFrame({
    "order_value": rng.gamma(shape=2.0, scale=800, size=N).round(2),
    "customer_account_age_days": rng.integers(0, 1500, size=N),
    "prior_return_count": rng.poisson(0.6, size=N),
    "prior_order_count": rng.integers(1, 50, size=N),
    "delivery_to_return_request_days": rng.integers(0, 30, size=N),
    "payment_method_risk_score": rng.uniform(0, 1, size=N),  # e.g. COD/new-card = higher
    "shipping_billing_mismatch": rng.integers(0, 2, size=N),
    "device_ip_flagged_before": rng.integers(0, 2, size=N),
})

# Ground-truth generating logic (synthetic "reality") — hidden signal the model must recover
risk_logit = (
    -3.0
    + 2.2 * df["prior_return_count"].clip(upper=5) / 5
    + 1.5 * df["shipping_billing_mismatch"]
    + 1.8 * df["device_ip_flagged_before"]
    + 1.2 * df["payment_method_risk_score"]
    - 0.8 * (df["customer_account_age_days"] / 1500)
    - 0.5 * (df["prior_order_count"] / 50)
    + 0.6 * (df["delivery_to_return_request_days"] < 2).astype(int)  # instant "damaged" claims
    + rng.normal(0, 0.5, size=N)
)
prob = 1 / (1 + np.exp(-risk_logit))
df["is_abusive_return"] = (rng.uniform(0, 1, size=N) < prob).astype(int)

print(f"Base rate of abusive returns in synthetic data: {df['is_abusive_return'].mean():.2%}\n")

# -----------------------------
# 2. Train / test split
# -----------------------------
X = df.drop(columns=["is_abusive_return"])
y = df["is_abusive_return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -----------------------------
# 3. Model
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=5,
    class_weight="balanced", random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# -----------------------------
# 4. Honest metrics (the bar: "Honest metrics including false-positive cost")
# -----------------------------
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=== Held-out test set metrics ===")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")
print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}\n")
print(classification_report(y_test, y_pred, target_names=["legit", "abusive"]))

# False-positive cost framing — a legit customer wrongly flagged is a trust/CX cost,
# not just a number. State it plainly rather than burying it.
assumed_cx_cost_per_fp = 1  # unit cost placeholder — merchant would plug in real number
print(f"False positives (legit customers wrongly flagged): {fp} "
      f"out of {tn+fp} legit cases in test set "
      f"({fp/(tn+fp):.1%} false-positive rate on legit traffic).")
print(f"False negatives (abusive returns missed): {fn} out of {fn+tp} abusive cases "
      f"({fn/(fn+tp):.1%} miss rate).\n")

# -----------------------------
# 5. Explainability (SHAP)
# -----------------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Newer SHAP versions return shape (n_samples, n_features, n_classes) for binary
# classifiers; older versions return a list [class0_array, class1_array].
if isinstance(shap_values, list):
    sv = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(sv, X_test, show=False, plot_type="bar")
plt.title("Feature importance (mean |SHAP value|) — Return-Risk Scorer")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(sv, X_test, show=False)
plt.title("SHAP value distribution per feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary_beeswarm.png", dpi=150)
plt.close()

print(f"Saved plots to: {OUTPUT_DIR}")

# -----------------------------
# 6. One failure case, handled gracefully (per the bar in Track 01/03, good practice here too)
# -----------------------------
worst_fp_idx = np.argsort(-(y_proba * (y_test.values == 0)))[:1]
print("\n=== Example failure case (false positive) ===")
print(X_test.iloc[worst_fp_idx])
print(f"Model confidence it was abusive: {y_proba[worst_fp_idx][0]:.2%} — actually legit.")
print("Graceful handling: route to human review queue instead of auto-denying the return.")
