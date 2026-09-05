"""
AI Risk Manager — Track 02
Return-Risk Scorer — Streamlit dashboard

Run with:
    streamlit run risk_dashboard_streamlit.py

Everything shown here — metrics, confusion matrix, SHAP importances, and the
live slider score — comes from one real, trained scikit-learn RandomForest.
Nothing is hand-approximated. Training data is synthetic (see note in app),
since we don't have real Razorpay order/return data for this build.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Return-Risk Scorer", layout="centered")

FEATURE_ORDER = [
    "order_value",
    "customer_account_age_days",
    "prior_return_count",
    "prior_order_count",
    "delivery_to_return_request_days",
    "payment_method_risk_score",
    "shipping_billing_mismatch",
    "device_ip_flagged_before",
]

# -----------------------------
# Train once, cache the real model + data
# -----------------------------
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    N = 800

    df = pd.DataFrame({
        "order_value": rng.gamma(shape=2.0, scale=800, size=N).round(2),
        "customer_account_age_days": rng.integers(0, 1500, size=N),
        "prior_return_count": rng.poisson(0.6, size=N),
        "prior_order_count": rng.integers(1, 50, size=N),
        "delivery_to_return_request_days": rng.integers(0, 30, size=N),
        "payment_method_risk_score": rng.uniform(0, 1, size=N),
        "shipping_billing_mismatch": rng.integers(0, 2, size=N),
        "device_ip_flagged_before": rng.integers(0, 2, size=N),
    })

    risk_logit = (
        -3.0
        + 2.2 * df["prior_return_count"].clip(upper=5) / 5
        + 1.5 * df["shipping_billing_mismatch"]
        + 1.8 * df["device_ip_flagged_before"]
        + 1.2 * df["payment_method_risk_score"]
        - 0.8 * (df["customer_account_age_days"] / 1500)
        - 0.5 * (df["prior_order_count"] / 50)
        + 0.6 * (df["delivery_to_return_request_days"] < 2).astype(int)
        + rng.normal(0, 0.5, size=N)
    )
    prob = 1 / (1 + np.exp(-risk_logit))
    df["is_abusive_return"] = (rng.uniform(0, 1, size=N) < prob).astype(int)

    X = df[FEATURE_ORDER]
    y = df["is_abusive_return"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)
    shap_values_test = explainer.shap_values(X_test)
    if isinstance(shap_values_test, list):
        sv = shap_values_test[1]
    elif isinstance(shap_values_test, np.ndarray) and shap_values_test.ndim == 3:
        sv = shap_values_test[:, :, 1]
    else:
        sv = shap_values_test

    return clf, explainer, X_train, X_test, y_train, y_test, sv


clf, explainer, X_train, X_test, y_train, y_test, shap_values_test = train_model()

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# -----------------------------
# Header
# -----------------------------
st.title("Return-Risk Scorer")
st.caption("Track 02 · AI Risk Manager — flags orders likely to end in an abusive return")

st.info(
    "Training data here is synthetic (no real Razorpay order/return data available for "
    "this build) — but the model, metrics, and the score below are all real: a genuine "
    "scikit-learn RandomForest, trained once, running live.",
    icon="ℹ️",
)

# -----------------------------
# Metrics
# -----------------------------
st.subheader("Held-out test performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{precision:.2f}")
c2.metric("Recall", f"{recall:.2f}")
c3.metric("F1", f"{f1:.2f}")
c4.metric("Test orders", f"{len(y_test)}")

st.subheader("Confusion matrix")
cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual: legit", "Actual: abusive"],
    columns=["Pred: legit", "Pred: abusive"],
)
st.table(cm_df)

fp_rate = fp / (tn + fp)
fn_rate = fn / (fn + tp)
st.markdown(
    f"**False-positive rate on legit customers:** {fp_rate:.1%} &nbsp;·&nbsp; "
    f"**Miss rate on abusive returns:** {fn_rate:.1%}"
)
st.caption(
    "At this threshold the model is conservative about accusing customers — a wrong "
    "accusation costs more trust than a missed return costs money."
)

# -----------------------------
# SHAP importance (real, computed on the actual test set)
# -----------------------------
st.subheader("What actually drives the score (SHAP)")
mean_abs_shap = pd.Series(
    np.abs(shap_values_test).mean(axis=0), index=FEATURE_ORDER
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(mean_abs_shap.index, mean_abs_shap.values, color="#c98a3e")
ax.set_xlabel("mean |SHAP value|")
fig.tight_layout()
st.pyplot(fig)

# -----------------------------
# Live scoring — REAL model inference
# -----------------------------
st.subheader("Score a hypothetical order")
st.caption("This calls `clf.predict_proba()` on the actual trained model — not an approximation.")

col1, col2 = st.columns(2)
with col1:
    order_value = st.slider("Order value (₹)", 0, 10000, 1500)
    account_age = st.slider("Customer account age (days)", 0, 1500, 365)
    prior_returns = st.slider("Prior return count", 0, 10, 0)
    prior_orders = st.slider("Prior order count", 1, 50, 5)
with col2:
    return_delay = st.slider("Days between delivery & return request", 0, 30, 10)
    pay_risk = st.slider("Payment method risk score", 0.0, 1.0, 0.2)
    mismatch = st.checkbox("Shipping / billing address mismatch")
    device_flag = st.checkbox("Device or IP flagged before")

input_row = pd.DataFrame([{
    "order_value": order_value,
    "customer_account_age_days": account_age,
    "prior_return_count": prior_returns,
    "prior_order_count": prior_orders,
    "delivery_to_return_request_days": return_delay,
    "payment_method_risk_score": pay_risk,
    "shipping_billing_mismatch": int(mismatch),
    "device_ip_flagged_before": int(device_flag),
}])[FEATURE_ORDER]

risk_prob = clf.predict_proba(input_row)[0, 1]
pct = risk_prob * 100

st.metric("Model's predicted risk", f"{pct:.1f}%")

if pct >= 60:
    st.error("Route to human review — do not auto-deny.")
elif pct >= 30:
    st.warning("Watch, don't block — not enough signal to intervene alone.")
else:
    st.success("Process normally — no meaningful risk signal.")

# Per-order SHAP explanation for this exact input — also real
row_shap = explainer.shap_values(input_row)
if isinstance(row_shap, list):
    row_sv = row_shap[1][0]
elif isinstance(row_shap, np.ndarray) and row_shap.ndim == 3:
    row_sv = row_shap[0, :, 1]
else:
    row_sv = row_shap[0]

st.caption("Why the model scored it this way (this order's own SHAP contributions):")
contrib = pd.Series(row_sv, index=FEATURE_ORDER).sort_values(key=abs, ascending=False)
st.bar_chart(contrib)

st.divider()
st.caption(
    "Strictly defense-only: this model scores and routes for human review — it never "
    "auto-denies a return or takes unilateral action against a customer."
)
