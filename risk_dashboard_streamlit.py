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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from return_risk_core import (
    FEATURE_ORDER,
    counterfactuals,
    evaluate_threshold,
    feature_ablation,
    shap_values_for,
    select_policy,
    threshold_sweep,
    train_pipeline,
)

st.set_page_config(page_title="Return-Risk Scorer", layout="centered")

@st.cache_resource
def load_pipeline():
    return train_pipeline()


pipeline = load_pipeline()
clf = pipeline["model"]
explanation_model = pipeline["explanation_model"]
X_test = pipeline["X_test"]
y_test = pipeline["y_test"]
X_validation = pipeline["X_validation"]
y_validation = pipeline["y_validation"]
shap_values_test = shap_values_for(explanation_model, X_test)

y_proba = clf.predict_proba(X_test)[:, 1]
validation_probabilities = clf.predict_proba(X_validation)[:, 1]

st.sidebar.header("Evaluation settings")
decision_threshold = st.sidebar.slider(
    "Abusive-return threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.15,
    step=0.05,
    help="Orders at or above this calibrated probability enter the review policy.",
)
y_pred = (y_proba >= decision_threshold).astype(int)

metrics = evaluate_threshold(y_test, y_proba, decision_threshold)
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]
tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
flagged_rate = metrics["flagged_rate"]

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
st.caption(
    f"Metrics at a {decision_threshold:.0%} model threshold. Changing this cutoff "
    "trades missed abusive returns for fewer false accusations."
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision", f"{precision:.2f}")
c2.metric("Recall", f"{recall:.2f}")
c3.metric("F1", f"{f1:.2f}")
c4.metric("Test orders", f"{len(y_test)}")
c5.metric("Flagged at threshold", f"{flagged_rate:.1%}")

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
# Policy economics and robustness
# -----------------------------
st.subheader("Policy tradeoff: customer impact vs. missed abuse")
st.caption(
    "Illustrative unit costs: a false accusation costs 1, a missed abusive return "
    "costs 2, and sending a case to review costs 0.1. These are not business claims."
)
policy_table = threshold_sweep(y_validation, validation_probabilities)
best_policy = select_policy(y_validation, validation_probabilities)
policy_chart = policy_table.set_index("threshold")[["expected_cost"]]
st.line_chart(policy_chart)
st.info(
    f"Validation-selected threshold: {best_policy['threshold']:.0%}. "
    f"Validation recall is {best_policy['recall']:.1%} at a "
    f"{best_policy['false_positive_rate']:.1%} false-positive rate; "
    "the policy guardrail is 10%."
)

st.subheader("Robustness: what happens when one signal disappears?")
st.caption(
    "Each feature is replaced with its training median. The F1 drop estimates how "
    "dependent the model is on that signal; it is not a causal claim."
)
ablation_table = feature_ablation(clf, X_test, y_test, decision_threshold)
st.dataframe(
    ablation_table[["feature", "f1_without_signal", "f1_drop"]].rename(
        columns={"f1_without_signal": "F1 after ablation", "f1_drop": "F1 drop"}
    ),
    hide_index=True,
    width="stretch",
)

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
row_sv = shap_values_for(explanation_model, input_row)[0]

st.caption("Why the model scored it this way (this order's own SHAP contributions):")
contrib = pd.Series(row_sv, index=FEATURE_ORDER).sort_values(key=abs, ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#ef6a6a" if value > 0 else "#59c3c3" for value in contrib.values]
bars = ax.barh(contrib.index, contrib.values, color=colors)
ax.axvline(0, color="#777777", linewidth=1)
ax.set_xlabel("SHAP contribution to abusive-return risk")
ax.set_title("Why this order received its risk score")
ax.tick_params(axis="y", labelsize=9)
for bar, value in zip(bars, contrib.values):
    offset = 0.01 if value >= 0 else -0.01
    alignment = "left" if value >= 0 else "right"
    ax.text(
        value + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.3f}",
        va="center",
        ha=alignment,
        fontsize=8,
    )
fig.tight_layout()
st.pyplot(fig)

st.caption("Counterfactual checks: one plausible change at a time")
counterfactual_table = counterfactuals(clf, input_row)
st.dataframe(
    counterfactual_table.rename(
        columns={
            "current_value": "Current",
            "alternative_value": "Alternative",
            "risk_change": "Risk change",
            "new_risk": "New risk",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.divider()
st.caption(
    "Strictly defense-only: this model scores and routes for human review — it never "
    "auto-denies a return or takes unilateral action against a customer."
)
