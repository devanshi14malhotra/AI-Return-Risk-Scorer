"""Shared training and evaluation utilities for the return-risk project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

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


def make_dataset(seed: int = 42, rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "order_value": rng.gamma(shape=2.0, scale=800, size=rows).round(2),
        "customer_account_age_days": rng.integers(0, 1500, size=rows),
        "prior_return_count": rng.poisson(0.6, size=rows),
        "prior_order_count": rng.integers(1, 50, size=rows),
        "delivery_to_return_request_days": rng.integers(0, 30, size=rows),
        "payment_method_risk_score": rng.uniform(0, 1, size=rows),
        "shipping_billing_mismatch": rng.binomial(1, 0.12, size=rows),
        "device_ip_flagged_before": rng.binomial(1, 0.06, size=rows),
    })

    risk_logit = (
        -4.5
        + 2.0 * data["prior_return_count"].clip(upper=5) / 5
        + 2.4 * data["shipping_billing_mismatch"]
        + 3.0 * data["device_ip_flagged_before"]
        + 1.6 * data["payment_method_risk_score"]
        - 0.8 * (data["customer_account_age_days"] / 1500)
        - 0.5 * (data["prior_order_count"] / 50)
        + 1.0 * (data["delivery_to_return_request_days"] < 2).astype(int)
        + rng.normal(0, 0.65, size=rows)
    )
    probability = 1 / (1 + np.exp(-risk_logit))
    data["is_abusive_return"] = (
        rng.uniform(0, 1, size=rows) < probability
    ).astype(int)
    return data


def train_pipeline(seed: int = 42) -> dict:
    data = make_dataset(seed=seed)
    X = data[FEATURE_ORDER]
    y = data["is_abusive_return"]
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=seed,
    )

    base_model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=20,
        learning_rate=0.04,
        l2_regularization=2.0,
        random_state=seed,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="sigmoid", cv=3
    )
    calibrated_model.fit(X_train, y_train)
    base_model.fit(X_train, y_train)

    return {
        "data": data,
        "model": calibrated_model,
        "explanation_model": base_model,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }


def shap_values_for(model, X: pd.DataFrame) -> np.ndarray:
    import shap

    values = shap.Explainer(model, X)(X)
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        return values[1]
    if isinstance(values, np.ndarray) and values.ndim == 3:
        return values[:, :, 1]
    return values


def evaluate_threshold(y_true, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "flagged_rate": predictions.mean(),
        "false_positive_rate": fp / max(tn + fp, 1),
        "miss_rate": fn / max(fn + tp, 1),
    }


def threshold_sweep(
    y_true, probabilities, fp_cost: float = 1.0, fn_cost: float = 2.0,
    review_cost: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        metrics = evaluate_threshold(y_true, probabilities, round(float(threshold), 2))
        metrics["expected_cost"] = (
            metrics["fp"] * fp_cost
            + metrics["fn"] * fn_cost
            + (metrics["tp"] + metrics["fp"]) * review_cost
        ) / len(y_true)
        rows.append(metrics)
    return pd.DataFrame(rows)


def select_policy(
    y_true,
    probabilities,
    max_false_positive_rate: float = 0.10,
) -> pd.Series:
    """Select the highest-recall threshold within a customer-impact guardrail."""
    candidates = threshold_sweep(y_true, probabilities)
    allowed = candidates[candidates["false_positive_rate"] <= max_false_positive_rate]
    if allowed.empty:
        return candidates.sort_values(
            ["false_positive_rate", "recall"], ascending=[True, False]
        ).iloc[0]
    return allowed.sort_values(
        ["recall", "precision", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]


def counterfactuals(model, input_row: pd.DataFrame) -> pd.DataFrame:
    current_probability = model.predict_proba(input_row)[0, 1]
    alternatives = {
        "customer_account_age_days": max(
            int(input_row.iloc[0]["customer_account_age_days"]) + 365, 0
        ),
        "prior_return_count": max(int(input_row.iloc[0]["prior_return_count"]) - 1, 0),
        "prior_order_count": int(input_row.iloc[0]["prior_order_count"]) + 5,
        "delivery_to_return_request_days": min(
            int(input_row.iloc[0]["delivery_to_return_request_days"]) + 3, 30
        ),
        "payment_method_risk_score": max(
            float(input_row.iloc[0]["payment_method_risk_score"]) - 0.2, 0.0
        ),
        "shipping_billing_mismatch": 0,
        "device_ip_flagged_before": 0,
    }
    rows = []
    for feature, alternative in alternatives.items():
        candidate = input_row.copy()
        candidate.loc[candidate.index[0], feature] = alternative
        new_probability = model.predict_proba(candidate)[0, 1]
        rows.append({
            "feature": feature,
            "current_value": input_row.iloc[0][feature],
            "alternative_value": alternative,
            "risk_change": new_probability - current_probability,
            "new_risk": new_probability,
        })
    return pd.DataFrame(rows).sort_values("risk_change")


def feature_ablation(
    model, X_test: pd.DataFrame, y_test, threshold: float = 0.5
) -> pd.DataFrame:
    baseline = evaluate_threshold(
        y_test, model.predict_proba(X_test)[:, 1], threshold=threshold
    )["f1"]
    rows = []
    for feature in FEATURE_ORDER:
        altered = X_test.copy()
        altered[feature] = altered[feature].median()
        score = evaluate_threshold(
            y_test, model.predict_proba(altered)[:, 1], threshold=threshold
        )["f1"]
        rows.append({
            "feature": feature,
            "baseline_f1": baseline,
            "f1_without_signal": score,
            "f1_drop": baseline - score,
        })
    return pd.DataFrame(rows).sort_values("f1_drop", ascending=False)