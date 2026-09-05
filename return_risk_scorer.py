"""Train, evaluate, and explain the AI Return-Risk Scorer."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, classification_report

from return_risk_core import (
    evaluate_threshold,
    feature_ablation,
      select_policy,
    shap_values_for,
    threshold_sweep,
    train_pipeline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = train_pipeline()
model = pipeline["model"]
explanation_model = pipeline["explanation_model"]
X_test = pipeline["X_test"]
y_test = pipeline["y_test"]
X_validation = pipeline["X_validation"]
y_validation = pipeline["y_validation"]
probabilities = model.predict_proba(X_test)[:, 1]
validation_probabilities = model.predict_proba(X_validation)[:, 1]
best_policy = select_policy(y_validation, validation_probabilities)
policy_threshold = float(best_policy["threshold"])
metrics = evaluate_threshold(y_test, probabilities, threshold=policy_threshold)
predictions = (probabilities >= policy_threshold).astype(int)

print(f"Base rate of abusive returns in synthetic data: "
      f"{pipeline['data']['is_abusive_return'].mean():.2%}\n")
print(f"=== Untouched test set metrics at {policy_threshold:.0%} policy threshold ===")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")
print(
    "Confusion matrix [[TN FP] [FN TP]]:\n"
    f"[[{metrics['tn']} {metrics['fp']}] [{metrics['fn']} {metrics['tp']}]]\n"
)
print(classification_report(y_test, predictions, target_names=["legit", "abusive"]))
print(f"Brier score (lower is better): "
      f"{brier_score_loss(y_test, probabilities):.3f}")
print(f"False positives: {metrics['fp']} out of {metrics['tn'] + metrics['fp']} "
      f"legit cases ({metrics['false_positive_rate']:.1%}).")
print(f"False negatives: {metrics['fn']} out of {metrics['fn'] + metrics['tp']} "
      f"abusive cases ({metrics['miss_rate']:.1%}).\n")

print("=== Customer-impact-constrained policy ===")
print(f"Best threshold: {best_policy['threshold']:.0%}")
print(f"Validation recall: {best_policy['recall']:.3f}")
print(f"Validation false-positive rate: {best_policy['false_positive_rate']:.1%}")
print("Threshold selected on validation data; metrics below use untouched test data.")
print("Policy constraint: false-positive rate <= 10%.\n")

shap_values = shap_values_for(explanation_model, X_test)
plt.figure()
import shap
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.title("Feature importance (mean |SHAP value|) - Return-Risk Scorer")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP value distribution per feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary_beeswarm.png", dpi=150)
plt.close()

ablation_table = feature_ablation(model, X_test, y_test, policy_threshold)
print("=== Largest F1 drops when a signal is removed ===")
print(ablation_table[["feature", "f1_drop"]].head(3).to_string(index=False))
print(f"\nSaved plots to: {OUTPUT_DIR}")