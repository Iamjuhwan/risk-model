import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

test = pd.read_csv("./test_set_scored.csv")
y_true = test["is_fraud"]

fpr_lr, tpr_lr, _ = roc_curve(y_true, test["fraud_probability_logreg"])
fpr_rf, tpr_rf, _ = roc_curve(y_true, test["fraud_probability_rf"])

plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression", color="#00A651", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label="Random Forest", color="#1B1F3B", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Fraud Detection Models (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./roc_curve.png", dpi=150)
plt.close()

prec_lr, rec_lr, _ = precision_recall_curve(y_true, test["fraud_probability_logreg"])
prec_rf, rec_rf, _ = precision_recall_curve(y_true, test["fraud_probability_rf"])

plt.figure(figsize=(6, 5))
plt.plot(rec_lr, prec_lr, label="Logistic Regression", color="#00A651", linewidth=2)
plt.plot(rec_rf, prec_rf, label="Random Forest", color="#1B1F3B", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Fraud Detection Models (Test Set)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("./precision_recall_curve.png", dpi=150)
plt.close()

print("Charts saved: roc_curve.png, precision_recall_curve.png")