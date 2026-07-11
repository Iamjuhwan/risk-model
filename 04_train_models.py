import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./transactions_rule_scored.csv", parse_dates=["datetime"])

feature_cols = ["amount_zscore", "is_new_device", "region_mismatch", "is_odd_hour",
                 "txn_count_last_1h", "txn_count_last_24h", "account_age_days"]

df = df.sort_values("datetime").reset_index(drop=True)

# ---------- Time-based split (train on earlier data, test on later -- avoids leakage) ----------
split_date = df["datetime"].quantile(0.75)
train = df[df["datetime"] < split_date].copy()
test = df[df["datetime"] >= split_date].copy()

print(f"Train period: {train['datetime'].min()} to {train['datetime'].max()} ({len(train)} rows, {train['is_fraud'].sum()} fraud)")
print(f"Test period:  {test['datetime'].min()} to {test['datetime'].max()} ({len(test)} rows, {test['is_fraud'].sum()} fraud)")

X_train, y_train = train[feature_cols], train["is_fraud"]
X_test, y_test = test[feature_cols], test["is_fraud"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Logistic Regression (interpretable, coefficients = feature importance) ----------
logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ---------- Random Forest (comparison model) ----------
rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                             random_state=42, min_samples_leaf=5)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]

def evaluate(y_true, proba, threshold, label):
    pred = (proba >= threshold).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    auc = roc_auc_score(y_true, proba)
    cm = confusion_matrix(y_true, pred)
    print(f"\n=== {label} (threshold={threshold}) ===")
    print(f"Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"Confusion matrix [[TN FP][FN TP]]:\n{cm}")
    return {"model": label, "threshold": threshold, "precision": round(p,4),
            "recall": round(r,4), "f1_score": round(f1,4), "roc_auc": round(auc,4)}

results = []
results.append(evaluate(y_test, proba_lr, 0.5, "Logistic Regression"))
results.append(evaluate(y_test, proba_rf, 0.5, "Random Forest"))

# rule-based on same test period for fair comparison
rule_test = test["rule_flagged"]
p_rule = precision_score(y_test, rule_test, zero_division=0)
r_rule = recall_score(y_test, rule_test, zero_division=0)
f1_rule = f1_score(y_test, rule_test, zero_division=0)
print(f"\n=== Rule-Based (same test period) ===")
print(f"Precision: {p_rule:.3f} | Recall: {r_rule:.3f} | F1: {f1_rule:.3f}")
results.append({"model": "Rule-Based", "threshold": "score>=50", "precision": round(p_rule,4),
                 "recall": round(r_rule,4), "f1_score": round(f1_rule,4), "roc_auc": None})

pd.DataFrame(results).to_csv("./model_comparison.csv", index=False)

# ---------- Feature importance ----------
lr_importance = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": logreg.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)

rf_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

lr_importance.to_csv("./logreg_feature_importance.csv", index=False)
rf_importance.to_csv("./rf_feature_importance.csv", index=False)

print("\nLogistic Regression coefficients (standardized):")
print(lr_importance)
print("\nRandom Forest feature importance:")
print(rf_importance)

# ---------- Save scored test set with model probabilities for the Excel report ----------
test_scored = test.copy()
test_scored["fraud_probability_logreg"] = proba_lr
test_scored["fraud_probability_rf"] = proba_rf
test_scored.to_csv("./test_set_scored.csv", index=False)

print("\nSaved model_comparison.csv, feature importance files, and test_set_scored.csv")