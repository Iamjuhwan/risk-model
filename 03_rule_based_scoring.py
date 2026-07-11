import pandas as pd
import numpy as np

df = pd.read_csv("./transactions_features.csv", parse_dates=["datetime"])

df["region_mismatch"] = df["region_mismatch"].fillna(0).astype(int)

# ---------- Rule-based point system ----------
# Points assigned per risk signal, thresholds chosen to be explainable to a non-technical ops team
df["pts_amount"] = np.where(df["amount_zscore"] > 3, 25, 0)
df["pts_new_device"] = np.where(df["is_new_device"] == 1, 30, 0)
df["pts_odd_hour"] = np.where(df["is_odd_hour"] == 1, 15, 0)
df["pts_velocity"] = np.where(df["txn_count_last_1h"] >= 2, 30, 0)
df["pts_region"] = np.where(df["region_mismatch"] == 1, 20, 0)

df["rule_score"] = (df["pts_amount"] + df["pts_new_device"] + df["pts_odd_hour"] +
                     df["pts_velocity"] + df["pts_region"])

# Risk tiers
def tier(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 15:
        return "Low"
    else:
        return "Minimal"

df["rule_risk_tier"] = df["rule_score"].apply(tier)

# ---------- Evaluate rule-based approach ----------
df["rule_flagged"] = (df["rule_score"] >= 50).astype(int)

tp = ((df["rule_flagged"] == 1) & (df["is_fraud"] == 1)).sum()
fp = ((df["rule_flagged"] == 1) & (df["is_fraud"] == 0)).sum()
fn = ((df["rule_flagged"] == 0) & (df["is_fraud"] == 1)).sum()
tn = ((df["rule_flagged"] == 0) & (df["is_fraud"] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=== Rule-Based Scoring Results (flag threshold: score >= 50) ===")
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives:  {tn}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print()
print("Risk tier distribution:")
print(df["rule_risk_tier"].value_counts())

df.to_csv("./transactions_rule_scored.csv", index=False)

rule_metrics = pd.DataFrame([{
    "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
    "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4)
}])
rule_metrics.to_csv("./rule_based_metrics.csv", index=False)