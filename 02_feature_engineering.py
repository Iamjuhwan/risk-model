import pandas as pd
import numpy as np

df = pd.read_csv("./transactions_labeled.csv", parse_dates=["datetime"])
customers = pd.read_csv("./customers.csv")

df = df.sort_values(["customer_id", "datetime"]).reset_index(drop=True)
df["_orig_idx"] = df.index  # stable key to merge engineered features back

# ---------- Feature 1: Amount z-score relative to customer's own prior history ----------
def expanding_zscore(group):
    amounts = group["amount"].values
    z = np.zeros(len(amounts))
    for i in range(len(amounts)):
        if i < 3:
            z[i] = 0.0  # not enough history yet
        else:
            hist = amounts[:i]
            mean = hist.mean()
            std = hist.std() if hist.std() > 0 else 1.0
            z[i] = (amounts[i] - mean) / std
    return pd.DataFrame({"_orig_idx": group["_orig_idx"].values, "amount_zscore": z})

zscore_df = df.groupby("customer_id", group_keys=False)[["_orig_idx", "amount"]].apply(expanding_zscore)
df = df.merge(zscore_df, on="_orig_idx", how="left")

# ---------- Feature 2: New device flag ----------
cust_primary_device = customers.set_index("customer_id")["primary_device"].to_dict()
df["is_new_device"] = (df["device_id"] != df["customer_id"].map(cust_primary_device)).astype(int)

# ---------- Feature 3: Region mismatch ----------
cust_home_region = customers.set_index("customer_id")["home_region"].to_dict()
df["region_mismatch"] = (df["transaction_region"] != df["customer_id"].map(cust_home_region)).astype(int)

# ---------- Feature 4: Odd hour flag (midnight - 5am) ----------
df["hour"] = df["datetime"].dt.hour
df["is_odd_hour"] = df["hour"].between(0, 5).astype(int)

# ---------- Feature 5: Transaction velocity (count in last 1h / 24h per customer) ----------
def velocity_features(group):
    times = group["datetime"].values
    count_1h = np.zeros(len(times), dtype=int)
    count_24h = np.zeros(len(times), dtype=int)
    for i in range(len(times)):
        t = times[i]
        w1 = t - np.timedelta64(1, "h")
        w24 = t - np.timedelta64(24, "h")
        count_1h[i] = np.sum((times >= w1) & (times < t))
        count_24h[i] = np.sum((times >= w24) & (times < t))
    return pd.DataFrame({
        "_orig_idx": group["_orig_idx"].values,
        "txn_count_last_1h": count_1h,
        "txn_count_last_24h": count_24h,
    })

vel_df = df.groupby("customer_id", group_keys=False)[["_orig_idx", "datetime"]].apply(velocity_features)
df = df.merge(vel_df, on="_orig_idx", how="left")

# ---------- Feature 6: Account age ----------
cust_account_age = customers.set_index("customer_id")["account_age_days"].to_dict()
df["account_age_days"] = df["customer_id"].map(cust_account_age)

df = df.drop(columns=["_orig_idx"]).sort_values("datetime").reset_index(drop=True)
df.to_csv("./transactions_features.csv", index=False)

print("Feature engineering complete. Shape:", df.shape)
print()
print("Feature averages by fraud label:")
print(df.groupby("is_fraud")[["amount_zscore", "is_new_device", "region_mismatch",
                                "is_odd_hour", "txn_count_last_1h", "txn_count_last_24h"]].mean())