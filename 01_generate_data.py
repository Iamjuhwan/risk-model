import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(123)

# ---------- Customers ----------
n_customers = 600
regions = ["South West", "South South", "South East", "North Central", "North West", "North East"]
region_weights = [0.38, 0.16, 0.12, 0.14, 0.12, 0.08]
devices_pool = [f"DEV-{i:05d}" for i in range(3000)]
channels = ["Mobile App", "USSD", "POS Terminal", "Web"]

customers = pd.DataFrame({
    "customer_id": [f"CUST-{100000+i}" for i in range(n_customers)],
    "home_region": rng.choice(regions, size=n_customers, p=region_weights),
    "avg_amount": rng.lognormal(mean=8.5, sigma=0.8, size=n_customers).round(2),  # typical spend level
    "primary_device": rng.choice(devices_pool, size=n_customers, replace=False),
    "account_age_days": rng.integers(10, 1200, size=n_customers),
})

# ---------- Generate transactions per customer ----------
start_date = datetime(2025, 10, 1)
end_date = datetime(2026, 6, 30)
total_days = (end_date - start_date).days

rows = []
txn_counter = 900000
fraud_customer_flags = {}

for _, cust in customers.iterrows():
    cid = cust["customer_id"]
    home_region = cust["home_region"]
    avg_amt = cust["avg_amount"]
    device = cust["primary_device"]
    # normal transaction frequency: 1-3 per week on average
    n_txns = int(rng.poisson(lam=total_days / 7 * rng.uniform(0.8, 2.5)))
    n_txns = max(3, n_txns)
    txn_days = np.sort(rng.uniform(0, total_days, size=n_txns))
    for td in txn_days:
        dt = start_date + timedelta(days=float(td))
        # normal hour distribution: weighted toward daytime
        hour = int(np.clip(rng.normal(14, 4), 0, 23))
        dt = dt.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))
        amount = round(max(100, rng.normal(avg_amt, avg_amt * 0.35)), 2)
        channel = rng.choice(channels, p=[0.55, 0.22, 0.18, 0.05])
        rows.append({
            "transaction_id": f"TXN-{txn_counter}",
            "customer_id": cid,
            "datetime": dt,
            "amount": amount,
            "channel": channel,
            "device_id": device,
            "transaction_region": home_region,
            "is_fraud": 0,
        })
        txn_counter += 1

df = pd.DataFrame(rows)
df = df.sort_values("datetime").reset_index(drop=True)

# ---------- Inject fraud patterns: account takeover bursts ----------
fraud_customers = rng.choice(customers["customer_id"], size=45, replace=False)
fraud_rows = []
for cid in fraud_customers:
    cust = customers[customers["customer_id"] == cid].iloc[0]
    home_region = cust["home_region"]
    avg_amt = cust["avg_amount"]
    other_regions = [r for r in regions if r != home_region]
    fraud_region = rng.choice(other_regions)
    fraud_device = rng.choice(devices_pool)  # a device never seen for this customer
    # pick a random burst start time
    burst_start_day = rng.uniform(20, total_days - 20)
    burst_start = start_date + timedelta(days=float(burst_start_day))
    # odd hour: late night
    hour = int(rng.choice([1, 2, 3, 4, 23]))
    burst_start = burst_start.replace(hour=hour, minute=int(rng.integers(0, 60)))
    n_burst_txns = int(rng.integers(4, 9))  # velocity spike
    for i in range(n_burst_txns):
        dt = burst_start + timedelta(minutes=int(rng.integers(1, 12)) * i)
        amount = round(avg_amt * rng.uniform(3, 9), 2)  # much bigger than usual
        fraud_rows.append({
            "transaction_id": f"TXN-{txn_counter}",
            "customer_id": cid,
            "datetime": dt,
            "amount": amount,
            "channel": rng.choice(["Mobile App", "Web"]),
            "device_id": fraud_device,
            "transaction_region": fraud_region,
            "is_fraud": 1,
        })
        txn_counter += 1

fraud_df = pd.DataFrame(fraud_rows)
df = pd.concat([df, fraud_df], ignore_index=True)

# ---------- Legit anomalies: real customers doing unusual-but-legitimate things ----------
# These are designed to trip individual rules (new device, region mismatch, high amount)
# WITHOUT being fraud -- this is what makes the detection problem realistic instead of trivial.
legit_anomaly_customers = rng.choice(
    [c for c in customers["customer_id"] if c not in fraud_customers], size=70, replace=False
)
legit_anomaly_rows = []
for cid in legit_anomaly_customers:
    cust = customers[customers["customer_id"] == cid].iloc[0]
    home_region = cust["home_region"]
    avg_amt = cust["avg_amount"]
    anomaly_type = rng.choice(["travel", "new_phone", "big_purchase"], p=[0.4, 0.35, 0.25])
    day = rng.uniform(20, total_days - 20)
    dt = start_date + timedelta(days=float(day))
    hour = int(np.clip(rng.normal(15, 4), 6, 22))  # normal daytime hour, not odd hour
    dt = dt.replace(hour=hour, minute=int(rng.integers(0, 60)))

    if anomaly_type == "travel":
        # customer genuinely travels: different region, familiar device, normal amount
        other_regions = [r for r in regions if r != home_region]
        region = rng.choice(other_regions)
        device = cust["primary_device"]
        amount = round(max(100, rng.normal(avg_amt, avg_amt * 0.35)), 2)
    elif anomaly_type == "new_phone":
        # customer legitimately upgrades device: home region, new device, normal amount
        region = home_region
        device = rng.choice(devices_pool)
        amount = round(max(100, rng.normal(avg_amt, avg_amt * 0.35)), 2)
    else:  # big_purchase
        # customer makes a real one-off large purchase: home region, familiar device, high amount
        region = home_region
        device = cust["primary_device"]
        amount = round(avg_amt * rng.uniform(3, 5), 2)

    legit_anomaly_rows.append({
        "transaction_id": f"TXN-{txn_counter}",
        "customer_id": cid,
        "datetime": dt,
        "amount": amount,
        "channel": rng.choice(channels, p=[0.55, 0.22, 0.18, 0.05]),
        "device_id": device,
        "transaction_region": region,
        "is_fraud": 0,
    })
    txn_counter += 1

legit_anomaly_df = pd.DataFrame(legit_anomaly_rows)

# ---------- Subtle fraud: single/double transactions, no velocity burst, familiar device ----------
# Simulates a compromised session rather than a full account takeover -- harder to catch on rules alone.
subtle_fraud_customers = rng.choice(
    [c for c in customers["customer_id"] if c not in fraud_customers and c not in legit_anomaly_customers],
    size=20, replace=False
)
subtle_fraud_rows = []
for cid in subtle_fraud_customers:
    cust = customers[customers["customer_id"] == cid].iloc[0]
    home_region = cust["home_region"]
    avg_amt = cust["avg_amount"]
    day = rng.uniform(20, total_days - 20)
    dt = start_date + timedelta(days=float(day))
    hour = int(np.clip(rng.normal(15, 5), 0, 23))  # can occur at any hour, not necessarily odd
    dt = dt.replace(hour=hour, minute=int(rng.integers(0, 60)))
    n_txns = int(rng.integers(1, 3))  # just 1-2 transactions, no burst
    for i in range(n_txns):
        amount = round(avg_amt * rng.uniform(1.8, 3.5), 2)  # elevated but not extreme
        subtle_fraud_rows.append({
            "transaction_id": f"TXN-{txn_counter}",
            "customer_id": cid,
            "datetime": dt + timedelta(minutes=int(rng.integers(5, 40)) * i),
            "amount": amount,
            "channel": rng.choice(["Mobile App", "Web"]),
            "device_id": cust["primary_device"],  # same device -- session compromise, not new device
            "transaction_region": home_region,  # same region
            "is_fraud": 1,
        })
        txn_counter += 1

subtle_fraud_df = pd.DataFrame(subtle_fraud_rows)

df = pd.concat([df, legit_anomaly_df, subtle_fraud_df], ignore_index=True)
df = df.sort_values(["customer_id", "datetime"]).reset_index(drop=True)

print("Total transactions:", len(df))
print("Fraud transactions:", df["is_fraud"].sum(), f"({df['is_fraud'].mean()*100:.2f}%)")

df.to_csv("./transactions_labeled.csv", index=False)
customers.to_csv("./customers.csv", index=False)