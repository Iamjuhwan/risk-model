# Fraud & Risk Scoring Model

An end-to-end fraud detection project comparing a rule-based scoring system against machine learning models (Logistic Regression, Random Forest) on simulated fintech transaction data. Built to demonstrate feature engineering, model evaluation under class imbalance, and honest analysis of trade-offs — not just a working classifier.

> **Note:** Uses a synthetically generated, labeled dataset simulating fintech transaction fraud patterns. Built for portfolio/demonstration purposes.

## Business Problem
Fintechs need to catch fraudulent transactions (account takeovers, compromised sessions) without blocking legitimate customers doing unusual-but-normal things (traveling, upgrading their phone, making a large one-off purchase). This is a genuinely hard problem: fraud is rare (well under 1% of transactions), and naive rules that trigger on any single "suspicious" signal generate too many false positives to be usable.

This project simulates that exact tension: 600 customers, ~38,600 transactions over 9 months, a realistic 0.79% fraud rate, obvious fraud (account-takeover bursts), subtle fraud (1-2 transactions, familiar device), and legitimate anomalies (real travel, real device upgrades, real big purchases) designed to fool naive rules.

## Tools Used
- **Python** — data generation, feature engineering, modeling
- **scikit-learn** — Logistic Regression, Random Forest, evaluation metrics (precision, recall, F1, ROC-AUC)
- **pandas** — time-series feature engineering (expanding z-scores, rolling transaction velocity)
- **matplotlib** — ROC and precision-recall curves
- **Excel (openpyxl)** — final report with tables, charts, conditional formatting

## Project Structure
```
├── 01_generate_data.py                    # synthetic transaction generator with realistic fraud patterns
├── 02_feature_engineering.py              # builds the 7 fraud-risk features
├── 03_rule_based_scoring.py               # point-based rule system + evaluation
├── 04_train_models.py                     # trains Logistic Regression + Random Forest, time-based split
├── 05_generate_charts.py                  # ROC and precision-recall curve images
├── 06_build_report.py                     # assembles the final Excel report
├── transactions_labeled.csv               # raw generated data
├── customers.csv                          # customer dimension
├── model_comparison.csv                   # rule-based vs LR vs RF metrics
├── OPay_Fraud_Risk_Scoring_Model.xlsx      # final deliverable
└── README.md
```

## Approach
1. **Feature engineering** (7 features): amount z-score (deviation from the customer's own historical average, computed with expanding statistics to avoid lookahead bias), new-device flag, region mismatch vs. home region, odd-hour flag, transaction velocity (count in last 1h/24h), account age.
2. **Rule-based scoring** — a transparent point system (e.g., new device = 30 points, amount anomaly = 25 points) with named risk tiers (Minimal/Low/Medium/High). This is the kind of system a non-technical ops team can audit and trust.
3. **ML models** — Logistic Regression (interpretable coefficients) and Random Forest, trained with **class_weight="balanced"** to handle the severe class imbalance, and evaluated on a **time-based train/test split** (train on the first ~75% chronologically, test on the most recent ~25%) to simulate real deployment rather than leaking future information into training.
4. **Comparison** — all three approaches evaluated on the identical held-out test period.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Rule-Based (score ≥ 50) | ~1.00 | ~0.97 | ~0.99 | — |
| Logistic Regression | ~0.72 | 1.00 | ~0.84 | ~1.000 |
| Random Forest | ~0.75 | 1.00 | ~0.86 | ~1.000 |

*(See `model_comparison.csv` / the Excel report for exact figures from your run — synthetic data regenerates with the same patterns but not identical numbers each time.)*

**Key finding:** the rule-based system is highly precise but misses some subtler fraud; the ML models catch every fraud case in the test period but flag more false positives. This is a genuine precision/recall trade-off — not a flaw in either approach — and the "right" choice depends on the actual cost a business assigns to a missed fraud vs. an inconvenienced legitimate customer.

**Top predictive features** (consistent across both models): amount z-score, new-device flag, and region mismatch — transaction velocity matters less than expected once those three are accounted for.

**Honest limitation, stated directly in the report:** performance here (ROC-AUC > 0.999) is unusually strong because the synthetic fraud signal was deliberately made learnable. Real transaction data has more overlap between fraud and legitimate behavior; this project demonstrates the method and pipeline, not a production-ready accuracy claim.

## How to Reproduce
```
python3 01_generate_data.py
python3 02_feature_engineering.py
python3 03_rule_based_scoring.py
python3 04_train_models.py
python3 05_generate_charts.py
python3 06_build_report.py
```
Each script writes its output as a CSV consumed by the next step. Running all six in order regenerates the full pipeline and the final Excel report.

## Author
[Your Name] — Data Analyst
[LinkedIn] · [Portfolio site] · [Email]
