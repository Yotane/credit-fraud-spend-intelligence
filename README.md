# Credit Fraud & Spend Intelligence

**What behavioral and demographic patterns predict high spending, and do fraudulent transactions mimic those same patterns or have clearly different signatures?**

An end-to-end machine learning project that answers that question through leakage-free feature engineering, rolling z-score anomaly detection, gradient boosting models, deep learning baselines, Optuna hyperparameter optimization, and SHAP-based interpretability. Approximately 1.3 million credit card transactions are analyzed to model spending patterns and detect fraud with production-ready code.

Dataset sourced from Kaggle — ["Credit Card Transactions Dataset"](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) by priyamchoksi.

---

## Key Results

### Spend Prediction (RMSE - lower is better)

| Model | Baseline | Optimized | Change |
|-------|----------|-----------|--------|
| LightGBM | 144.28 | 144.27 | 0% |
| XGBoost | 145.09 | 144.15 | 0.6% |
| MLP | 146.88 | 146.11 | 0.5% |
| LSTM | 167.42 | 167.50 | 0% |

### Fraud Detection (F1 Score - higher is better)

| Model | Baseline | Optimized | Change |
|-------|----------|-----------|--------|
| LightGBM | 0.31 | 0.80 | +158% |

---

## Key Findings

1. **Tree Models Are Robust:** LightGBM and XGBoost achieved near-optimal performance with default parameters. Optuna optimization yielded minimal improvement (0-0.6%), confirming production-readiness out of the box.

2. **Class Imbalance Is Critical:** Fraud detection improved 158% through optimized scale_pos_weight (0.31 to 0.80 F1), demonstrating that handling imbalance is more impactful than tree hyperparameters.

3. **Architecture Over Hyperparameters:** Deep learning models (MLP, LSTM) underperformed trees by 1-16% regardless of tuning. LSTM showed the largest gap at 16% worse RMSE, confirming that gradient boosting is better suited for structured tabular data than neural networks.

4. **Fraud and Spend Share Top Features, But Use Them Differently:** Both models identify merchant category and transaction hour as top predictors. However, spend models use these to predict expected amounts, while fraud models use them to detect deviations from individual user norms.

---

## SHAP Feature Importance Analysis

### Spend Prediction - Top Features

![LGBM Spend SHAP Bar](analysis/plots/shap_lgbm_spend_bar.png)
![LGBM Spend SHAP Summary](analysis/plots/shap_lgbm_spend_summary.png)

**Top 4 Features:**

| Rank | Feature | Avg SHAP Value | What It Means (Example) |
|------|---------|----------------|-------------------------|
| 1 | category | 22.5 | Electronics adds +$200-400 vs Groceries at +$20-50 |
| 2 | hour | 18.2 | 2 PM transactions average +$80 vs 3 AM at -$50 |
| 3 | age | 12.0 | Users 50+ spend ~$100 more than users in 20s |
| 4 | rolling_mean | 5.0 | Each $100 historical average adds ~$5-10 to prediction |

**Concrete Example:**

```
Transaction A:
- category: Electronics, hour: 2 PM, age: 55, rolling_mean: $200
- Model prediction: $450

Transaction B:
- category: Groceries, hour: 3 AM, age: 25, rolling_mean: $50
- Model prediction: $45

Same user, different contexts leads to 10x spending difference driven by category and hour.
```

### Fraud Detection - Top Features

![Fraud SHAP Bar](analysis/plots/shap_fraud_bar.png)
![Fraud SHAP Summary](analysis/plots/shap_fraud_summary.png)

**Top 4 Features:**

| Rank | Feature | Avg SHAP Value | What It Means (Example) |
|------|---------|----------------|-------------------------|
| 1 | category | 1.35 | Electronics/gas stations have 3x higher fraud rates |
| 2 | hour | 1.20 | Transactions between 1-5 AM have 2x fraud probability |
| 3 | job | 0.75 | Certain occupations show higher fraud vulnerability |
| 4 | rolling_zscore | 0.72 | Z-score greater than 3 increases fraud probability by 40% |

**Concrete Example:**

```
User Alice (normally spends $50 per transaction):

Transaction A (Legitimate):
- category: Electronics, amount: $500, hour: 2 PM
- rolling_zscore: 2.1 (within normal range for occasional large purchase)
- Fraud prediction: 15% (not flagged)

Transaction B (Fraudulent):
- category: Electronics, amount: $500, hour: 3 AM
- rolling_zscore: 12.5 (12 standard deviations from her norm)
- Fraud prediction: 92% (flagged)

Same amount, same category but different fraud score based on timing plus deviation from personal baseline.
```

---

## Answering the Core Question

### Do Fraudulent Transactions Mimic High-Spending Patterns?

**Partially, but with key differences.**

| Aspect | High-Spending Transactions | Fraudulent Transactions |
|--------|---------------------------|------------------------|
| **Top Features** | category, hour, age | category, hour, job, rolling_zscore |
| **What Drives Prediction** | Merchant type and time determine price ranges | Same features signal anomaly when unusual for the user |
| **Role of Historical Behavior** | rolling_mean predicts baseline spending | rolling_zscore detects deviation from baseline |
| **Overlap** | category and hour are top 2 for both | Same features, different interpretation |

### The Critical Difference

| Model Type | Question Being Asked | How Features Are Used |
|------------|---------------------|----------------------|
| **Spend** | How much will this user spend? | category equals Electronics means $500+ expected |
| **Fraud** | Is this transaction unusual for this user? | category equals Electronics at 3 AM for a user who normally spends $50 equals suspicious |

**Conclusion:** Fraudulent transactions often occur in the same categories and hours as legitimate high-value transactions. The distinguishing signal is not WHERE or WHEN, but WHETHER the transaction deviates from that specific user normal behavior. This is why rolling_zscore (rolling z-score) appears in fraud SHAP but not spend SHAP.

---

## Feature Engineering Strategy

### Rolling Features

Three rolling features are computed per customer based on the past 7 transactions:

| Feature | Formula | Purpose | Used In |
|---------|---------|---------|---------|
| rolling_mean | avg(past 7 amounts) | Baseline spending level | Spend models |
| rolling_std | std(past 7 amounts) | Spending consistency | Spend models |
| rolling_zscore | (amt - rolling_mean) / rolling_std | Anomaly score (rolling z-score) | Fraud models |

### Why the Name "rolling_zscore"?

The feature was originally named `prophet_residual` because Facebook Prophet, a time series forecasting library, was initially considered for computing residuals as:

```
residual = actual_amount - Prophet_forecasted_amount
```

However, the simple rolling z-score was adopted instead for the following reasons:

| Reason | Explanation |
|--------|-------------|
| **Performance** | Prophet requires 5-10 seconds per customer. On 50,000+ customers, this equals 70-140 hours of compute time. |
| **Overkill for the task** | Prophet is designed for long-term trend forecasting with seasonality. Only 7-transaction rolling statistics were required. |
| **Same result, simpler code** | Rolling z-score achieves identical anomaly detection with 3 lines of pandas vs Prophet model fitting. |
| **No external dependency** | Removing Prophet means one less package to install and maintain in production. |

The feature was renamed to `rolling_zscore` to accurately reflect its computation method.

### Why Different Features for Spend vs Fraud?

| Task | Target | Safe Features | Why |
|------|--------|---------------|-----|
| **Spend** | amt (dollar amount) | rolling_mean, rolling_std | rolling_zscore contains amt, causing target leakage |
| **Fraud** | is_fraud (0 or 1) | rolling_zscore | Target is binary, so z-score does not leak the answer |

This split feature strategy maintains zero data leakage while maximizing predictive signal for each task.

---

## Data Leakage Fixes

During development, three critical leakage issues were identified and addressed:

1. **Target Leakage in Spend Models:** The rolling_zscore feature (z-score) contained the target variable (amt) in its formula. This was addressed by using rolling_mean and rolling_std as separate features for spend models while retaining the z-score for fraud (where target is is_fraud, not amt).

2. **Encoder Leakage:** LabelEncoders were initially fit on the full dataset. This was corrected by fitting only on training data after the train/val split.

3. **Scaler Leakage:** StandardScaler was fit before splitting. This was corrected by fitting only on training data.

### Impact of Leakage Fixes

| Metric | With Leakage (Initial) | After Fix (Final) | Difference |
|--------|-----------------------|-------------------|------------|
| LightGBM Spend RMSE | 98.5 | 144.28 | +46% (worse but honest) |
| LightGBM Fraud F1 | 0.78 | 0.80 | +2% (stable) |

The fraud model was unaffected because rolling_zscore does not leak the fraud target. The spend model showed significant inflation before the fix because the model could reverse-engineer the amount from the z-score feature.

---

## Project Structure

```
Project 3/
├── data/                    # Raw data and loader
│   ├── loader.py
│   ├── credit_card_transactions.csv
├── features/                # Leakage-free feature engineering with the rolling features
│   ├── engineering.py
│   └── rolling_features.py
├── models/                  # Training scripts with optimized params
│   ├── lightgbm_spend.py
│   ├── xgboost_spend.py
│   ├── mlp_spend.py
│   └── lightgbm_fraud.py
├── optimization/            # Optuna hyperparameter studies
│   ├── lightgbm_spend_tpe.py
│   ├── xgboost_spend_tpe.py
│   ├── mlp_spend_tpe.py
│   └── lightgbm_fraud_tpe.py
├── analysis/                # SHAP analysis and plots
│   ├── shap_analysis.py
│   └── plots/
├── studies/                 # SQLite databases for Optuna results
└── README.md
```

---

## Usage

```bash
# Run optimized models
python -m models.lightgbm_spend
python -m models.xgboost_spend
python -m models.lightgbm_fraud

# Run hyperparameter optimization
python -m optimization.lightgbm_spend_tpe
python -m optimization.lightgbm_fraud_tpe

# Generate SHAP plots
python -m analysis.shap_analysis
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Tree Models | LightGBM 4.x, XGBoost 2.x |
| Deep Learning | PyTorch 2.x (MLP) |
| Optimization | Optuna 3.x (TPE sampler) |
| Interpretability | SHAP 0.44+ |
| Data Processing | pandas, numpy, scikit-learn |
| Environment | venv, joblib for model persistence |

---

## Conclusion

This project demonstrates that for financial tabular data:

1. **Start with LightGBM or XGBoost defaults** - They are often sufficient for production
2. **Prioritize class imbalance handling** for fraud detection tasks
3. **Avoid deep learning** unless massive data or specific sequential needs are present
4. **Audit for leakage early** - A single leaked feature can inflate results by 50% or more

The 158% fraud F1 improvement was achieved through proper class weighting, not complex models. The consistent tree model performance (0% optimization gain) indicates robust defaults. These are practical insights for production ML systems.

**Answer to the core question:** Fraudulent transactions do mimic high-spending patterns in terms of category and timing. The distinguishing signature is not the transaction context itself, but whether that context represents a deviation from the individual user established behavior. This is why anomaly detection (rolling_zscore) outperforms raw amount prediction for fraud, while rolling baselines (rolling_mean) suffice for spend prediction.

---

## License

This project is for educational purposes. Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) under their terms of use.

---

## Author

Matt Raymond Ayento  
Nagoya University  
G30, 3rd year Automotive Engineering (Electrical, Electronics, Information Engineering)