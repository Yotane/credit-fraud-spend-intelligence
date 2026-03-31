import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Keep prophet_residual for fraud (target is is_fraud, not amt, so no leakage)
FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_zscore"
]

TARGET = "is_fraud"

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.0604,
    "num_leaves": 213,
    "max_depth": 13,
    "min_child_samples": 14,
    "subsample": 0.793,
    "colsample_bytree": 0.517,
    "reg_alpha": 1.17e-08,
    "reg_lambda": 2.54e-07,
}


def train(df: pd.DataFrame, params: dict = None) -> tuple:
    # split first, then encode
    X_train, X_val, y_train, y_val = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=1, stratify=df[TARGET]
    )
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        encoders[col] = le
    
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale = (n_legit / n_fraud) * 0.758  # scale_multiplier from Optuna
    
    p = params if params else PARAMS
    model = lgb.LGBMClassifier(**p, scale_pos_weight=scale)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)]
    )
    
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    
    print(f"LightGBM Fraud -- F1: {f1:.4f}")
    print(classification_report(y_val, preds, target_names=["Legitimate", "Fraud"]))
    return model, encoders, {"f1": f1}


def save(model, path: str = "models/lgbm_fraud.pkl"):
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    from data.loader import load_transactions
    from features.engineering import add_features
    
    df = load_transactions()
    df = add_features(df)
    model, encoders, metrics = train(df)
    save(model)
    print("Model saved.")