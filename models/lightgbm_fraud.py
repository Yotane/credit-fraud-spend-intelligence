import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size", 
    "prophet_residual"
]

TARGET = "is_fraud"

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}


def prepare_data(df: pd.DataFrame):
    df = df.copy()

    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET]
    return X, y, encoders


def train(df: pd.DataFrame, params: dict = None) -> tuple:
    X, y, encoders = prepare_data(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    # weight fraud cases to compensate for imbalance
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale = n_legit / n_fraud

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