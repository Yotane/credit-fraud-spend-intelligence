import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Keep prophet_residual for fraud (target is is_fraud, not amt, so no leakage)
FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "prophet_residual"
]

TARGET = "is_fraud"

STUDY_NAME = "lightgbm_fraud_tpe"
STORAGE = "sqlite:///studies/lightgbm_fraud_tpe.db"
N_TRIALS = 50
TIMEOUT = 3600


def objective(trial, df):
    df = df.copy()
    
    # split first, fix: use df[TARGET] not y for stratify
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
    base_scale = n_legit / n_fraud
    scale_multiplier = trial.suggest_float("scale_multiplier", 0.5, 2.0)
    scale_pos_weight = base_scale * scale_multiplier
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)]
    )
    
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    
    return f1


def run_study(df):
    Path("studies").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=1),
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print(f"\nBest trial:")
    print(f"  F1 Score: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    return study


if __name__ == "__main__":
    from data.loader import load_transactions
    from features.engineering import add_features
    
    print("Loading data...")
    df = load_transactions()
    df = add_features(df)
    
    print(f"Running Optuna study: {STUDY_NAME}")
    print(f"  Trials: {N_TRIALS}, Timeout: {TIMEOUT}s")
    
    study = run_study(df)
    print("Study complete. Results saved to studies/")