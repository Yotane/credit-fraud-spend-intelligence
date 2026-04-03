import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"

CATEGORICAL_FEATURES = ["gender", "category", "job", "age_group", "city_size"]

STUDY_NAME = "lightgbm_spend_tpe"
STORAGE = "sqlite:///studies/lightgbm_spend_tpe.db"
N_TRIALS = 50
TIMEOUT = 3600
RANDOM_STATE = 1


def objective(trial, X_train, X_val, y_train, y_val, cat_indices):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 1000,
        "max_depth": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "categorical_feature": cat_indices,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    
    return rmse


def run_study(df):
    Path("studies").mkdir(exist_ok=True)
    
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    cat_cols = CATEGORICAL_FEATURES
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        direction="minimize",
        load_if_exists=False
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val, cat_indices),
        n_trials=N_TRIALS,
        timeout=TIMEOUT
    )
    
    print(f"\nBest trial:")
    print(f"  RMSE: {study.best_value:.4f}")
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
    print(f"  Random state: {RANDOM_STATE} (fixed for all trials)")
    
    study = run_study(df)
    print("Study complete. Results saved to studies/")