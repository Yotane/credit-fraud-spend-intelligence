import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "prophet_residual"
]

TARGET = "amt"

STUDY_NAME = "xgboost_spend_tpe"
STORAGE = "sqlite:///studies/xgboost_spend_tpe.db"
N_TRIALS = 50
TIMEOUT = 3600


def objective(trial, df):
    df = df.copy()
    
    X_train, X_val, y_train, y_val = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=1
    )
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        encoders[col] = le
    
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
    }
    
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50, verbose=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    
    return rmse


def run_study(df):
    Path("studies").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=1),
        pruner=optuna.pruners.MedianPruner(),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS, timeout=TIMEOUT)
    
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
    
    study = run_study(df)
    print("Study complete. Results saved to studies/")