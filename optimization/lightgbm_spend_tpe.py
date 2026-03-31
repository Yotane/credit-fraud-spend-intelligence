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
N_TRIALS = 100
TIMEOUT = 7200


def objective(trial, df):
    df = df.copy()
    
    X_train, X_val, y_train, y_val = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=trial.number
    )
    
    cat_cols = CATEGORICAL_FEATURES
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        encoders[col] = le
    
    cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 511),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 100.0, log=True),
        "feature_fraction_by_node": trial.suggest_float("feature_fraction_by_node", 0.4, 1.0),
        "categorical_feature": cat_indices,
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0)
        ]
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    
    return rmse


def run_study(df):
    Path("studies").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=2000),
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