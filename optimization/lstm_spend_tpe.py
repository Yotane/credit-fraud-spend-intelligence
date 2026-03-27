import optuna
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"

STUDY_NAME = "tabnet_spend_tpe"
STORAGE = "sqlite:///studies/tabnet_spend_tpe.db"
N_TRIALS = 5
TIMEOUT = 7200

BATCH_SIZE = 2048
EPOCHS = 15


def objective(trial, df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=1)
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col] = val_df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )
        encoders[col] = le
    
    X_train = train_df[FEATURES].values
    X_val = val_df[FEATURES].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    params = {
        "n_d": trial.suggest_int("n_d", 32, 128),
        "n_a": trial.suggest_int("n_a", 32, 128),
        "n_steps": trial.suggest_int("n_steps", 3, 5),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True)),
        "verbose": 0,
    }
    
    model = TabNetRegressor(**params)
    
    model.fit(
        X_train=X_train,
        y_train=y_train.reshape(-1, 1),
        batch_size=BATCH_SIZE,
        max_epochs=EPOCHS,
        patience=10,
        from_unsupervised=None,
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print(f"\nBest trial:")
    if study.best_trial:
        print(f"  RMSE: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
    else:
        print("  No successful trials completed")
    
    return study


if __name__ == "__main__":
    from data.loader import load_transactions
    from features.engineering import add_features
    
    print("Loading data...")
    df = load_transactions()
    df = add_features(df)
    
    print(f"Running Optuna study: {STUDY_NAME}")
    print(f"  Trials: {N_TRIALS}, Timeout: {TIMEOUT}s")
    print(f"  Epochs per trial: {EPOCHS}")
    
    study = run_study(df)
    print("Study complete. Results saved to studies/")