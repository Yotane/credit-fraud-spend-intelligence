import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import joblib
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "prophet_residual"
]

TARGET = "amt"

PARAMS = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=2e-2),
    "verbose": 0,
}

BATCH_SIZE = 1024
EPOCHS = 50


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    y = df[TARGET].values
    
    return X, y, encoders, scaler


def train(df: pd.DataFrame, params: dict = None) -> tuple:
    X, y, encoders, scaler = prepare_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    
    p = params if params else PARAMS
    model = TabNetRegressor(**p)
    
    model.fit(
        X_train=X_train,
        y_train=y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        batch_size=BATCH_SIZE,
        max_epochs=EPOCHS,
        patience=10,
        from_unsupervised=None,
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)
    
    print(f"TabNet Spend -- RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return model, encoders, scaler, {"rmse": rmse, "mae": mae}


def save(model, encoders, scaler, path: str = "models/tabnet_spend.pkl"):
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders, "scaler": scaler}, path)


if __name__ == "__main__":
    from data.loader import load_transactions
    from features.engineering import add_features
    
    df = load_transactions()
    df = add_features(df)
    model, encoders, scaler, metrics = train(df)
    save(model, encoders, scaler)
    print("Model saved.")