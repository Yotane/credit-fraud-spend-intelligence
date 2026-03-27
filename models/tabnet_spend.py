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
    "rolling_mean", "rolling_std"
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

BATCH_SIZE = 2048
EPOCHS = 20


def train(df: pd.DataFrame, params: dict = None) -> tuple:
    df = df.copy()
    
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
    
    p = params if params else PARAMS
    model = TabNetRegressor(**p)
    
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