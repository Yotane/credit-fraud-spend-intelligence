import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_estimators": 2000,
    "learning_rate": 0.0466,
    "num_leaves": 63,
    "max_depth": 6,
    "min_child_samples": 416,
    "subsample": 0.970,
    "colsample_bytree": 0.723,
    "colsample_bynode": 0.730,
    "reg_alpha": 7.39e-06,
    "reg_lambda": 0.000458,
    "min_child_weight": 1.17e-05,
    "feature_fraction_by_node": 0.443,
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    
    p = params if params else PARAMS
    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)]
    )
    
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)
    
    print(f"LightGBM Spend -- RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return model, encoders, {"rmse": rmse, "mae": mae}


def save(model, encoders, path: str = "models/lgbm_spend.pkl"):
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders}, path)


if __name__ == "__main__":
    from data.loader import load_transactions
    from features.engineering import add_features
    
    df = load_transactions()
    df = add_features(df)
    model, encoders, metrics = train(df)
    save(model, encoders)
    print("Model saved.")