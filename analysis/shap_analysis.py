import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder


def prepare_shap_data(df, features, encoders, scaler):
    df = df.copy()
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    
    if encoders is not None:
        for col, le in encoders.items():
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )
    else:
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    X = df[features].values
    if scaler is not None:
        X = scaler.transform(X)
    
    return X, features


def plot_shap_summary(explainer, X, feature_names, save_path: str, title: str):
    shap_values = explainer.shap_values(X[:2000])
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X[:2000], feature_names=feature_names, show=False, plot_type="dot")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shap_bar(explainer, X, feature_names, save_path: str, title: str):
    shap_values = explainer.shap_values(X[:2000])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X[:2000], feature_names=feature_names, 
                      show=False, plot_type="bar")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def load_model_data(path: str):
    data = joblib.load(path)
    
    if isinstance(data, dict):
        return data.get("model"), data.get("encoders"), data.get("scaler")
    else:
        return data, None, None


def analyze_spend_model(model_name: str = "lgbm"):
    print(f"\nAnalyzing {model_name.upper()} Spend Model")
    
    model, encoders, scaler = load_model_data(f"models/{model_name}_spend.pkl")
    
    from data.loader import load_transactions
    from features.engineering import add_features
    
    df = load_transactions()
    df = add_features(df)
    
    features = [
        "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
        "city_pop", "gender", "category", "job", "age_group", "city_size",
        "rolling_mean", "rolling_std"
    ]
    
    X, feature_names = prepare_shap_data(df, features, encoders, scaler)
    
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    Path("analysis/plots").mkdir(exist_ok=True)
    
    display_name = "XGBOOST" if model_name == "xgb" else model_name.upper()
    
    plot_shap_summary(
        explainer, X, feature_names, 
        f"analysis/plots/shap_{model_name}_spend_summary.png",
        f"{display_name} Spend - SHAP Feature Importance"
    )
    
    plot_shap_bar(
        explainer, X, feature_names,
        f"analysis/plots/shap_{model_name}_spend_bar.png",
        f"{display_name} Spend - Top Features"
    )
    
    print(f"{model_name} analysis complete!\n")


def analyze_fraud_model():
    print("\nAnalyzing LightGBM Fraud Model")
    
    model, encoders, scaler = load_model_data("models/lgbm_fraud.pkl")
    
    from data.loader import load_transactions
    from features.engineering import add_features
    
    df = load_transactions()
    df = add_features(df)
    
    features = [
        "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
        "city_pop", "gender", "category", "job", "age_group", "city_size",
        "prophet_residual"
    ]
    
    X, feature_names = prepare_shap_data(df, features, encoders, scaler)
    
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    Path("analysis/plots").mkdir(exist_ok=True)
    
    plot_shap_summary(
        explainer, X, feature_names,
        "analysis/plots/shap_fraud_summary.png",
        "LightGBM Fraud - SHAP Feature Importance"
    )
    
    plot_shap_bar(
        explainer, X, feature_names,
        "analysis/plots/shap_fraud_bar.png",
        "LightGBM Fraud - Top Features"
    )
    
    print("Fraud analysis complete!\n")


if __name__ == "__main__":
    analyze_spend_model("lgbm")
    analyze_spend_model("xgb")
    analyze_fraud_model()
    print("All SHAP analyses complete.")