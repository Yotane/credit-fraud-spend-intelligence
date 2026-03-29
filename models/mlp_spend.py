import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT = 0.2

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"


class TransactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


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
    
    p = params if params else {}
    hidden_layers = p.get("hidden_layers", HIDDEN_LAYERS)
    dropout = p.get("dropout", DROPOUT)
    lr = p.get("lr", LEARNING_RATE)
    epochs = p.get("epochs", EPOCHS)
    batch_size = p.get("batch_size", BATCH_SIZE)
    
    input_size = len(FEATURES)
    model = MLPModel(input_size, hidden_layers, dropout)
    
    train_dataset = TransactionDataset(X_train, y_train)
    val_dataset = TransactionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    rmse = np.sqrt(best_val_loss)
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_val)).detach().numpy()
    mae = np.mean(np.abs(preds - y_val))
    
    print(f"MLP Spend -- RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return model, encoders, scaler, {"rmse": rmse, "mae": mae}


def save(model, encoders, scaler, path: str = "models/mlp_spend.pkl"):
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