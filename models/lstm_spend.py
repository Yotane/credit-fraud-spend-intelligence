import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

SEQUENCE_LENGTH = 5
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "prophet_residual"
]

TARGET = "amt"


class TransactionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    sequences, targets = _create_sequences(df, SEQUENCE_LENGTH)
    
    return sequences, targets, encoders, scaler


def _create_sequences(df: pd.DataFrame, seq_length: int):
    sequences = []
    targets = []
    
    customers = df["cc_num"].unique()
    
    for cc_num in customers:
        customer_data = df[df["cc_num"] == cc_num].sort_values("trans_date_trans_time")
        
        if len(customer_data) < seq_length + 1:
            continue
        
        features = customer_data[FEATURES].values
        
        for i in range(len(customer_data) - seq_length):
            seq = features[i:i + seq_length]
            tgt = customer_data.iloc[i + seq_length][TARGET]
            sequences.append(seq)
            targets.append(tgt)
    
    return np.array(sequences), np.array(targets)


def train(df: pd.DataFrame, params: dict = None) -> tuple:
    sequences, targets, encoders, scaler = prepare_data(df)
    
    split_idx = int(len(sequences) * 0.8)
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_tgt, val_tgt = targets[:split_idx], targets[split_idx:]
    
    train_dataset = TransactionDataset(train_seq, train_tgt)
    val_dataset = TransactionDataset(val_seq, val_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    p = params if params else {}
    hidden_size = p.get("hidden_size", HIDDEN_SIZE)
    num_layers = p.get("num_layers", NUM_LAYERS)
    dropout = p.get("dropout", DROPOUT)
    lr = p.get("lr", LEARNING_RATE)
    epochs = p.get("epochs", EPOCHS)
    
    input_size = len(FEATURES)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_seq, batch_tgt in train_loader:
            optimizer.zero_grad()
            preds = model(batch_seq)
            loss = criterion(preds, batch_tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                preds = model(batch_seq)
                loss = criterion(preds, batch_tgt)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    rmse = np.sqrt(best_val_loss)
    mae = np.mean(np.abs(model(torch.FloatTensor(val_seq)).detach().numpy() - val_tgt))
    
    print(f"LSTM Spend -- RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return model, encoders, scaler, {"rmse": rmse, "mae": mae}


def save(model, encoders, scaler, path: str = "models/lstm_spend.pkl"):
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