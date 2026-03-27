import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    "rolling_mean", "rolling_std"
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


def _create_sequences(df, seq_length):
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
    df = df.copy()
    
    # split customers first to avoid identity leakage
    train_df, val_df = train_test_split(
        df["cc_num"].unique(), test_size=0.2, random_state=1
    )
    
    train_df = df[df["cc_num"].isin(train_df)].copy()
    val_df = df[df["cc_num"].isin(val_df)].copy()
    
    # fit encoders on train only
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # handle unseen labels in val set
        val_df[col] = val_df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )
        encoders[col] = le
    
    sequences, targets = _create_sequences(train_df, SEQUENCE_LENGTH)
    val_sequences, val_targets = _create_sequences(val_df, SEQUENCE_LENGTH)
    
    # fit scaler on train only
    scaler = StandardScaler()
    train_seq_flat = sequences.reshape(-1, len(FEATURES))
    val_seq_flat = val_sequences.reshape(-1, len(FEATURES))
    train_seq_flat = scaler.fit_transform(train_seq_flat)
    val_seq_flat = scaler.transform(val_seq_flat)
    train_seq = train_seq_flat.reshape(sequences.shape)
    val_seq = val_seq_flat.reshape(val_sequences.shape)
    
    train_dataset = TransactionDataset(train_seq, targets)
    val_dataset = TransactionDataset(val_seq, val_targets)
    
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
    best_model_state = None
    
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
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    rmse = np.sqrt(best_val_loss)
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(val_seq)).detach().numpy()
    mae = np.mean(np.abs(preds - val_targets))
    
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