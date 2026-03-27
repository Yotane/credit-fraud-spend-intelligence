import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

SEQUENCE_LENGTH = 5
BATCH_SIZE = 256
EPOCHS = 20

# Use rolling_mean and rolling_std instead of prophet_residual to avoid target leakage
FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"

STUDY_NAME = "lstm_spend_tpe"
STORAGE = "sqlite:///studies/lstm_spend_tpe.db"
N_TRIALS = 10
TIMEOUT = 14400


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


def prepare_data(df):
    df = df.copy()
    
    cat_cols = ["gender", "category", "job", "age_group", "city_size"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    sequences, targets = _create_sequences(df, SEQUENCE_LENGTH)
    
    return sequences, targets, encoders


def train_and_evaluate(trial, df):
    # split customers first
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
        val_df[col] = le.transform(val_df[col].astype(str))
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
    
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    input_size = len(FEATURES)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCHS):
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
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Trial {trial.number} Epoch {epoch+1}/{EPOCHS} -- Val Loss: {val_loss:.4f}")
    
    return val_loss


def objective(trial, df):
    try:
        val_loss = train_and_evaluate(trial, df)
        return val_loss
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_study(df):
    Path("studies").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=1),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print(f"\nBest trial:")
    print(f"  Val Loss: {study.best_value:.4f}")
    print(f"  RMSE: {np.sqrt(study.best_value):.4f}")
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
    print(f"  Epochs per trial: {EPOCHS}")
    
    study = run_study(df)
    print("Study complete. Results saved to studies/")