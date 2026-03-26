import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

SEQUENCE_LENGTH = 5
BATCH_SIZE = 256
EPOCHS = 20  # reduced from 50 for faster trials
BASE_LEARNING_RATE = 0.001

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "prophet_residual"
]

TARGET = "amt"

STUDY_NAME = "lstm_spend_tpe"
STORAGE = "sqlite:///studies/lstm_spend_tpe.db"
N_TRIALS = 10  # reduced from 30 for faster completion
TIMEOUT = 14400  # 4 hours max


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


def prepare_data(df):
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


def train_and_evaluate(trial, sequences, targets):
    split_idx = int(len(sequences) * 0.8)
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_tgt, val_tgt = targets[:split_idx], targets[split_idx:]
    
    train_dataset = TransactionDataset(train_seq, train_tgt)
    val_dataset = TransactionDataset(val_seq, val_tgt)
    
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


def objective(trial, sequences, targets):
    try:
        val_loss = train_and_evaluate(trial, sequences, targets)
        return val_loss
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_study(sequences, targets):
    Path("studies").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.TPESampler(seed=1),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, sequences, targets), n_trials=N_TRIALS, timeout=TIMEOUT)
    
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
    
    print("Preparing sequences...")
    sequences, targets, encoders, scaler = prepare_data(df)
    print(f"  Total sequences: {len(sequences):,}")
    
    print(f"Running Optuna study: {STUDY_NAME}")
    print(f"  Trials: {N_TRIALS}, Timeout: {TIMEOUT}s")
    print(f"  Epochs per trial: {EPOCHS}")
    print("  (Progress will print every 5 epochs)")
    
    study = run_study(sequences, targets)
    print("Study complete. Results saved to studies/")