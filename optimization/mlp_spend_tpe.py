import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

BATCH_SIZE = 4096
EPOCHS = 10

FEATURES = [
    "age", "distance_km", "hour", "day_of_week", "month", "is_weekend",
    "city_pop", "gender", "category", "job", "age_group", "city_size",
    "rolling_mean", "rolling_std"
]

TARGET = "amt"

STUDY_NAME = "mlp_spend_tpe"
STORAGE = "sqlite:///studies/mlp_spend_tpe.db"
N_TRIALS = 30
TIMEOUT = 7200


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


def train_and_evaluate(trial, df):
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
    
    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_layers = []
    for i in range(num_layers):
        hidden_size = trial.suggest_int(f"hidden_size_{i}", 64, 512)
        hidden_layers.append(hidden_size)
    
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
    
    input_size = len(FEATURES)
    model = MLPModel(input_size, hidden_layers, dropout)
    
    train_dataset = TransactionDataset(X_train, y_train)
    val_dataset = TransactionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(EPOCHS):
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 3:
            print(f"  Trial {trial.number} early stopped at epoch {epoch+1}")
            break
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Trial {trial.number} Epoch {epoch+1}/{EPOCHS} -- Val Loss: {val_loss:.4f}")
    
    return best_val_loss


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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print(f"\nBest trial:")
    if study.best_trial:
        print(f"  Val Loss: {study.best_value:.4f}")
        print(f"  RMSE: {np.sqrt(study.best_value):.4f}")
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