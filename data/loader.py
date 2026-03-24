import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "credit_card_transactions.csv"

DROP_COLS = ["Unnamed: 0", "first", "last", "street", "trans_num"]

def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    return df