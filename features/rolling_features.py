import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def compute_rolling_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # rolling z-score with no leakage. uses shift(1) to only see past data
    df = df.copy()
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    
    if verbose:
        print(f"Computing rolling residuals for {len(df):,} transactions...")
    
    grouped = df.groupby("cc_num")
    
    # shift(1) ensures only past data is used
    df["rolling_mean"] = grouped["amt"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    
    df["rolling_std"] = grouped["amt"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    
    df["rolling_std"] = df["rolling_std"].fillna(1)
    df["rolling_std"] = df["rolling_std"].replace(0, 1)
    
    # residual for fraud detection (safe because target is is_fraud, not amt)
    df["rolling_zscore"] = (df["amt"] - df["rolling_mean"]) / df["rolling_std"]
    df["rolling_zscore"] = df["rolling_zscore"].fillna(0)
    
    # fill NaNs for first transaction of each customer
    df["rolling_mean"] = df["rolling_mean"].fillna(0)
    
    if verbose:
        count = len(df[df["rolling_zscore"] != 0])
        print(f"  Successfully computed rolling features for {count:,} transactions")
        print(f"  Residual stats: mean={df['rolling_zscore'].mean():.2f}, std={df['rolling_zscore'].std():.2f}")
    
    return df