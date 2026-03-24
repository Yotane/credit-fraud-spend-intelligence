import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

MIN_TRANSACTIONS = 10


def compute_prophet_residuals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    
    df["prophet_residual"] = np.nan
    customers = df["cc_num"].unique()
    
    if verbose:
        print(f"Computing Prophet residuals for {len(customers):,} customers...")
        print(f"  (Only customers with >= {MIN_TRANSACTIONS} transactions will be fitted)")
    
    fitted_customers = 0  # track actual customers fitted
    error_count = 0
    for cc_num in tqdm(customers, disable=not verbose, desc="Customers"):
        customer_mask = df["cc_num"] == cc_num
        customer_data = df[customer_mask].copy()
        
        if len(customer_data) < MIN_TRANSACTIONS:
            continue
        
        try:
            residuals = _fit_customer_prophet(customer_data)
            df.loc[customer_mask, "prophet_residual"] = residuals.values
            fitted_customers += 1  # count this customer as fitted
        except Exception as e:
            if error_count < 3:
                print(f"  Error: {e}")
            error_count += 1
            continue
    
    df["prophet_residual"] = df["prophet_residual"].fillna(0)
    
    if verbose:
        print(f"  Successfully fitted {fitted_customers:,} customers ({fitted_customers/len(customers)*100:.1f}%)")
        print(f"  Residual stats: mean={df['prophet_residual'].mean():.2f}, std={df['prophet_residual'].std():.2f}")
    
    return df


def _fit_customer_prophet(customer_df: pd.DataFrame) -> pd.Series:
    original_index = customer_df.index.copy()
    
    customer_df = customer_df.copy()
    # use string format for date to avoid type mismatches
    customer_df["date"] = customer_df["trans_date_trans_time"].dt.strftime("%Y-%m-%d")
    
    daily = (
        customer_df.groupby("date")["amt"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "amt": "y"})
    )
    
    if len(daily) < 2:
        return pd.Series(np.zeros(len(customer_df)), index=original_index)
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(daily)
    
    future_dates = pd.DataFrame({
        "ds": customer_df["trans_date_trans_time"].dt.strftime("%Y-%m-%d").unique()
    })
    
    forecast = model.predict(future_dates)
    # convert ds to string to match customer_df date format
    forecast["date"] = forecast["ds"].dt.strftime("%Y-%m-%d")
    forecast["expected_daily_spend"] = forecast["yhat"]
    
    customer_df = customer_df.merge(
        forecast[["date", "expected_daily_spend"]],
        on="date",
        how="left"
    )
    
    residuals = customer_df["amt"] - customer_df["expected_daily_spend"]
    residuals.index = original_index
    
    return residuals