import pandas as pd
import numpy as np
from features.prophet_residual import compute_prophet_residuals

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # work on a copy so the original dataframe is never mutated
    df = df.copy()

    # age in full years at the time of the transaction
    df["age"] = (
        (df["trans_date_trans_time"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    ).astype(int)

    df["distance_km"] = _haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # pd.cut buckets continuous values into labeled bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 29, 39, 49, 59, 100],
        labels=["20s", "30s", "40s", "50s", "60+"]
    )

    df["city_size"] = pd.cut(
        df["city_pop"],
        bins=[0, 10000, 100000, float("inf")],
        labels=["rural", "suburban", "urban"]
    )

    df = compute_prophet_residuals(df, verbose=True)

    return df


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371  # earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # standard haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))