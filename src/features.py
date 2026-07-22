"""Feature engineering pipeline following Google Professional ML Engineer standards.

Strictly prevents Data Leakage by:
1. Shifting all rolling windows and lag features by at least 1 day.
2. Fitting target encodings and category statistics ONLY on the Training split.
3. Excluding contemporaneous payment metrics from future-day target forecasting.
"""

import pandas as pd
import numpy as np

GROUP_KEYS = ["store_id", "category"]

def add_lag_features(
    df: pd.DataFrame, 
    targets: tuple[str, ...] = ("amount_total", "units_sold", "replenishment_signal"), 
    lags: tuple[int, ...] = (1, 7, 14, 28)
) -> pd.DataFrame:
    """Add lagged values of key metrics per store/category, sorted by date."""
    df = df.sort_values(["store_id", "category", "date"]).copy()
    for target in targets:
        if target in df.columns:
            for lag in lags:
                df[f"{target}_lag_{lag}d"] = df.groupby(GROUP_KEYS)[target].shift(lag)
    return df

def add_rolling_features(
    df: pd.DataFrame, 
    targets: tuple[str, ...] = ("amount_total", "units_sold"), 
    windows: tuple[int, ...] = (7, 14, 28)
) -> pd.DataFrame:
    """Add rolling mean/std of targets per store/category (shifted by 1 to prevent leakage)."""
    df = df.sort_values(["store_id", "category", "date"]).copy()
    for target in targets:
        if target in df.columns:
            shifted = df.groupby(GROUP_KEYS)[target].shift(1)
            for window in windows:
                df[f"{target}_roll_mean_{window}d"] = shifted.groupby(
                    [df["store_id"], df["category"]]
                ).transform(lambda s: s.rolling(window, min_periods=1).mean())
                
                df[f"{target}_roll_std_{window}d"] = shifted.groupby(
                    [df["store_id"], df["category"]]
                ).transform(lambda s: s.rolling(window, min_periods=1).std())
    return df

def add_store_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute store operational density features (sales per sqm, sales per checkout)."""
    df = df.copy()
    
    # Use lagged amount_total to prevent contemporaneous leakage
    if "amount_total_lag_1d" in df.columns:
        df["sales_per_sqm"] = df["amount_total_lag_1d"] / (df["size_sqm"] + 1)
        df["sales_per_checkout"] = df["amount_total_lag_1d"] / (df["num_checkouts"] + 1)
    
    # Store age feature
    if "opening_year" in df.columns and "year" in df.columns:
        df["store_age_years"] = df["year"] - df["opening_year"]
        
    return df

def add_payment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical rolling 30d cash-to-total ratio per store and category."""
    df = df.sort_values(["store_id", "category", "date"]).copy()
    
    if "amount_cash" in df.columns and "amount_total" in df.columns:
        # Shift by 1 to prevent same-day leakage
        shifted_cash = df.groupby(GROUP_KEYS)["amount_cash"].shift(1)
        shifted_total = df.groupby(GROUP_KEYS)["amount_total"].shift(1)
        
        roll_cash = shifted_cash.groupby([df["store_id"], df["category"]]).transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
        roll_total = shifted_total.groupby([df["store_id"], df["category"]]).transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
        
        df["cash_ratio_roll_30d"] = (roll_cash / (roll_total + 1e-5)).clip(0, 1)
        
    return df

def add_calendar_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rich calendar features: payday window, proximity to payday, month phases."""
    df = df.copy()
    
    # Payday window: days 14, 15, 16 and last 2 days of month
    df["day_of_month"] = df["date"].dt.day
    df["days_in_month"] = df["date"].dt.days_in_month
    
    is_mid_payday = df["day_of_month"].isin([14, 15, 16])
    is_end_payday = df["day_of_month"] >= (df["days_in_month"] - 1)
    df["is_payday_window"] = (is_mid_payday | is_end_payday | df["is_payday"].fillna(False)).astype(int)
    
    # Days to next payday (15 or month end)
    df["days_to_next_payday"] = np.where(
        df["day_of_month"] <= 15,
        15 - df["day_of_month"],
        df["days_in_month"] - df["day_of_month"]
    )
    
    # Month phase
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= (df["days_in_month"] - 4)).astype(int)
    
    return df

def fit_target_encoders(df_train: pd.DataFrame, target: str = "replenishment_signal") -> dict:
    """Fit category and store format target encodings strictly on the TRAIN split."""
    encodings = {}
    
    # Category target encoding
    if "category" in df_train.columns and target in df_train.columns:
        cat_means = df_train.groupby("category")[target].mean().to_dict()
        global_mean = float(df_train[target].mean())
        encodings["category_target_enc"] = (cat_means, global_mean)
        
    # Store format target encoding
    if "store_format" in df_train.columns and target in df_train.columns:
        fmt_means = df_train.groupby("store_format")[target].mean().to_dict()
        global_mean = float(df_train[target].mean())
        encodings["store_format_target_enc"] = (fmt_means, global_mean)
        
    return encodings

def transform_target_encoders(df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    """Apply target encodings fitted on TRAIN to any split."""
    df = df.copy()
    
    if "category_target_enc" in encodings and "category" in df.columns:
        cat_means, global_mean = encodings["category_target_enc"]
        df["category_target_enc"] = df["category"].map(cat_means).fillna(global_mean)
        
    if "store_format_target_enc" in encodings and "store_format" in df.columns:
        fmt_means, global_mean = encodings["store_format_target_enc"]
        df["store_format_target_enc"] = df["store_format"].map(fmt_means).fillna(global_mean)
        
    # Ordinal encoding for socioeconomic level
    socio_map = {"C": 0, "C+": 1, "B": 2, "A/B": 3}
    if "socioeconomic_level" in df.columns:
        df["socioeconomic_level_ord"] = df["socioeconomic_level"].map(socio_map).fillna(0)
        
    return df

def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature transformations except target encodings."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_store_ratios(df)
    df = add_payment_ratios(df)
    df = add_calendar_engineered_features(df)
    return df
