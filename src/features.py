"""Feature engineering: lags, rolling windows, and calendar variables."""

import pandas as pd

GROUP_KEYS = ["store_id", "category"]


def add_lag_features(
    df: pd.DataFrame, target: str = "amount_total", lags: tuple[int, ...] = (1, 7, 14, 28)
) -> pd.DataFrame:
    """Add lagged values of `target` per store/category, sorted by date."""
    df = df.sort_values("date").copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(GROUP_KEYS)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target: str = "amount_total", windows: tuple[int, ...] = (7, 28)
) -> pd.DataFrame:
    """Add rolling mean/std of `target` per store/category (shifted to avoid leakage)."""
    df = df.sort_values("date").copy()
    shifted = df.groupby(GROUP_KEYS)[target].shift(1)
    for window in windows:
        df[f"{target}_roll_mean_{window}"] = shifted.groupby(
            [df["store_id"], df["category"]]
        ).transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{target}_roll_std_{window}"] = shifted.groupby(
            [df["store_id"], df["category"]]
        ).transform(lambda s: s.rolling(window, min_periods=1).std())
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive extra calendar signals (already-merged columns get one-hot/bool casts)."""
    df = df.copy()
    df["is_month_start"] = df["date"].dt.is_month_start
    df["is_month_end"] = df["date"].dt.is_month_end
    return df


def build_features(df: pd.DataFrame, target: str = "amount_total") -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)
    df = add_calendar_features(df)
    return df
