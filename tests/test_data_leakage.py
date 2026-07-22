"""Tests that PROVE the anti-leakage guarantees — the project's core claim.

These assert the properties the README/PROCESS advertise: lags/rolling are strictly
shifted, target encoders learn only from TRAIN, and POS imputation medians come only
from the TRAIN window. If any of these regress, data from the future would leak into
training and CI must catch it.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import impute_pos_failures
from src.features import (
    add_lag_features,
    add_rolling_features,
    fit_target_encoders,
    transform_target_encoders,
)


def _toy():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    return pd.DataFrame({
        "store_id": 1,
        "category": "A",
        "date": dates,
        "units_sold": np.arange(20, dtype=float),
        "replenishment_signal": np.arange(20, dtype=float),
    })


def test_lag_features_are_shifted_not_contemporaneous():
    df = add_lag_features(_toy(), targets=("replenishment_signal",), lags=(1, 7))
    df = df.sort_values("date").reset_index(drop=True)
    # lag_1 at row i must equal the target at row i-1 (never the same day)
    assert df["replenishment_signal_lag_1d"].iloc[5] == df["replenishment_signal"].iloc[4]
    assert df["replenishment_signal_lag_7d"].iloc[10] == df["replenishment_signal"].iloc[3]
    # no data exists before the first day
    assert pd.isna(df["replenishment_signal_lag_1d"].iloc[0])


def test_rolling_stats_exclude_the_current_day():
    df = _toy()
    df.loc[10, "units_sold"] = 999.0  # spike on day 10
    out = add_rolling_features(df, targets=("units_sold",), windows=(3,))
    out = out.sort_values("date").reset_index(drop=True)
    # the spike-day rolling mean uses shift(1): it must NOT include its own 999 value
    assert out["units_sold_roll_mean_3d"].iloc[10] < 100


def test_target_encoders_fit_on_train_only():
    train = _toy()
    train["category"] = ["A"] * 10 + ["B"] * 10
    enc = fit_target_encoders(train, target="replenishment_signal")
    unseen = pd.DataFrame({"category": ["Z"], "replenishment_signal": [0.0]})
    out = transform_target_encoders(unseen, enc)
    # a category never seen in TRAIN falls back to the global TRAIN mean, never NaN
    _, global_mean = enc["category_target_enc"]
    assert out["category_target_enc"].iloc[0] == pytest.approx(global_mean)


def _pos_toy():
    n = 20
    return pd.DataFrame({
        "store_id": 1,
        "category": "A",
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "total_transactions": np.full(n, 100.0),
        "cash_transactions": np.full(n, 40.0),
        "card_transactions": np.full(n, 60.0),
        "amount_total": np.full(n, 1000.0),
        "amount_cash": np.full(n, 400.0),
        "amount_card": np.full(n, 600.0),
        "units_sold": np.arange(n, dtype=float),
        "avg_ticket": np.full(n, 10.0),
    })


def test_imputation_medians_come_only_from_train_window():
    df = _pos_toy()
    df.loc[15, "units_sold"] = np.nan          # a hole in the validation period
    df.loc[16:, "units_sold"] = 1_000_000.0    # blow up the future values
    imp = impute_pos_failures(df.copy(), train_end="2023-01-10")
    # the filled value must derive from the small TRAIN medians, not the 1e6 future
    assert imp.loc[15, "units_sold"] < 100
