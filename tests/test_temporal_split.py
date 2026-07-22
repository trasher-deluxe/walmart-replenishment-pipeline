"""Tests for chronological split integrity and the gap-safe feature invariant."""

import pandas as pd

from src.pipeline import (
    GAP_HORIZON,
    HOLDOUT_END,
    HOLDOUT_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)


def test_splits_are_chronological_and_ordered():
    # ISO date strings compare correctly; each split starts after the previous ends
    assert TRAIN_END < VAL_START <= VAL_END < HOLDOUT_START <= HOLDOUT_END


def test_splits_do_not_overlap():
    dates = pd.date_range("2023-01-01", "2024-02-29", freq="D")
    train = (dates >= "2023-01-01") & (dates <= TRAIN_END)
    val = (dates >= VAL_START) & (dates <= VAL_END)
    holdout = (dates >= HOLDOUT_START) & (dates <= HOLDOUT_END)
    # every day belongs to at most one split (no temporal leakage across splits)
    overlap = train.astype(int) + val.astype(int) + holdout.astype(int)
    assert overlap.max() == 1


def test_gap_safe_selection_excludes_short_lags_and_rolling():
    # replicates the feature filter in run_ml_pipeline: only lags >= GAP_HORIZON survive
    cols = [
        "replenishment_signal_lag_1d",
        "replenishment_signal_lag_7d",
        "replenishment_signal_lag_14d",
        "units_sold_roll_mean_7d",
        "cash_ratio_roll_30d",
        "size_sqm",
    ]
    gap_safe_lags = [
        c for c in cols
        if "_lag_" in c and int(c.split("_lag_")[1].rstrip("d")) >= GAP_HORIZON
    ]
    assert "replenishment_signal_lag_7d" in gap_safe_lags
    assert "replenishment_signal_lag_14d" in gap_safe_lags
    # lag_1 and rolling features are unavailable during a multi-day blind window
    assert "replenishment_signal_lag_1d" not in gap_safe_lags
    assert all("_roll_" not in c for c in gap_safe_lags)
