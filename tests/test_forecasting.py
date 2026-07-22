"""Unit tests for the AutoETS production forecaster's data prep (src/forecasting.py)."""

import numpy as np
import pandas as pd

from src.forecasting import build_series


def _toy_master():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    rows = []
    for store in (1, 2):
        for cat in ("A", "B"):
            for i, d in enumerate(dates):
                rows.append({"store_id": store, "category": cat, "date": d,
                             "replenishment_signal": float(i)})
    df = pd.DataFrame(rows)
    df.loc[3, "replenishment_signal"] = np.nan  # a hole to be filled
    return df


def test_build_series_long_format_and_ids():
    s = build_series(_toy_master(), "2023-01-10")
    assert set(s.columns) == {"unique_id", "ds", "y"}
    assert s["unique_id"].nunique() == 4           # 2 stores x 2 categories
    assert s["unique_id"].iloc[0] == "1_A"
    assert s["y"].isna().sum() == 0                # holes filled (ffill/bfill)


def test_build_series_respects_end_date():
    s = build_series(_toy_master(), "2023-01-05")
    assert s["ds"].max() == pd.Timestamp("2023-01-05")   # nothing past the cutoff leaks in
