"""Production forecaster: AutoETS (Nixtla statsforecast) — the model that beats the naive.

A tabular GBM cannot extrapolate a trending series and loses to a lag-7 seasonal-naive
(see PROCESS.md §4). A native exponential-smoothing forecaster (level + trend + weekly
seasonality, with smoothed state over the whole history) beats it by ~7 WAPE points. Evaluated
with rolling-origin, 7-day-ahead cross-validation across all (store × category) series — the same
gap-safe scenario the business faces at month-end.
"""

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from src.metrics import business_impact_mxn, rmse, wape

SEASON_LENGTH = 7


def build_series(df_master: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """Long format (unique_id, ds, y) for statsforecast, up to end_date, gaps filled per series."""
    s = (
        df_master[["store_id", "category", "date", "replenishment_signal"]]
        .assign(unique_id=lambda d: d["store_id"].astype(str) + "_" + d["category"].astype(str))
        .rename(columns={"date": "ds", "replenishment_signal": "y"})[["unique_id", "ds", "y"]]
    )
    s = s[s["ds"] <= pd.Timestamp(end_date)].sort_values(["unique_id", "ds"])
    s["y"] = s.groupby("unique_id")["y"].ffill().bfill()
    return s.dropna(subset=["y"])


def _metrics(y, p) -> dict:
    return {
        "wape": float(round(wape(y, p), 4)),
        "rmse": float(round(rmse(y, p), 2)),
        "financial_impact_mxn": business_impact_mxn(y, p),
    }


def evaluate_production(df_master: pd.DataFrame, val_end: str, horizon: int, n_windows: int = 8) -> dict:
    """Rolling-origin CV on the validation window: AutoETS (production) vs SeasonalNaive (baseline)."""
    s = build_series(df_master, val_end)
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=SEASON_LENGTH), AutoETS(season_length=SEASON_LENGTH)],
        freq="D",
        n_jobs=-1,
    )
    cv = sf.cross_validation(df=s, h=horizon, step_size=horizon, n_windows=n_windows)
    y = cv["y"].to_numpy()
    return {
        "n_series": int(s["unique_id"].nunique()),
        "n_windows": int(cv["cutoff"].nunique()),
        "horizon_days": horizon,
        "autoets": _metrics(y, cv["AutoETS"].to_numpy()),
        "seasonal_naive": _metrics(y, cv["SeasonalNaive"].to_numpy()),
    }


def fit_production(df_master: pd.DataFrame, end_date: str) -> StatsForecast:
    """Fit the final AutoETS on all history up to end_date, ready to forecast the blind window."""
    s = build_series(df_master, end_date)
    sf = StatsForecast(models=[AutoETS(season_length=SEASON_LENGTH)], freq="D", n_jobs=-1)
    sf.fit(df=s)
    return sf
