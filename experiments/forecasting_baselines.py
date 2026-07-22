"""Experiment: do classical/statistical forecasting models beat the seasonal-naive baseline?

The AWS Forecast / GCP Vertex Forecasting approach in miniature: fit statistical models across
the 480 (store x category) series and evaluate 7-day-ahead (the gap-safe horizon) with
rolling-origin cross-validation on the VALIDATION window (Dec 2023 - Jan 2024).

Run: uv sync --group experiments && uv run python experiments/forecasting_baselines.py

Fast trend+seasonality models (AutoETS, DynamicOptimizedTheta) run on the full fleet; AutoARIMA
(architecturally similar but much slower) runs on a deterministic sample for a data point.

Conclusion (see PROCESS.md §4): none beat SeasonalNaive here — the target's variation beyond
"last week's level" is noise, and the Dec-Jan holiday regime is absent from training (train ends
2023-11-30). Model choice is not the bottleneck; data is.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    DynamicOptimizedTheta,
    SeasonalNaive,
)

from src.data_processing import merge_master_dataset

VAL_END = "2024-01-31"
HORIZON = 7
N_WINDOWS = 8
SAMPLE_SIZE = 48


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else float("nan")


def _long_frame():
    df = merge_master_dataset(train_end="2023-11-30")
    s = (
        df[["store_id", "category", "date", "replenishment_signal"]]
        .assign(unique_id=lambda d: d["store_id"].astype(str) + "_" + d["category"].astype(str))
        .rename(columns={"date": "ds", "replenishment_signal": "y"})[["unique_id", "ds", "y"]]
    )
    s = s[s["ds"] <= VAL_END].sort_values(["unique_id", "ds"])
    s["y"] = s.groupby("unique_id")["y"].ffill().bfill()
    return s.dropna(subset=["y"])


def _cv_wape(s, models):
    sf = StatsForecast(models=models, freq="D", n_jobs=-1)
    cv = sf.cross_validation(df=s, h=HORIZON, step_size=HORIZON, n_windows=N_WINDOWS)
    return {repr(m): wape(cv["y"].to_numpy(), cv[repr(m)].to_numpy()) for m in models}


def main() -> None:
    s = _long_frame()
    print(f"Series: {s['unique_id'].nunique()} | filas: {len(s):,} | rango: {s['ds'].min().date()} → {s['ds'].max().date()}")

    print(f"\n[1] Full-fleet · trend+seasonality rápidos · h={HORIZON}, n_windows={N_WINDOWS}")
    fast = _cv_wape(s, [SeasonalNaive(season_length=7), AutoETS(season_length=7), DynamicOptimizedTheta(season_length=7)])
    for name, score in sorted(fast.items(), key=lambda r: r[1]):
        print(f"   {name:<26}{score:>8.4f}{'   <- baseline' if name == 'SeasonalNaive' else ''}")

    ids = sorted(s["unique_id"].unique())
    sample = set(np.random.default_rng(42).choice(ids, size=min(SAMPLE_SIZE, len(ids)), replace=False))
    ss = s[s["unique_id"].isin(sample)]
    print(f"\n[2] AutoARIMA sobre muestra de {len(sample)} series (mismo naive para comparar)")
    arima = _cv_wape(ss, [SeasonalNaive(season_length=7), AutoARIMA(season_length=7)])
    for name, score in sorted(arima.items(), key=lambda r: r[1]):
        print(f"   {name:<26}{score:>8.4f}{'   <- baseline' if name == 'SeasonalNaive' else ''}")

    naive = fast["SeasonalNaive"]
    best_fast = min((k for k in fast if k != "SeasonalNaive"), key=lambda k: fast[k])
    print("\n" + "=" * 48)
    beat = fast[best_fast] < naive or arima["AutoARIMA"] < arima["SeasonalNaive"]
    print(f"Veredicto: los statistical models {'SUPERAN' if beat else 'NO superan'} al seasonal-naive.")
    print(f"  full-fleet: naive {naive:.4f} vs mejor ({best_fast}) {fast[best_fast]:.4f}")
    print(f"  muestra   : naive {arima['SeasonalNaive']:.4f} vs AutoARIMA {arima['AutoARIMA']:.4f}")


if __name__ == "__main__":
    main()
