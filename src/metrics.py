"""Forecast accuracy (WAPE, RMSE) and business impact in MXN."""

import numpy as np
import pandas as pd


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(|actual|)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def business_impact_mxn(
    y_true: pd.Series,
    y_pred: pd.Series,
    overstock_cost_rate: float = 0.15,
    stockout_cost_rate: float = 0.30,
) -> dict:
    """Estimate MXN cost of forecast error, split into overstock vs. stockout.

    Over-forecasting (y_pred > y_true) drives excess inventory (`overstock_cost_rate`
    of the excess amount). Under-forecasting (y_pred < y_true) drives lost sales
    (`stockout_cost_rate` of the shortfall).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error = y_pred - y_true
    overstock_cost = np.clip(error, 0, None).sum() * overstock_cost_rate
    stockout_cost = np.clip(-error, 0, None).sum() * stockout_cost_rate
    return {
        "overstock_cost_mxn": float(overstock_cost),
        "stockout_cost_mxn": float(stockout_cost),
        "total_cost_mxn": float(overstock_cost + stockout_cost),
    }
