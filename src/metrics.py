"""Forecast accuracy metrics (WAPE, RMSE) and Business Financial Impact in MXN."""

import numpy as np
import pandas as pd

def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(|actual|)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denom)

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def business_impact_mxn(
    y_true: pd.Series,
    y_pred: pd.Series,
    unit_value_mxn: float = 150.0,
    overstock_cost_rate: float = 0.15,
    stockout_cost_rate: float = 0.30
) -> dict:
    """Estimate MXN cost of replenishment forecast error split into overstock vs stockout.
    
    - Over-forecasting (y_pred > y_true): Excess inventory holding cost (overstock_cost_rate of unit_value_mxn).
    - Under-forecasting (y_pred < y_true): Lost sales gross margin & customer friction (stockout_cost_rate of unit_value_mxn).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    error_units = y_pred - y_true
    
    overstock_units = np.clip(error_units, 0, None).sum()
    stockout_units = np.clip(-error_units, 0, None).sum()
    
    overstock_cost_mxn = float(overstock_units * unit_value_mxn * overstock_cost_rate)
    stockout_cost_mxn = float(stockout_units * unit_value_mxn * stockout_cost_rate)
    total_cost_mxn = overstock_cost_mxn + stockout_cost_mxn
    
    return {
        "overstock_units": float(overstock_units),
        "stockout_units": float(stockout_units),
        "overstock_cost_mxn": float(round(overstock_cost_mxn, 2)),
        "stockout_cost_mxn": float(round(stockout_cost_mxn, 2)),
        "total_financial_loss_mxn": float(round(total_cost_mxn, 2)),
    }
