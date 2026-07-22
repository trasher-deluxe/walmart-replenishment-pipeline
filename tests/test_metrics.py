"""Unit tests for forecast metrics and financial impact (src/metrics.py)."""

import numpy as np
import pytest

from src.metrics import business_impact_mxn, rmse, wape


def test_wape_basic():
    # sum|err| = |100-110| + |100-90| = 20 ; sum|true| = 200 -> 0.1
    assert wape([100, 100], [110, 90]) == pytest.approx(0.1)


def test_wape_zero_actual_returns_zero():
    # guarded division: no actuals -> 0.0, never NaN/inf
    assert wape([0, 0], [5, 5]) == 0.0


def test_rmse_basic():
    # errors 3, 4 -> sqrt((9 + 16) / 2)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_business_impact_overstock_only():
    # pred over-forecasts by 2 units -> overstock 2 * 150 * 0.15 = 45, no stockout
    r = business_impact_mxn([10], [12])
    assert r["overstock_units"] == 2
    assert r["stockout_units"] == 0
    assert r["overstock_cost_mxn"] == pytest.approx(45.0)
    assert r["stockout_cost_mxn"] == 0.0


def test_stockout_costs_double_the_overstock_for_same_error():
    # a stockout (30%) must hurt twice as much as an overstock (15%) of equal size
    over = business_impact_mxn([10], [12])["total_financial_loss_mxn"]   # 45
    under = business_impact_mxn([10], [8])["total_financial_loss_mxn"]   # 90
    assert under == pytest.approx(2 * over)
