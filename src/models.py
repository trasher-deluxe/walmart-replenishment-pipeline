"""Training and prediction with LightGBM/XGBoost."""

from dataclasses import dataclass, field
from typing import Literal

import lightgbm as lgb
import pandas as pd
import xgboost as xgb


@dataclass
class DemandForecaster:
    """Thin wrapper around a LightGBM or XGBoost regressor."""

    algorithm: Literal["lightgbm", "xgboost"] = "lightgbm"
    params: dict = field(default_factory=dict)
    model: object = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandForecaster":
        if self.algorithm == "lightgbm":
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call `.fit()` first.")
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call `.fit()` first.")
        return pd.Series(
            self.model.feature_importances_, index=self.model.feature_name_
            if self.algorithm == "lightgbm" else self.model.feature_names_in_,
        ).sort_values(ascending=False)
