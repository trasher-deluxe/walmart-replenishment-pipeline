"""Training, prediction, and feature importance analysis using LightGBM and XGBoost."""

from dataclasses import dataclass, field
from typing import Literal
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

@dataclass
class DemandForecaster:
    """Wrapper around LightGBM or XGBoost Regressors."""

    algorithm: Literal["lightgbm", "xgboost"] = "lightgbm"
    params: dict = field(default_factory=dict)
    model: object = None
    feature_names: list = field(default_factory=list)

    def __post_init__(self):
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1
        }
        if self.algorithm == "lightgbm":
            default_params.update({"max_depth": 8, "num_leaves": 31, "verbosity": -1})
        else:
            default_params.update({"max_depth": 6})
            
        default_params.update(self.params)
        self.params = default_params

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: tuple[pd.DataFrame, pd.Series] = None) -> "DemandForecaster":
        self.feature_names = list(X.columns)
        
        if self.algorithm == "lightgbm":
            self.model = lgb.LGBMRegressor(**self.params)
            if eval_set is not None:
                X_val, y_val = eval_set
                self.model.fit(
                    X, y,
                    eval_X=X_val, eval_y=y_val,
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                self.model.fit(X, y)
        else:
            self.model = xgb.XGBRegressor(**self.params)
            if eval_set is not None:
                X_val, y_val = eval_set
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X, y)
                
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call `.fit()` first.")
        preds = self.model.predict(X[self.feature_names])
        return pd.Series(preds, index=X.index, name="predicted_replenishment_signal")

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call `.fit()` first.")
        
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
