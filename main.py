"""End-to-end pipeline entrypoint: load -> features -> train -> evaluate."""

import pandas as pd

from src.data_processing import load_and_prepare
from src.features import build_features
from src.metrics import business_impact_mxn, rmse, wape
from src.models import DemandForecaster

TARGET = "amount_total"
TEST_DAYS = 28


def main() -> None:
    df = load_and_prepare()
    df = build_features(df, target=TARGET)
    df = df.dropna(subset=[c for c in df.columns if c.startswith(f"{TARGET}_lag")])

    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]

    feature_cols = [c for c in df.columns if c.startswith(f"{TARGET}_lag") or c.startswith(f"{TARGET}_roll")]
    model = DemandForecaster(algorithm="lightgbm").fit(train[feature_cols], train[TARGET])
    preds = model.predict(test[feature_cols])

    print(f"WAPE: {wape(test[TARGET], preds):.3f}")
    print(f"RMSE: {rmse(test[TARGET], preds):.2f}")
    print(business_impact_mxn(test[TARGET], preds))


if __name__ == "__main__":
    main()
