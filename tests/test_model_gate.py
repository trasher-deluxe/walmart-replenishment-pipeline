"""Champion/challenger gate: the production forecaster must beat the seasonal-naive baseline.

Reads `outputs/ml_results.json`, produced by `python src/pipeline.py`. This is the guardrail
against promoting a worse-than-baseline model to `@production` in the MLflow Model Registry.

The production model is AutoETS (statsforecast), which beats the seasonal-naive by ~7 WAPE points
(see PROCESS.md §4). The tabular GBM investigation, which loses to the naive, is kept documented
but is NOT the production model.
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "ml_results.json"


def test_production_model_beats_naive_baseline():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    savings = results["model_performance"]["savings_best_model_vs_naive_mxn"]
    assert savings >= 0, (
        f"El modelo de producción pierde ${abs(savings):,.2f} MXN frente al baseline "
        "seasonal-naive. No debe promoverse a @production."
    )
