"""Champion/challenger gate: the trained model must not lose to the seasonal-naive baseline.

Reads `outputs/ml_results.json`, produced by `python src/pipeline.py`. This is the guardrail
against re-inflating metrics (see PROCESS.md §4): a model that loses to lag-7 must not be
promoted to `@production` in the MLflow Model Registry, and CI must go red.
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "ml_results.json"


def test_model_beats_naive_baseline():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    savings = results["model_performance"]["savings_best_model_vs_naive_mxn"]
    assert savings >= 0, (
        f"El mejor modelo pierde ${abs(savings):,.2f} MXN frente al baseline seasonal-naive "
        "(lag-7). No debe promoverse a @production."
    )
