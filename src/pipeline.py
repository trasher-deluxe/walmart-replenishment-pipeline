"""End-to-end Machine Learning Pipeline for Walmart Supply Chain Resilience.

Follows Google Professional ML Engineer Certification standards:
- Strictly prevents Data Leakage.
- Implements Chronological Temporal Splits (Train / Validation / Holdout).
- Trains LightGBM & XGBoost models to predict `replenishment_signal`.
- Evaluates technical accuracy (WAPE, RMSE) and financial impact in MXN.
"""

import json
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# mlflow-skinny (not the full `mlflow` package): the latest mlflow still pins pandas<3,
# incompatible with this project's pandas>=3.0.3. mlflow-skinny has no pandas pin and
# provides the same tracking/registry API used here. See PROCESS.md §5.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # opt into the local file backend (no tracking server)
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.data_processing import merge_master_dataset
from src.features import (
    build_features_pipeline,
    fit_target_encoders,
    transform_target_encoders
)
from src.models import DemandForecaster
from src.metrics import wape, rmse, business_impact_mxn

# Chronological split boundaries (single source of truth)
TRAIN_END = "2023-11-30"
VAL_START, VAL_END = "2023-12-01", "2024-01-31"
HOLDOUT_START, HOLDOUT_END = "2024-02-01", "2024-02-29"

# Length of the month-end inventory-blindness window (consecutive days with null signal).
# Drives which features are "gap-safe" and the seasonal-naive baseline horizon.
GAP_HORIZON = 7

REPO_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = f"file:{REPO_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT = "walmart-replenishment"
REGISTERED_MODEL_NAME = "walmart-replenishment"

def run_ml_pipeline() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        return _run_ml_pipeline(run)


def _run_ml_pipeline(run) -> dict:
    print("=" * 60)
    print("🤖 EJECUTANDO PIPELINE ML (GOOGLE PROFESSIONAL ML ENGINEER)")
    print("=" * 60)

    mlflow.log_params({
        "gap_horizon_days": GAP_HORIZON,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
    })

    # 1. Load Master Dataset (imputation medians fitted ONLY on the train window)
    print("\n📦 [Paso 1] Cargando y construyendo Master Dataset...")
    df_master = merge_master_dataset(train_end=TRAIN_END)
    print(f"   ✓ Master Dataset cargado: {df_master.shape[0]:,} filas x {df_master.shape[1]} columnas")
    
    # 2. Build Feature Engineering Pipeline
    print("\n⚡ [Paso 2] Generando features sin Data Leakage (Lags, Rolling, Ratios)...")
    df_featured = build_features_pipeline(df_master)
    print(f"   ✓ Features generadas. Dataset: {df_featured.shape[1]} columnas")
    
    # 3. Define Chronological Temporal Splits
    print("\n📅 [Paso 3] Aplicando División Temporal Estricta (Train / Validation / Holdout)...")
    
    # Temporal thresholds
    train_mask = (df_featured["date"] >= "2023-01-01") & (df_featured["date"] <= TRAIN_END)
    val_mask = (df_featured["date"] >= VAL_START) & (df_featured["date"] <= VAL_END)
    holdout_mask = (df_featured["date"] >= HOLDOUT_START) & (df_featured["date"] <= HOLDOUT_END)
    
    df_train = df_featured[train_mask].copy()
    df_val = df_featured[val_mask].copy()
    df_holdout = df_featured[holdout_mask].copy()
    
    print(f"   ✓ TRAIN Split      (2023-01-01 a 2023-11-30): {len(df_train):,} filas")
    print(f"   ✓ VALIDATION Split (2023-12-01 a 2024-01-31): {len(df_val):,} filas")
    print(f"   ✓ HOLDOUT Split    (2024-02-01 a 2024-02-29): {len(df_holdout):,} filas")
    
    # 4. Target Encoding fitted ONLY on TRAIN
    print("\n🔒 [Paso 4] Ajustando Target Encoders ÚNICAMENTE en el split TRAIN...")
    target_encoders = fit_target_encoders(df_train, target="replenishment_signal")
    
    df_train = transform_target_encoders(df_train, target_encoders)
    df_val = transform_target_encoders(df_val, target_encoders)
    df_holdout = transform_target_encoders(df_holdout, target_encoders)
    
    # 5. Define Feature Matrix X and Target y — GAP-SAFE features only.
    #    Business scenario: `replenishment_signal` is null for a stretch of consecutive days
    #    (month-end inventory blindness). A model is only useful there if every feature it uses
    #    is known >= GAP_HORIZON days before the target day. So we exclude:
    #      - lag_1 and rolling/ratio features (they need data inside the blind window),
    #    and keep static store/calendar features + lags of >= GAP_HORIZON days.
    static_cols = [
        "is_payday_window", "days_to_next_payday", "is_month_start", "is_month_end",
        "category_target_enc", "store_format_target_enc", "socioeconomic_level_ord",
        "size_sqm", "num_checkouts", "is_holiday", "is_payday", "is_weekend",
        "is_buen_fin", "is_navidad_season"
    ]
    gap_safe_lags = [
        c for c in df_train.columns
        if "_lag_" in c and int(c.split("_lag_")[1].rstrip("d")) >= GAP_HORIZON
    ]
    feature_cols = list(dict.fromkeys(static_cols + gap_safe_lags))
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_dict({"feature_cols": feature_cols}, "feature_cols.json")

    # Filter rows with complete features and valid target in Train and Val
    train_clean = df_train.dropna(subset=feature_cols + ["replenishment_signal"])
    val_clean = df_val.dropna(subset=feature_cols + ["replenishment_signal"])
    
    X_train, y_train = train_clean[feature_cols], train_clean["replenishment_signal"]
    X_val, y_val = val_clean[feature_cols], val_clean["replenishment_signal"]
    
    print(f"   ✓ Matrix X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # 6. Train Models (LightGBM & XGBoost)
    print("\n🌲 [Paso 5] Entrenando modelos predictivos (LightGBM & XGBoost)...")
    
    lgb_forecaster = DemandForecaster(algorithm="lightgbm", params={"n_estimators": 400, "learning_rate": 0.03})
    lgb_forecaster.fit(X_train, y_train, eval_set=(X_val, y_val))
    mlflow.log_params({f"lgb_{k}": v for k, v in lgb_forecaster.params.items()})

    xgb_forecaster = DemandForecaster(algorithm="xgboost", params={"n_estimators": 300, "learning_rate": 0.03})
    xgb_forecaster.fit(X_train, y_train, eval_set=(X_val, y_val))
    mlflow.log_params({f"xgb_{k}": v for k, v in xgb_forecaster.params.items()})
    
    # 7. Evaluate Model Performance on Validation Set
    print("\n📊 [Paso 6] Evaluando desempeño en VALIDATION Set...")
    
    val_preds_lgb = lgb_forecaster.predict(X_val)
    val_preds_xgb = xgb_forecaster.predict(X_val)
    
    wape_lgb = wape(y_val, val_preds_lgb)
    rmse_lgb = rmse(y_val, val_preds_lgb)
    impact_lgb = business_impact_mxn(y_val, val_preds_lgb)
    
    wape_xgb = wape(y_val, val_preds_xgb)
    rmse_xgb = rmse(y_val, val_preds_xgb)
    impact_xgb = business_impact_mxn(y_val, val_preds_xgb)

    # Seasonal-naive baseline: value one week ago (lag-7). This is the FAIR baseline for the
    # gapped scenario — lag-1 (yesterday) is unavailable when the signal is null for days, so
    # comparing against it would be dishonest. lag-7 survives a <=7-day blind window.
    val_preds_naive = val_clean["replenishment_signal_lag_7d"]
    wape_naive = wape(y_val, val_preds_naive)
    impact_naive = business_impact_mxn(y_val, val_preds_naive)

    # Real savings = baseline loss - best model loss (no invented factor).
    best_model_loss = min(impact_lgb["total_financial_loss_mxn"], impact_xgb["total_financial_loss_mxn"])
    savings_vs_naive_mxn = round(impact_naive["total_financial_loss_mxn"] - best_model_loss, 2)

    print(f"   📉 Baseline (seasonal-naive lag-7) -> WAPE: {wape_naive:.4f} ({wape_naive*100:.2f}%) | Pérdida: ${impact_naive['total_financial_loss_mxn']:,.2f} MXN")
    print(f"   🏆 LightGBM -> WAPE: {wape_lgb:.4f} ({wape_lgb*100:.2f}%) | RMSE: {rmse_lgb:.2f} | Pérdida Financiera Estimada: ${impact_lgb['total_financial_loss_mxn']:,.2f} MXN")
    print(f"   ⚡ XGBoost  -> WAPE: {wape_xgb:.4f} ({wape_xgb*100:.2f}%) | RMSE: {rmse_xgb:.2f} | Pérdida Financiera Estimada: ${impact_xgb['total_financial_loss_mxn']:,.2f} MXN")
    print(f"   💰 Ahorro real del mejor modelo vs baseline naive: ${savings_vs_naive_mxn:,.2f} MXN")

    mlflow.log_metrics({
        "wape_lightgbm": wape_lgb,
        "rmse_lightgbm": rmse_lgb,
        "financial_loss_lightgbm_mxn": impact_lgb["total_financial_loss_mxn"],
        "wape_xgboost": wape_xgb,
        "rmse_xgboost": rmse_xgb,
        "financial_loss_xgboost_mxn": impact_xgb["total_financial_loss_mxn"],
        "wape_baseline_naive": wape_naive,
        "financial_loss_baseline_naive_mxn": impact_naive["total_financial_loss_mxn"],
        "savings_best_model_vs_naive_mxn": savings_vs_naive_mxn,
    })
    passes_baseline_gate = savings_vs_naive_mxn >= 0
    mlflow.set_tag("passes_baseline_gate", str(passes_baseline_gate))

    # 8. Feature Importances
    fi_lgb = lgb_forecaster.feature_importance().head(10).to_dict()
    mlflow.log_dict({k: float(v) for k, v in fi_lgb.items()}, "feature_importance_lightgbm.json")

    # 8b. Log & Register Models. Both candidates are logged as run artifacts for full
    #     traceability; only the one that wins on VALIDATION loss is pushed to the
    #     Model Registry. Alias "staging" always points at the latest candidate;
    #     "production" only moves if it clears the champion/challenger gate (must not
    #     lose to the seasonal-naive baseline — same check enforced by the CI test).
    print("\n📋 [Paso 8] Registrando modelos en MLflow Model Registry...")
    lgb_signature = infer_signature(X_val, val_preds_lgb)
    lgb_model_info = mlflow.lightgbm.log_model(
        lgb_forecaster.model, artifact_path="model_lightgbm",
        signature=lgb_signature, input_example=X_val.head(3), serialization_format="pickle"
    )
    xgb_signature = infer_signature(X_val, val_preds_xgb)
    xgb_model_info = mlflow.xgboost.log_model(
        xgb_forecaster.model, artifact_path="model_xgboost",
        signature=xgb_signature, input_example=X_val.head(3)
    )

    if impact_lgb["total_financial_loss_mxn"] <= impact_xgb["total_financial_loss_mxn"]:
        best_algo, best_model_uri = "lightgbm", lgb_model_info.model_uri
    else:
        best_algo, best_model_uri = "xgboost", xgb_model_info.model_uri

    client = MlflowClient()
    registered_version = mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME).version
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "staging", registered_version)
    if passes_baseline_gate:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", registered_version)
        print(f"   ✓ '{best_algo}' v{registered_version} registrado como '{REGISTERED_MODEL_NAME}' -> @staging, @production (supera al baseline)")
    else:
        print(f"   ✓ '{best_algo}' v{registered_version} registrado como '{REGISTERED_MODEL_NAME}' -> @staging únicamente (NO supera al baseline, @production sin mover)")

    # 9. Predict on Holdout / Blind Test Period (Feb 2024)
    print("\n🔮 [Paso 7] Realizando predicciones sobre HOLDOUT (Ceguera de Inventario - Feb 2024)...")
    holdout_clean = df_holdout.dropna(subset=feature_cols).copy()
    holdout_preds = lgb_forecaster.predict(holdout_clean[feature_cols])
    holdout_clean["predicted_replenishment_signal"] = holdout_preds
    
    # Summary results
    results = {
        "dataset_summary": {
            "total_rows": int(len(df_master)),
            "train_rows": int(len(df_train)),
            "val_rows": int(len(df_val)),
            "holdout_rows": int(len(df_holdout))
        },
        "evaluation_setup": {
            "task": f"{GAP_HORIZON}-day-ahead forecast (gap-safe features only)",
            "gap_horizon_days": GAP_HORIZON,
            "baseline": "seasonal-naive (lag-7)"
        },
        "model_performance": {
            "baseline_seasonal_naive_lag7": {
                "wape": float(round(wape_naive, 4)),
                "financial_impact_mxn": impact_naive
            },
            "savings_best_model_vs_naive_mxn": float(savings_vs_naive_mxn),
            "lightgbm": {
                "wape": float(round(wape_lgb, 4)),
                "rmse": float(round(rmse_lgb, 2)),
                "financial_impact_mxn": impact_lgb
            },
            "xgboost": {
                "wape": float(round(wape_xgb, 4)),
                "rmse": float(round(rmse_xgb, 2)),
                "financial_impact_mxn": impact_xgb
            }
        },
        "top_10_feature_importances": {k: float(round(v, 4)) for k, v in fi_lgb.items()},
        "holdout_predictions_summary": {
            "predicted_records_count": int(len(holdout_clean)),
            "mean_predicted_signal": float(round(holdout_preds.mean(), 2)),
            "min_predicted_signal": float(round(holdout_preds.min(), 2)),
            "max_predicted_signal": float(round(holdout_preds.max(), 2))
        },
        "mlflow": {
            "run_id": run.info.run_id,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": registered_version,
            "registered_algorithm": best_algo,
            "passes_baseline_gate": passes_baseline_gate
        }
    }

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_file = output_dir / "ml_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(str(out_file))

    figures_dir = REPO_ROOT / "figures"
    if figures_dir.is_dir():
        mlflow.log_artifacts(str(figures_dir), artifact_path="figures")

    print("\n============================================================")
    print(f"✅ PIPELINE ML COMPLETADO Y GUARDADO EN '{out_file}'")
    print(f"   MLflow run: {run.info.run_id} (experiment '{MLFLOW_EXPERIMENT}')")
    print("============================================================")

    return results

if __name__ == "__main__":
    run_ml_pipeline()
