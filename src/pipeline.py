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
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import merge_master_dataset
from src.features import (
    build_features_pipeline,
    fit_target_encoders,
    transform_target_encoders
)
from src.models import DemandForecaster
from src.metrics import wape, rmse, business_impact_mxn

def run_ml_pipeline() -> dict:
    print("=" * 60)
    print("🤖 EJECUTANDO PIPELINE ML (GOOGLE PROFESSIONAL ML ENGINEER)")
    print("=" * 60)
    
    # 1. Load Master Dataset
    print("\n📦 [Paso 1] Cargando y construyendo Master Dataset...")
    df_master = merge_master_dataset()
    print(f"   ✓ Master Dataset cargado: {df_master.shape[0]:,} filas x {df_master.shape[1]} columnas")
    
    # 2. Build Feature Engineering Pipeline
    print("\n⚡ [Paso 2] Generando features sin Data Leakage (Lags, Rolling, Ratios)...")
    df_featured = build_features_pipeline(df_master)
    print(f"   ✓ Features generadas. Dataset: {df_featured.shape[1]} columnas")
    
    # 3. Define Chronological Temporal Splits
    print("\n📅 [Paso 3] Aplicando División Temporal Estricta (Train / Validation / Holdout)...")
    
    # Temporal thresholds
    train_mask = (df_featured["date"] >= "2023-01-01") & (df_featured["date"] <= "2023-11-30")
    val_mask = (df_featured["date"] >= "2023-12-01") & (df_featured["date"] <= "2024-01-31")
    holdout_mask = (df_featured["date"] >= "2024-02-01") & (df_featured["date"] <= "2024-02-29")
    
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
    
    # 5. Define Feature Matrix X and Target y
    feature_cols = [
        "sales_per_sqm", "sales_per_checkout", "cash_ratio_roll_30d",
        "is_payday_window", "days_to_next_payday", "is_month_start", "is_month_end",
        "category_target_enc", "store_format_target_enc", "socioeconomic_level_ord",
        "size_sqm", "num_checkouts", "is_holiday", "is_payday", "is_weekend",
        "is_buen_fin", "is_navidad_season"
    ]
    
    # Add lag and rolling feature names and deduplicate while preserving order
    lag_roll_cols = [c for c in df_train.columns if "_lag_" in c or "_roll_" in c]
    feature_cols.extend(lag_roll_cols)
    feature_cols = list(dict.fromkeys(feature_cols))
    
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
    
    xgb_forecaster = DemandForecaster(algorithm="xgboost", params={"n_estimators": 300, "learning_rate": 0.03})
    xgb_forecaster.fit(X_train, y_train, eval_set=(X_val, y_val))
    
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
    
    print(f"   🏆 LightGBM -> WAPE: {wape_lgb:.4f} ({wape_lgb*100:.2f}%) | RMSE: {rmse_lgb:.2f} | Pérdida Financiera Estimada: ${impact_lgb['total_financial_loss_mxn']:,.2f} MXN")
    print(f"   ⚡ XGBoost  -> WAPE: {wape_xgb:.4f} ({wape_xgb*100:.2f}%) | RMSE: {rmse_xgb:.2f} | Pérdida Financiera Estimada: ${impact_xgb['total_financial_loss_mxn']:,.2f} MXN")
    
    # 8. Feature Importances
    fi_lgb = lgb_forecaster.feature_importance().head(10).to_dict()
    
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
        "model_performance": {
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
        }
    }
    
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_file = output_dir / "ml_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n============================================================")
    print(f"✅ PIPELINE ML COMPLETADO Y GUARDADO EN '{out_file}'")
    print(f"============================================================")
    
    return results

if __name__ == "__main__":
    run_ml_pipeline()
