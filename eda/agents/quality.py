import pandas as pd
from sklearn.ensemble import IsolationForest

def analyze_quality(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> dict:
    """Analyze missingness, anomalies, exact duplicates, and run IQR & Isolation Forest for outliers."""
    
    # 1. Missingness
    missing_summary = {}
    for name, df in [("transactions", df_trans), ("stores", df_stores), ("calendar", df_cal)]:
        null_counts = df.isnull().sum()
        total_rows = len(df)
        col_missing = {}
        for col in df.columns:
            cnt = int(null_counts[col])
            pct = float(round((cnt / total_rows) * 100, 4))
            col_missing[col] = {"null_count": cnt, "null_percentage": pct}
        missing_summary[name] = {
            "total_rows": total_rows,
            "columns": col_missing,
            "total_null_cells": int(null_counts.sum()),
            "total_cells": total_rows * len(df.columns),
            "global_null_percentage": float(round((null_counts.sum() / (total_rows * len(df.columns))) * 100, 4))
        }

    # Grouped missingness in transactions (e.g. by category & store)
    trans_grouped_nulls = {}
    for col in ["cash_transactions", "amount_cash", "units_sold", "avg_ticket", "replenishment_signal"]:
        if col in df_trans.columns:
            nulls_by_cat = df_trans[df_trans[col].isnull()].groupby("category").size().to_dict()
            trans_grouped_nulls[col] = {str(k): int(v) for k, v in nulls_by_cat.items()}

    # 2. Exact Duplicates & Constant Columns
    exact_duplicates = {
        "transactions": int(df_trans.duplicated().sum()),
        "stores": int(df_stores.duplicated().sum()),
        "calendar": int(df_cal.duplicated().sum())
    }

    constant_cols = {}
    for name, df in [("transactions", df_trans), ("stores", df_stores), ("calendar", df_cal)]:
        consts = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        near_consts = [col for col in df.columns if (df[col].value_counts(normalize=True, dropna=False).iloc[0] > 0.99)]
        constant_cols[name] = {"constant": consts, "near_constant_99pct": near_consts}

    # 3. Outlier Analysis (IQR & Isolation Forest) on continuous variables
    continuous_vars = ["amount_total", "units_sold", "avg_ticket", "amount_cash", "amount_card"]
    iqr_results = {}
    
    for var in continuous_vars:
        if var in df_trans.columns:
            series = df_trans[var].dropna()
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]
            
            iqr_results[var] = {
                "q25": float(round(q25, 2)),
                "q75": float(round(q75, 2)),
                "iqr": float(round(iqr, 2)),
                "lower_bound": float(round(lower_bound, 2)),
                "upper_bound": float(round(upper_bound, 2)),
                "outlier_count": int(len(outliers_iqr)),
                "outlier_pct": float(round(len(outliers_iqr) / len(series) * 100, 2))
            }

    # Isolation Forest on main metrics
    iso_features = ["amount_total", "card_transactions", "total_transactions"]
    df_iso = df_trans[iso_features].dropna()
    iso_model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_preds = iso_model.fit_predict(df_iso)
    
    anomaly_indices = df_iso.index[iso_preds == -1]
    df_anomalies = df_trans.loc[anomaly_indices].copy()
    
    # Classify anomalies into probable POS error vs Commercial Event
    # Merge calendar to check holiday/buen_fin/navidad
    df_anom_cal = df_anomalies.merge(df_cal, on="date", how="left")
    
    commercial_events = df_anom_cal[
        df_anom_cal["is_holiday"].fillna(False) | 
        df_anom_cal["is_buen_fin"].fillna(False) | 
        df_anom_cal["is_navidad_season"].fillna(False) | 
        df_anom_cal["is_payday"].fillna(False)
    ]
    
    pos_error_candidates = df_anom_cal[
        (df_anom_cal["amount_total"] == 0) | 
        (df_anom_cal["cash_transactions"].isnull() & df_anom_cal["units_sold"].isnull()) |
        (~df_anom_cal.index.isin(commercial_events.index))
    ]

    isolation_forest_summary = {
        "contamination_rate": 0.01,
        "total_anomalies_detected": int(len(anomaly_indices)),
        "anomaly_pct": float(round(len(anomaly_indices) / len(df_iso) * 100, 2)),
        "classified_valid_commercial_event_anomalies": int(len(commercial_events)),
        "classified_probable_pos_or_operational_errors": int(len(pos_error_candidates))
    }

    return {
        "missingness": missing_summary,
        "grouped_missingness_transactions": trans_grouped_nulls,
        "exact_duplicates": exact_duplicates,
        "constant_columns": constant_cols,
        "iqr_outliers": iqr_results,
        "isolation_forest": isolation_forest_summary
    }
