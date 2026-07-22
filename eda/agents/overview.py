import pandas as pd
import numpy as np

def analyze_overview(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> dict:
    """Analyze high-level dimensions, referential integrity, and temporal continuity."""
    
    # Dimensions and dtypes
    tables_overview = {}
    for name, df in [("transactions", df_trans), ("stores", df_stores), ("calendar", df_cal)]:
        tables_overview[name] = {
            "row_count": int(len(df)),
            "col_count": int(len(df.columns)),
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "cardinality": {col: int(df[col].nunique(dropna=False)) for col in df.columns}
        }
        
    # Referential integrity: check store_ids
    trans_stores = set(df_trans["store_id"].dropna().unique())
    stores_stores = set(df_stores["store_id"].dropna().unique())
    
    orphan_stores_in_trans = list(trans_stores - stores_stores)
    stores_without_trans = list(stores_stores - trans_stores)
    
    referential_integrity = {
        "unique_stores_in_transactions": len(trans_stores),
        "unique_stores_in_stores_csv": len(stores_stores),
        "orphan_stores_in_transactions_count": len(orphan_stores_in_trans),
        "orphan_stores_in_transactions": orphan_stores_in_trans,
        "stores_without_transactions_count": len(stores_without_trans),
        "stores_without_transactions": stores_without_trans,
        "referential_match_pct": float(round(len(trans_stores.intersection(stores_stores)) / max(len(stores_stores), 1) * 100, 2))
    }
    
    # Temporal continuity per store
    df_trans["date_dt"] = pd.to_datetime(df_trans["date"])
    global_min_date = str(df_trans["date_dt"].min().date())
    global_max_date = str(df_trans["date_dt"].max().date())
    
    all_dates = pd.date_range(start=global_min_date, end=global_max_date, freq="D")
    expected_days_count = len(all_dates)
    
    store_temporal = {}
    total_missing_pos_days = 0
    
    for store_id, group in df_trans.groupby("store_id"):
        store_dates = set(group["date_dt"])
        min_d = str(group["date_dt"].min().date())
        max_d = str(group["date_dt"].max().date())
        
        # Unique days store was active
        active_days = len(store_dates)
        missing_days = expected_days_count - active_days
        total_missing_pos_days += max(0, missing_days)
        
        store_temporal[str(store_id)] = {
            "min_date": min_d,
            "max_date": max_d,
            "active_days": int(active_days),
            "expected_days": int(expected_days_count),
            "missing_days": int(missing_days),
            "continuity_pct": float(round(active_days / expected_days_count * 100, 2))
        }
        
    temporal_continuity = {
        "global_start_date": global_min_date,
        "global_end_date": global_max_date,
        "total_expected_days": int(expected_days_count),
        "stores_count": len(store_temporal),
        "total_missing_store_days": int(total_missing_pos_days),
        "per_store_temporal_summary": store_temporal
    }
    
    # Clean up temporary datetime column
    df_trans.drop(columns=["date_dt"], inplace=True, errors="ignore")
    
    return {
        "tables_overview": tables_overview,
        "referential_integrity": referential_integrity,
        "temporal_continuity": temporal_continuity
    }
