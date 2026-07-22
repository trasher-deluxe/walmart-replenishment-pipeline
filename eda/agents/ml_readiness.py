import pandas as pd
import numpy as np

def analyze_ml_readiness(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> dict:
    """Evaluate dataset readiness for predictive modeling (Demand Forecast & Replenishment)."""
    
    # 1. Data Leakage risks
    leakage_risks = [
        {
            "feature": "replenishment_signal",
            "risk_level": "ALTO",
            "explanation": "La variable replenishment_signal es calculada internamente con base en la demanda observada contemporánea. Utilizarla como predictor contemporáneo de amount_total produce data leakage.",
            "mitigation": "Utilizar únicamente lags pasados (t-1, t-7) de replenishment_signal."
        },
        {
            "feature": "cash_transactions / card_transactions / amount_cash / amount_card / avg_ticket",
            "risk_level": "ALTO",
            "explanation": "Estas variables son resultados contemporáneos de la venta del día. En un modelo de pronóstico de demanda previo a la apertura del día, estas variables no están disponibles.",
            "mitigation": "Utilizar únicamente agregados pasados o excluir del feature set de predicción diaria anticipada."
        }
    ]
    
    # 2. Multicollinearity analysis
    num_cols = ["total_transactions", "cash_transactions", "card_transactions", 
                "amount_total", "amount_cash", "amount_card", "units_sold"]
    corr_matrix = df_trans[num_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            val = float(round(corr_matrix.loc[col1, col2], 4))
            if abs(val) > 0.85:
                high_corr_pairs.append({
                    "var1": col1,
                    "var2": col2,
                    "correlation": val,
                    "risk": "Multicolinealidad alta (> 0.85)"
                })
                
    # 3. Class imbalance & Temporal Sparsity
    # Temporal sparsity: missing date-store-category combinations
    unique_stores = df_trans["store_id"].nunique()
    unique_cats = df_trans["category"].nunique()
    df_trans["date_dt"] = pd.to_datetime(df_trans["date"])
    total_dates = df_trans["date_dt"].nunique()
    df_trans.drop(columns=["date_dt"], inplace=True, errors="ignore")
    
    theoretical_total_rows = unique_stores * unique_cats * total_dates
    actual_rows = len(df_trans)
    sparsity_pct = float(round((1 - (actual_rows / theoretical_total_rows)) * 100, 2))
    
    sparsity_analysis = {
        "unique_stores": unique_stores,
        "unique_categories": unique_cats,
        "total_calendar_days": total_dates,
        "theoretical_total_grid_rows": theoretical_total_rows,
        "actual_transaction_rows": actual_rows,
        "missing_grid_rows": theoretical_total_rows - actual_rows,
        "temporal_sparsity_pct": sparsity_pct,
        "impact": "Existen combinaciones tienda-categoría-día faltantes que deben reindexarse con valor 0 para evitar sesgos en modelos de series de tiempo."
    }

    return {
        "data_leakage_risks": leakage_risks,
        "high_multicollinearity_pairs": high_corr_pairs,
        "temporal_sparsity": sparsity_analysis
    }
