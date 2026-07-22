import pandas as pd

def analyze_features(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> dict:
    """Analyze feature engineering strategies based strictly on dataset properties."""
    
    # Categorical cardinalities for encoding strategies
    cat_encodings = {
        "category": {
            "unique_values": df_trans["category"].dropna().unique().tolist(),
            "cardinality": int(df_trans["category"].nunique()),
            "recommended_encoding": "One-Hot Encoding (Baja cardinalidad = 6)"
        },
        "region": {
            "unique_values": df_stores["region"].dropna().unique().tolist(),
            "cardinality": int(df_stores["region"].nunique()),
            "recommended_encoding": "One-Hot Encoding (Baja cardinalidad = 5)"
        },
        "socioeconomic_level": {
            "unique_values": df_stores["socioeconomic_level"].dropna().unique().tolist(),
            "cardinality": int(df_stores["socioeconomic_level"].nunique()),
            "recommended_encoding": "Ordinal Encoding (Jerarquía explícita: C < C+ < B < A/B)"
        },
        "store_format": {
            "unique_values": df_stores["store_format"].dropna().unique().tolist(),
            "cardinality": int(df_stores["store_format"].nunique()),
            "recommended_encoding": "One-Hot Encoding (Formato de tienda = 3)"
        }
    }
    
    # Imputation strategies
    imputation_strategies = {
        "cash_transactions_and_amount_cash": {
            "null_pct": float(round(df_trans["amount_cash"].isnull().mean() * 100, 2)),
            "cause": "Fallas de conectividad / caídas de punto de venta (POS)",
            "strategy": "Imputación mediante modelo de proporción histórica (Cash/Total ratio por tienda-categoría) o interpolación temporal regional."
        },
        "units_sold_and_avg_ticket": {
            "null_pct": float(round(df_trans["units_sold"].isnull().mean() * 100, 2)),
            "cause": "Error de transmisión de datos en línea de caja",
            "strategy": "Recalcular `avg_ticket` = `amount_total` / `total_transactions` donde `total_transactions` > 0. Imputar `units_sold` mediante `amount_total` / `precio_unitario_promedio_categoria`."
        },
        "replenishment_signal": {
            "null_pct": float(round(df_trans["replenishment_signal"].isnull().mean() * 100, 2)),
            "cause": "Latencia o corte en la ventana final del periodo de reposición",
            "strategy": "Forward fill (ffill) por tienda-categoría o imputación por promedio móvil de 7 días."
        }
    }
    
    # Derived variables recommendations
    derived_variables = [
        {
            "name": "cash_ratio_amount",
            "formula": "amount_cash / amount_total",
            "justification": "Captura la preferencia de pago por efectivo frente a tarjeta por tienda/región."
        },
        {
            "name": "sales_lag_7d & sales_lag_14d",
            "formula": "shift(7) / shift(14) en amount_total por store_id y category",
            "justification": "Captura estacionalidad semanal y quincenal de patrones de compra."
        },
        {
            "name": "rolling_mean_7d & rolling_std_7d",
            "formula": "Promedio y desviación estándar móvil de 7 días",
            "justification": "Suaviza volatilidad diaria y detecta tendencias operativas recientes."
        },
        {
            "name": "is_payday_window",
            "formula": "Flag 1 si date coincide con quincena (15 o fin de mes) o días +- 1",
            "justification": "Aumenta la capacidad predictiva ante picos sistemáticos de liquidez."
        }
    ]
    
    return {
        "categorical_encodings": cat_encodings,
        "imputation_strategies": imputation_strategies,
        "derived_variables": derived_variables
    }
