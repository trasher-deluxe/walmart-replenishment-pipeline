import pandas as pd
import numpy as np

def analyze_profiling(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame) -> dict:
    """Compute descriptive statistics and core retail business metrics."""
    
    numeric_cols = [
        "total_transactions", "cash_transactions", "card_transactions",
        "amount_total", "amount_cash", "amount_card", "units_sold",
        "avg_ticket", "replenishment_signal"
    ]
    
    # 1. Descriptive stats per continuous column
    descriptive_stats = {}
    for col in numeric_cols:
        if col in df_trans.columns:
            s = df_trans[col].dropna()
            if len(s) == 0:
                continue
            
            p5, p25, p50, p75, p95, p99 = np.percentile(s, [5, 25, 50, 75, 95, 99])
            descriptive_stats[col] = {
                "count": int(len(s)),
                "mean": float(round(s.mean(), 2)),
                "std": float(round(s.std(), 2)),
                "min": float(round(s.min(), 2)),
                "p5": float(round(p5, 2)),
                "p25": float(round(p25, 2)),
                "median_p50": float(round(p50, 2)),
                "p75": float(round(p75, 2)),
                "p95": float(round(p95, 2)),
                "p99": float(round(p99, 2)),
                "max": float(round(s.max(), 2)),
                "skewness": float(round(s.skew(), 4)),
                "kurtosis": float(round(s.kurtosis(), 4))
            }
            
    # 2. Business Metrics: Cash vs Card ratio
    total_amount_cash = float(df_trans["amount_cash"].sum(min_count=1))
    total_amount_card = float(df_trans["amount_card"].sum(min_count=1))
    total_amount = float(df_trans["amount_total"].sum())
    
    total_cash_tx = float(df_trans["cash_transactions"].sum(min_count=1))
    total_card_tx = float(df_trans["card_transactions"].sum(min_count=1))
    total_tx = float(df_trans["total_transactions"].sum())
    
    cash_vs_card = {
        "total_amount_total_mxn": float(round(total_amount, 2)),
        "total_amount_cash_mxn": float(round(total_amount_cash, 2)),
        "total_amount_card_mxn": float(round(total_amount_card, 2)),
        "cash_amount_share_pct": float(round(total_amount_cash / total_amount * 100, 2)),
        "card_amount_share_pct": float(round(total_amount_card / total_amount * 100, 2)),
        "total_cash_transactions": int(total_cash_tx),
        "total_card_transactions": int(total_card_tx),
        "cash_tx_share_pct": float(round(total_cash_tx / total_tx * 100, 2)),
        "card_tx_share_pct": float(round(total_card_tx / total_tx * 100, 2))
    }
    
    # 3. Merged analysis: Store Format performance
    df_merged = df_trans.merge(df_stores, on="store_id", how="left")
    
    format_performance = {}
    for fmt, group in df_merged.groupby("store_format"):
        format_performance[fmt] = {
            "store_count": int(group["store_id"].nunique()),
            "total_sales_mxn": float(round(group["amount_total"].sum(), 2)),
            "mean_daily_sales_mxn": float(round(group["amount_total"].mean(), 2)),
            "median_daily_sales_mxn": float(round(group["amount_total"].median(), 2)),
            "mean_avg_ticket_mxn": float(round(group["avg_ticket"].mean(), 2)),
            "median_avg_ticket_mxn": float(round(group["avg_ticket"].median(), 2)),
            "mean_units_sold": float(round(group["units_sold"].mean(), 2))
        }
        
    # Socioeconomic level performance
    socio_performance = {}
    for soc, group in df_merged.groupby("socioeconomic_level"):
        socio_performance[soc] = {
            "store_count": int(group["store_id"].nunique()),
            "total_sales_mxn": float(round(group["amount_total"].sum(), 2)),
            "mean_daily_sales_mxn": float(round(group["amount_total"].mean(), 2)),
            "mean_avg_ticket_mxn": float(round(group["avg_ticket"].mean(), 2)),
            "cash_share_pct": float(round(group["amount_cash"].sum() / group["amount_total"].sum() * 100, 2))
        }
        
    # 4. Impact of Promotions (`has_promotion`)
    promo_perf = {}
    for promo_val, group in df_trans.groupby("has_promotion"):
        label = "with_promotion" if promo_val == 1 else "without_promotion"
        promo_perf[label] = {
            "record_count": int(len(group)),
            "record_pct": float(round(len(group) / len(df_trans) * 100, 2)),
            "mean_amount_total": float(round(group["amount_total"].mean(), 2)),
            "median_amount_total": float(round(group["amount_total"].median(), 2)),
            "mean_units_sold": float(round(group["units_sold"].mean(), 2)),
            "mean_avg_ticket": float(round(group["avg_ticket"].mean(), 2))
        }
        
    promo_sales_lift_pct = float(round(
        ((promo_perf["with_promotion"]["mean_amount_total"] - promo_perf["without_promotion"]["mean_amount_total"]) / 
         promo_perf["without_promotion"]["mean_amount_total"]) * 100, 2
    ))
    promo_perf["sales_lift_pct"] = promo_sales_lift_pct

    # 5. Replenishment signal behavior
    replenishment_analysis = {
        "null_count": int(df_trans["replenishment_signal"].isnull().sum()),
        "null_pct": float(round(df_trans["replenishment_signal"].isnull().mean() * 100, 2)),
        "correlation_with_amount_total": float(round(df_trans["replenishment_signal"].corr(df_trans["amount_total"]), 4)),
        "correlation_with_units_sold": float(round(df_trans["replenishment_signal"].corr(df_trans["units_sold"]), 4))
    }
    
    # Check nulls in recent period for replenishment_signal
    df_trans["date_dt"] = pd.to_datetime(df_trans["date"])
    max_date = df_trans["date_dt"].max()
    last_7_days = df_trans[df_trans["date_dt"] >= (max_date - pd.Timedelta(days=7))]
    replenishment_analysis["last_7_days_null_pct"] = float(round(last_7_days["replenishment_signal"].isnull().mean() * 100, 2))
    df_trans.drop(columns=["date_dt"], inplace=True, errors="ignore")

    return {
        "descriptive_stats": descriptive_stats,
        "cash_vs_card": cash_vs_card,
        "format_performance": format_performance,
        "socioeconomic_performance": socio_performance,
        "promotion_impact": promo_perf,
        "replenishment_signal_analysis": replenishment_analysis
    }
