import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visuals(df_trans: pd.DataFrame, df_stores: pd.DataFrame, df_cal: pd.DataFrame, output_dir: str = "figures") -> list:
    """Generate 5 mandatory high-resolution figures at 300 DPI."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})
    
    generated_files = []
    
    # 1. missing_matrix.png
    plt.figure(figsize=(10, 6))
    df_merged = df_trans.merge(df_stores, on="store_id", how="left").merge(df_cal, on="date", how="left")
    missing_pct = df_merged.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) > 0:
        ax = sns.barplot(x=missing_pct.values, y=missing_pct.index, hue=missing_pct.index, palette="viridis", legend=False)
        plt.title("Porcentaje de Datos Faltantes por Variable", fontsize=14, fontweight="bold")
        plt.xlabel("% Faltante")
        plt.ylabel("Variable")
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 0.2, i, f"{v:.2f}%", va="center", fontsize=9)
    else:
        plt.text(0.5, 0.5, "Sin datos faltantes", ha="center", va="center")
        plt.title("Matriz de Datos Faltantes")
        
    path_missing = os.path.join(output_dir, "missing_matrix.png")
    plt.savefig(path_missing, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(path_missing)

    # 2. sales_time_series.png
    df_daily = df_trans.groupby("date")[["amount_total", "total_transactions"]].sum().reset_index()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily.sort_values("date", inplace=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(df_daily["date"], df_daily["amount_total"] / 1e6, color="#1f77b4", label="Ventas Totales (Millones MXN)", linewidth=1.5)
    line2 = ax2.plot(df_daily["date"], df_daily["total_transactions"], color="#ff7f0e", label="Total Transacciones", linewidth=1.2, alpha=0.7)
    
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Ventas Totales (Millones MXN)", color="#1f77b4")
    ax2.set_ylabel("Total Transacciones", color="#ff7f0e")
    
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.title("Evolución Temporal de Ventas Totales y Transacciones Diarias", fontsize=14, fontweight="bold")
    
    path_ts = os.path.join(output_dir, "sales_time_series.png")
    plt.savefig(path_ts, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(path_ts)

    # 3. store_format_performance.png
    df_store_perf = df_trans.merge(df_stores, on="store_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=df_store_perf, x="store_format", y="amount_total", hue="socioeconomic_level", ax=axes[0], palette="Set2")
    axes[0].set_title("Ventas por Formato y Nivel Socioeconómico", fontweight="bold")
    axes[0].set_ylabel("Venta Total (MXN)")
    axes[0].set_yscale("log")
    
    sns.boxplot(data=df_store_perf, x="store_format", y="avg_ticket", hue="socioeconomic_level", ax=axes[1], palette="Set2")
    axes[1].set_title("Ticket Promedio por Formato y Nivel Socioeconómico", fontweight="bold")
    axes[1].set_ylabel("Ticket Promedio (MXN)")
    
    plt.suptitle("Desempeño Operativo por Formato de Tienda y Nivel Socioeconómico", fontsize=15, fontweight="bold")
    path_format = os.path.join(output_dir, "store_format_performance.png")
    plt.savefig(path_format, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(path_format)

    # 4. calendar_impact.png
    df_cal_perf = df_trans.merge(df_cal, on="date", how="left")
    events = ["is_payday", "is_buen_fin", "is_navidad_season", "is_holiday"]
    event_means = []
    
    for ev in events:
        mean_on = df_cal_perf[df_cal_perf[ev]]["amount_total"].mean()
        mean_off = df_cal_perf[~df_cal_perf[ev]]["amount_total"].mean()
        event_means.append({"Evento": ev.replace("is_", "").replace("_", " ").title(), "Activo": mean_on, "Inactivo": mean_off})
        
    df_events = pd.DataFrame(event_means).melt(id_vars="Evento", var_name="Estado", value_name="Venta Promedio MXN")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_events, x="Evento", y="Venta Promedio MXN", hue="Estado", palette="Blues_d")
    plt.title("Impacto de Eventos de Calendario en Venta Promedio Diaria", fontsize=14, fontweight="bold")
    plt.ylabel("Venta Promedio Diaria (MXN)")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"${height:,.0f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
            
    path_cal = os.path.join(output_dir, "calendar_impact.png")
    plt.savefig(path_cal, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(path_cal)

    # 5. correlation_matrix.png
    num_cols = ["amount_total", "total_transactions", "cash_transactions", "card_transactions", 
                "amount_cash", "amount_card", "units_sold", "avg_ticket", "replenishment_signal"]
    corr = df_trans[num_cols].corr(method="pearson")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Matriz de Correlación de Pearson entre Variables Cuantitativas", fontsize=14, fontweight="bold")
    
    path_corr = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(path_corr, dpi=300, bbox_inches="tight")
    plt.close()
    generated_files.append(path_corr)

    return generated_files
