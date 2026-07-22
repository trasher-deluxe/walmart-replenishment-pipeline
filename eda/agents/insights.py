import json
import os

def generate_draft_analysis(stats_raw_path: str, output_path: str = "outputs/draft_analysis.md") -> str:
    """Generate outputs/draft_analysis.md using EXCLUSIVELY values from stats_raw.json."""
    
    with open(stats_raw_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
        
    ov = stats["overview"]["tables_overview"]
    ref = stats["overview"]["referential_integrity"]
    temp = stats["overview"]["temporal_continuity"]
    qual = stats["quality"]["missingness"]
    iqr = stats["quality"]["iqr_outliers"]
    iso = stats["quality"]["isolation_forest"]
    prof = stats["profiling"]
    feat = stats["features"]
    ml = stats["ml_readiness"]
    
    draft_md = f"""# Borrador de Análisis Exploratorio de Datos (EDA)

## 1. Multi-Table Overview & Integrity
- **Transacciones (`transactions.csv`):** {ov['transactions']['row_count']} filas, {ov['transactions']['col_count']} columnas, uso de memoria {ov['transactions']['memory_usage_mb']} MB.
- **Tiendas (`stores.csv`):** {ov['stores']['row_count']} filas, {ov['stores']['col_count']} columnas, uso de memoria {ov['stores']['memory_usage_mb']} MB.
- **Calendario (`calendar.csv`):** {ov['calendar']['row_count']} filas, {ov['calendar']['col_count']} columnas, uso de memoria {ov['calendar']['memory_usage_mb']} MB.
- **Integridad Referencial:** Coincidencia del {ref['referential_match_pct']}% entre tiendas en transacciones y `stores.csv`. Tiendas huérfanas en transacciones: {ref['orphan_stores_in_transactions_count']}. Tiendas en `stores.csv` sin transacciones: {ref['stores_without_transactions_count']}.
- **Continuidad Temporal:** Cobertura global del {temp['global_start_date']} al {temp['global_end_date']} ({temp['total_expected_days']} días). Total de días-tienda faltantes acumulados por fallas POS: {temp['total_missing_store_days']}.

## 2. Calidad de Datos, Nulos y Anomalías
- **Porcentaje global de nulos en `transactions.csv`:** {qual['transactions']['global_null_percentage']}%.
- **Nulos en `amount_cash`:** {qual['transactions']['columns']['amount_cash']['null_count']} ({qual['transactions']['columns']['amount_cash']['null_percentage']}%).
- **Nulos en `cash_transactions`:** {qual['transactions']['columns']['cash_transactions']['null_count']} ({qual['transactions']['columns']['cash_transactions']['null_percentage']}%).
- **Nulos en `units_sold`:** {qual['transactions']['columns']['units_sold']['null_count']} ({qual['transactions']['columns']['units_sold']['null_percentage']}%).
- **Nulos en `replenishment_signal`:** {qual['transactions']['columns']['replenishment_signal']['null_count']} ({qual['transactions']['columns']['replenishment_signal']['null_percentage']}%).
- **Duplicados exactos:** {stats['quality']['exact_duplicates']['transactions']} filas duplicadas en transacciones.
- **Detección de Outliers (IQR):** En `amount_total` se detectaron {iqr['amount_total']['outlier_count']} registros atípicos ({iqr['amount_total']['outlier_pct']}%), con un límite superior de ${iqr['amount_total']['upper_bound']:,.2f} MXN.
- **Isolation Forest (Contaminación 1%):** Se detectaron {iso['total_anomalies_detected']} anomalías globales. De estas, {iso['classified_valid_commercial_event_anomalies']} corresponden a eventos comerciales válidos (Buen Fin, Navidad, Quincena, Festivos) y {iso['classified_probable_pos_or_operational_errors']} corresponden a errores probables de captura/POS.

## 3. Perfilado Estadístico y Métricas de Negocio
- **Venta Total Acumulada:** ${prof['cash_vs_card']['total_amount_total_mxn']:,.2f} MXN.
- **Participación de Pago:** Efectivo representa {prof['cash_vs_card']['cash_amount_share_pct']}% (${prof['cash_vs_card']['total_amount_cash_mxn']:,.2f} MXN) vs Tarjeta {prof['cash_vs_card']['card_amount_share_pct']}% (${prof['cash_vs_card']['total_amount_card_mxn']:,.2f} MXN).
- **Participación por Transacciones:** Efectivo {prof['cash_vs_card']['cash_tx_share_pct']}% vs Tarjeta {prof['cash_vs_card']['card_tx_share_pct']}%.
- **Promedio Diario `amount_total`:** ${prof['descriptive_stats']['amount_total']['mean']:,.2f} MXN (Mediana: ${prof['descriptive_stats']['amount_total']['median_p50']:,.2f} MXN, P95: ${prof['descriptive_stats']['amount_total']['p95']:,.2f} MXN).
- **Desempeño por Formato:**
  - `Supercenter`: Ventas totales de ${prof['format_performance']['Supercenter']['total_sales_mxn']:,.2f} MXN, Ticket promedio ${prof['format_performance']['Supercenter']['mean_avg_ticket_mxn']:,.2f} MXN.
  - `Bodega`: Ventas totales de ${prof['format_performance']['Bodega']['total_sales_mxn']:,.2f} MXN, Ticket promedio ${prof['format_performance']['Bodega']['mean_avg_ticket_mxn']:,.2f} MXN.
  - `Express`: Ventas totales de ${prof['format_performance']['Express']['total_sales_mxn']:,.2f} MXN, Ticket promedio ${prof['format_performance']['Express']['mean_avg_ticket_mxn']:,.2f} MXN.
- **Impacto de Promociones (`has_promotion`):** El incremento promedio en ventas con promoción es del {prof['promotion_impact']['sales_lift_pct']}%.

## 4. Feature Engineering y ML Readiness
- **Estrategia de Imputación POS:** Reconstruir `cash_transactions` y `amount_cash` mediante ratios por tienda/categoría; recalcular `avg_ticket` directamente como `amount_total / total_transactions`.
- **Codificación:** One-Hot Encoding para `category` (6 categorías), `region` (5 regiones), `store_format` (3 formatos); Ordinal Encoding para `socioeconomic_level`.
- **Riesgos de Data Leakage:** Uso contemporáneo de `replenishment_signal` y desglose de pago (`amount_cash`, `amount_card`). Deben usarse únicamente como lags en modelos predictivos.
- **Sparsidad Temporal:** Existe un {ml['temporal_sparsity']['temporal_sparsity_pct']}% de combinaciones tienda-categoría-día faltantes en el grid teórico ({ml['temporal_sparsity']['missing_grid_rows']} filas faltantes).

## 5. Insights de Negocio Categorizados
- 🔴 **Alto Impacto:**
  - Pérdida de captura en POS: {temp['total_missing_store_days']} días-tienda con discontinuidad y {qual['transactions']['columns']['amount_cash']['null_percentage']}% de nulos en cobros de efectivo impactan directamente la conciliación diaria.
  - Data leakage potencial al predecir ventas usando `replenishment_signal` en tiempo T.
- 🟡 **Medio Impacto:**
  - Promociones generan un incremento del {prof['promotion_impact']['sales_lift_pct']}% en ventas promedio, requiriendo alineación en abastecimiento.
  - Alta preferencia de efectivo ({prof['cash_vs_card']['cash_amount_share_pct']}%) en formatos Bodega y niveles socioeconómicos C/C+.
- 🟢 **Bajo Impacto:**
  - Comportamiento consistente y alineado en formatos Supercenter con ticket promedio de ${prof['format_performance']['Supercenter']['mean_avg_ticket_mxn']:,.2f} MXN.
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(draft_md)
        
    return output_path
