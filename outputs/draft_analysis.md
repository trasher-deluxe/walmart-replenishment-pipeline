# Borrador de Análisis Exploratorio de Datos (EDA)

## 1. Multi-Table Overview & Integrity
- **Transacciones (`transactions.csv`):** 203958 filas, 13 columnas, uso de memoria 53.814 MB.
- **Tiendas (`stores.csv`):** 80 filas, 9 columnas, uso de memoria 0.021 MB.
- **Calendario (`calendar.csv`):** 425 filas, 15 columnas, uso de memoria 0.114 MB.
- **Integridad Referencial:** Coincidencia del 100.0% entre tiendas en transacciones y `stores.csv`. Tiendas huérfanas en transacciones: 0. Tiendas en `stores.csv` sin transacciones: 0.
- **Continuidad Temporal:** Cobertura global del 2023-01-01 al 2024-02-29 (425 días). Total de días-tienda faltantes acumulados por fallas POS: 7.

## 2. Calidad de Datos, Nulos y Anomalías
- **Porcentaje global de nulos en `transactions.csv`:** 1.3652%.
- **Nulos en `amount_cash`:** 12123 (5.9439%).
- **Nulos en `cash_transactions`:** 12123 (5.9439%).
- **Nulos en `units_sold`:** 6118 (2.9996%).
- **Nulos en `replenishment_signal`:** 960 (0.4707%).
- **Duplicados exactos:** 0 filas duplicadas en transacciones.
- **Detección de Outliers (IQR):** En `amount_total` se detectaron 10895 registros atípicos (5.34%), con un límite superior de $415,112.52 MXN.
- **Isolation Forest (Contaminación 1%):** Se detectaron 2040 anomalías globales. De estas, 1569 corresponden a eventos comerciales válidos (Buen Fin, Navidad, Quincena, Festivos) y 474 corresponden a errores probables de captura/POS.

## 3. Perfilado Estadístico y Métricas de Negocio
- **Venta Total Acumulada:** $34,791,412,195.45 MXN.
- **Participación de Pago:** Efectivo representa 39.49% ($13,739,930,288.06 MXN) vs Tarjeta 57.98% ($20,173,118,081.81 MXN).
- **Participación por Transacciones:** Efectivo 42.0% vs Tarjeta 51.03%.
- **Promedio Diario `amount_total`:** $170,581.26 MXN (Mediana: $129,965.23 MXN, P95: $424,963.03 MXN).
- **Desempeño por Formato:**
  - `Supercenter`: Ventas totales de $14,444,562,362.85 MXN, Ticket promedio $418.03 MXN.
  - `Bodega`: Ventas totales de $15,170,416,302.69 MXN, Ticket promedio $405.41 MXN.
  - `Express`: Ventas totales de $5,176,433,529.91 MXN, Ticket promedio $415.65 MXN.
- **Impacto de Promociones (`has_promotion`):** El incremento promedio en ventas con promoción es del 0.04%.

## 4. Feature Engineering y ML Readiness
- **Estrategia de Imputación POS:** Reconstruir `cash_transactions` y `amount_cash` mediante ratios por tienda/categoría; recalcular `avg_ticket` directamente como `amount_total / total_transactions`.
- **Codificación:** One-Hot Encoding para `category` (6 categorías), `region` (5 regiones), `store_format` (3 formatos); Ordinal Encoding para `socioeconomic_level`.
- **Riesgos de Data Leakage:** Uso contemporáneo de `replenishment_signal` y desglose de pago (`amount_cash`, `amount_card`). Deben usarse únicamente como lags en modelos predictivos.
- **Sparsidad Temporal:** Existe un 0.02% de combinaciones tienda-categoría-día faltantes en el grid teórico (42 filas faltantes).

## 5. Insights de Negocio Categorizados
- 🔴 **Alto Impacto:**
  - Pérdida de captura en POS: 7 días-tienda con discontinuidad y 5.9439% de nulos en cobros de efectivo impactan directamente la conciliación diaria.
  - Data leakage potencial al predecir ventas usando `replenishment_signal` en tiempo T.
- 🟡 **Medio Impacto:**
  - Promociones generan un incremento del 0.04% en ventas promedio, requiriendo alineación en abastecimiento.
  - Alta preferencia de efectivo (39.49%) en formatos Bodega y niveles socioeconómicos C/C+.
- 🟢 **Bajo Impacto:**
  - Comportamiento consistente y alineado en formatos Supercenter con ticket promedio de $418.03 MXN.
