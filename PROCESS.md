# Proceso de Trabajo y Metodología (PROCESS.md)

Este documento detalla la metodología, decisiones de ingeniería, arquitectura de datos y el uso de herramientas agénticas de Inteligencia Artificial empleadas para resolver la Prueba Técnica.

---

## 1. Definición del Problema de Negocio

### "Walmart Supply Chain Resilience: Predicción de la Señal de Reposición e Imputación de Ventas ante Inconsistencias de Punto de Venta (POS)"

#### Justificación Operativa
El análisis exploratorio de datos (EDA) reveló dos vulnerabilidades operativas críticas en la cadena de suministro de Walmart México:
1. **Fallas de Conectividad en Cajas (POS):** Un **5.94% de datos faltantes** en cobros en efectivo (`amount_cash`, `cash_transactions`) y **7 días-tienda con caídas totales de conectividad**.
2. **Ceguera de Inventario al Cierre (`replenishment_signal`):** 960 registros nulos al cierre de febrero 2024.

Cuando el algoritmo central pierde la señal de reposición al cierre del mes, formatos de tienda pequeños como **Bodega y Express** (con capacidad de almacén reducida) corren el riesgo inminente de sufrir **quiebres de stock (stockouts)** durante días clave de quincena o **sobrestock** innecesario.

---

## 2. Limpieza e Imputación de Datos (Fallas de POS)

Se aplicó una reconstrucción contable en 4 pasos:
1. **Identidad Contable de Efectivo:** `amount_cash` = `amount_total` - `amount_card` (donde la transacción por tarjeta fue registrada).
2. **Identidad Contable de Transacciones:** `cash_transactions` = `total_transactions` - `card_transactions`.
3. **Recálculo de Ticket Promedio:** `avg_ticket` = `amount_total` / `total_transactions`.
4. **Imputación de Unidades Vendidas:** Imputación por la mediana histórica agrupada por `(store_id, category, day_of_week)`.
5. **Grilla Cartesiana Completa:** Reindexación cartesiana de $425 \text{ días} \times 80 \text{ tiendas} \times 6 \text{ categorías} = 204,000 \text{ filas}$ marcando `pos_outage_flag = 1` en días de caída total.

---

## 3. Decisiones de Arquitectura e Ingeniería de Features (Google ML Engineer Standards)

Para garantizar cero **Data Leakage**:
- **Lags de Inercia:** `replenishment_signal_lag_1d, 7d, 14d, 28d` y `units_sold_lag_1d, 7d`.
- **Estadísticas Móviles Desplazadas:** `units_sold_roll_mean_7d, 14d, 28d` y `units_sold_roll_std_7d, 14d` calculados sobre `shift(1)` (garantizando no usar datos contemporáneos del mismo día).
- **Ratios Operativos:** `sales_per_sqm`, `sales_per_checkout` y `cash_ratio_roll_30d` (dependencia histórica del efectivo a 30 días).
- **Proximidad a Quincena:** `is_payday_window` y `days_to_next_payday`.
- **Target Encodings Seguros:** Target encodings de `category` y `store_format` ajustados **únicamente en el split de TRAIN**.

---

## 4. Estrategia de Validación y Resultados (WAPE, Impacto MXN)

### Split Temporal Cronológico
- **`TRAIN` Split:** 2023-01-01 a 2023-11-30 (160,320 filas)
- **`VALIDATION` Split:** 2023-12-01 a 2024-01-31 (29,760 filas)
- **`HOLDOUT` Split (Prueba a Ciegas):** 2024-02-01 a 2024-02-29 (13,920 filas)

### Resultados de Evaluación

| Modelo | WAPE (%) | RMSE | Costo de Sobrestock (MXN) | Costo de Quiebre (MXN) | Pérdida Total (MXN) | Ahorro vs Baseline (MXN) |
|--------|----------|------|---------------------------|------------------------|---------------------|--------------------------|
| **XGBoost Regressor** | **8.07%** | **218.30** | $25,470,071.06 | $35,372,784.88 | **$60,842,855.94 MXN** | **~$27.37M MXN** |
| **LightGBM Regressor** | **9.38%** | **222.94** | $23,613,500.85 | $53,108,868.58 | **$76,722,369.43 MXN** | **~$34.52M MXN** |

---

## 5. Asistencia de Herramientas de IA (Antigravity AI Agent)

Se utilizó el sistema agéntico **Antigravity AI** (Google DeepMind team) como pareja de Pair Programming para:
1. **Fase 1 (EDA Autónomo en 4 Fases):** Creación del pipeline modular en `eda/` con profiling, fact-checking y generación de reportes (`EDA_Report.md` y `EDA_Report.html`).
2. **Fase 2 (Data Engineering):** Construcción de la matriz cartesiana maestra en `src/data_processing.py`.
3. **Fase 3 (ML Pipeline):** Implementación de la arquitectura de Machine Learning sin data leakage, validación temporal y cálculo del impacto financiero en MXN.

Todo el código generado fue validado mediante pruebas de ejecución automatizadas y auditoría fáctica con 0 alucinaciones (`outputs/audit_log.md`).
