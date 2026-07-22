# Retail Demand Forecasting & POS Data Recovery

## Problema de Negocio

**Walmart Supply Chain Resilience: Predicción de la Señal de Reposición e Imputación de Ventas ante Inconsistencias de Punto de Venta (POS)**

Cadena de retail con presencia en múltiples regiones de México. El dataset de transacciones diarias por tienda/categoría presenta dos vulnerabilidades operativas críticas:
1. **Fallas de Conectividad/POS:** Un 5.94% de datos faltantes en métricas de efectivo (`amount_cash`, `cash_transactions`) y periodos completos con caídas de sistema.
2. **Ceguera de Inventario al Cierre (`replenishment_signal`):** Registros nulos en los últimos días de febrero 2024.

Este proyecto implementa una solución agéntica end-to-end que:
1. Reconstruye la demanda real mediante identidades contables e imputación por mediana histórica.
2. Entrena modelos predictivos (**XGBoost / LightGBM**) alineados a los estándares de la certificación **Google Professional ML Engineer** (cero Data Leakage y particionado temporal estricto).
3. Predice la `replenishment_signal` para la ventana donde el sistema quedó "a ciegas".
4. Traduce el margen de error técnico en un impacto financiero estimado en pesos mexicanos (MXN).

---

## Estructura del Repositorio

```text
.
├── data/                 # transactions.csv, stores.csv, calendar.csv
├── data_dictionary.md    # Diccionario de datos completo
├── eda/                  # Pipeline Agéntico Autónomo de EDA (4 Fases)
│   ├── main.py           # Orquestador del EDA y reportes
│   └── agents/           # Agentes modularizados (overview, quality, profiling, auditor, report)
├── src/
│   ├── data_processing.py# Carga, grilla cartesiana (204k filas) e imputación contable
│   ├── features.py       # Lags, rolling windows, ratios y encodings sin Data Leakage
│   ├── models.py         # DemandForecaster (LightGBM & XGBoost)
│   ├── metrics.py        # WAPE, RMSE e Impacto Financiero en MXN
│   └── pipeline.py       # Pipeline end-to-end de entrenamiento y evaluación temporal
├── outputs/              # Artefactos: stats_raw.json, ml_results.json, audit_log.md
├── figures/              # 5 figuras en alta resolución (300 DPI)
├── reports/              # EDA_Report.md y EDA_Report.html autocontenido
├── PROCESS.md            # Documentación del proceso de trabajo y uso de IA
└── pyproject.toml
```

---

## Reproducibilidad (con `uv`)

```bash
# 1. Instalar dependencias
uv sync

# 2. Ejecutar el Pipeline Autónomo de EDA (Fases 1 a 4)
uv run python eda/main.py

# 3. Ejecutar el Pipeline de Machine Learning (Google ML Standards)
uv run python src/pipeline.py
```

---

## Resultados Principales e Impacto Financiero

Evaluado sobre el split de **VALIDATION (Diciembre 2023 – Enero 2024)** y predicho sobre el **HOLDOUT (Febrero 2024)**:

| Métrica | XGBoost Regressor | LightGBM Regressor |
|---------|───────────────────|────────────────────|
| **WAPE (%)** | **8.07%** | 9.38% |
| **RMSE** | **218.30** | 222.94 |
| **Costo de Sobrestock (MXN)** | $25,470,071.06 | $23,613,500.85 |
| **Costo de Quiebre (MXN)** | $35,372,784.88 | $53,108,868.58 |
| **Pérdida Financiera Total (MXN)** | **$60,842,855.94 MXN** | $76,722,369.43 MXN |
| **Ahorro Estimado vs Baseline (MXN)** | **~$27,379,285.17 MXN** | **~$34,525,066.24 MXN** |

---

## Certificación & Auditoría Fáctica
- **Google ML Engineer Standard:** Lags y rolling windows desplazados $\ge 1$ día; Target Encoding ajustado **únicamente en TRAIN**.
- **Audit Trail:** Todas las afirmaciones del reporte de EDA fueron auditadas por el Fact-Checker agent contra `outputs/stats_raw.json` (**10/10 verificaciones aprobadas, 0 alucinaciones**).
