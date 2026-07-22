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
│   └── pipeline.py       # Pipeline end-to-end: entrenamiento, evaluación y tracking MLflow
├── tests/                # Gate champion/challenger (CI)
├── .github/workflows/    # CI: lint, pipeline, gate, artifact
├── outputs/              # Artefactos: stats_raw.json, ml_results.json, audit_log.md
├── figures/              # 5 figuras en alta resolución (300 DPI)
├── reports/              # EDA_Report.md y EDA_Report.html autocontenido
├── mlruns/               # Tracking store local de MLflow (gitignoreado, regenerable)
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

# 3. Ejecutar el Pipeline de Machine Learning (Google ML Standards, con tracking MLflow)
uv run python src/pipeline.py

# 4. Correr los tests (incluye el gate champion/challenger de CI)
uv run pytest tests/ -v
```

---

## Resultados Principales e Impacto Financiero

Evaluación **gap-aware**: forecast a **7 días** con features *gap-safe* (sin `lag_1` ni rolling contemporáneo — no existen durante la ceguera de inventario), sobre el split de **VALIDATION (Diciembre 2023 – Enero 2024)**. Comparado contra un baseline **seasonal-naive (lag-7)**, calculado, no inventado.

| Modelo | WAPE (%) | RMSE | Pérdida Financiera Total (MXN) |
|--------|----------|------|---------------------------------|
| **Baseline seasonal-naive (lag-7)** | **22.64%** | — | **$170.5M** |
| LightGBM (gap-safe) | 32.62% | 406.62 | $244.1M |
| XGBoost (gap-safe) | 37.26% | 503.14 | $244.5M |

**Hallazgo honesto:** en este escenario realista, ambos modelos GBM pierden contra el baseline (ahorro real ≈ **–$73.6M MXN**). Detalle del diagnóstico y próximos pasos para superarlo en `PROCESS.md §4`. Un gate de CI (`tests/test_model_gate.py`) falla a propósito mientras esto siga así — ver `PROCESS.md §5`.

---

## MLOps: Tracking, Model Registry y CI/CD

- **Tracking:** cada corrida de `src/pipeline.py` se registra en MLflow (`mlruns/`, backend local de archivos) con params, métricas e artifacts.
- **Model Registry:** el mejor modelo se registra como `walmart-replenishment`; alias `@staging` siempre, `@production` solo si supera al baseline.
- **CI (`.github/workflows/ci.yml`):** `ruff check` + `python src/pipeline.py` + `pytest` (gate champion/challenger) en cada PR.

### Inferencia con el modelo registrado

Tras correr `src/pipeline.py` (que registra el modelo), carga el candidato `@staging` desde el
Model Registry local y predice sobre una matriz de features **gap-safe**:

```python
import mlflow

mlflow.set_tracking_uri("file:mlruns")
model = mlflow.pyfunc.load_model("models:/walmart-replenishment@staging")

# X_new: DataFrame con las columnas de outputs/feature_cols.json (estáticas + lags >= 7d)
preds = model.predict(X_new)  # replenishment_signal estimada para la ventana ciega
```

> El alias `@production` solo existe cuando el modelo supera al baseline (hoy no — ver §4); en
> producción se cargaría `models:/walmart-replenishment@production` con el mismo código.

Detalle completo en `PROCESS.md §5`.

---

## Certificación & Auditoría Fáctica
- **Google ML Engineer Standard:** Lags y rolling windows desplazados $\ge 1$ día; Target Encoding ajustado **únicamente en TRAIN**.
- **Audit Trail:** Todas las afirmaciones del reporte de EDA fueron auditadas por el Fact-Checker agent contra `outputs/stats_raw.json` (**10/10 verificaciones aprobadas, 0 alucinaciones**).
