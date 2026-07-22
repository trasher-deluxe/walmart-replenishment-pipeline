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

Evaluación **gap-aware**: forecast a **7 días** con cross-validation rolling-origin sobre las **480 series** (store × category), contra un baseline **seasonal-naive (lag-7)** calculado, no inventado. El modelo de **producción** es **AutoETS** (statsforecast).

| Modelo | WAPE | vs baseline |
|--------|------|-------------|
| **AutoETS (PRODUCCIÓN)** | **16.0%** | **−7.3 pts · ahorro +$48.0M MXN** ✅ |
| SeasonalNaive (baseline) | 23.3% | — |
| LightGBM tabular (investigación) | 32.6% | pierde |
| XGBoost tabular (investigación) | 37.3% | pierde |

**La decisión clave fue de paradigma.** El GBM tabular **no** le gana al naive —los árboles no extrapolan tendencia—; un forecaster nativo (**ETS**: nivel + tendencia + estacionalidad) sí, por ~7 puntos de WAPE — el mismo enfoque que corre bajo AWS Forecast / GCP Vertex. El gate champion/challenger (`tests/test_model_gate.py`) está en **verde** y AutoETS se promueve a `@production`. Detalle e investigación completa en `PROCESS.md §4`.

---

## MLOps: Tracking, Model Registry y CI/CD

- **Tracking:** cada corrida de `src/pipeline.py` se registra en MLflow (`mlruns/`, backend local de archivos) con params, métricas e artifacts.
- **Model Registry:** AutoETS se registra como `walmart-replenishment`; alias `@staging` siempre, `@production` solo si supera al baseline (hoy **sí** → `@production` activo).
- **CI (`.github/workflows/ci.yml`):** `ruff check` + `python src/pipeline.py` + `pytest` (gate champion/challenger, en **verde**) en cada PR.

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
