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
4. **Imputación de Unidades Vendidas y Ticket:** Imputación por la mediana histórica agrupada por `(store_id, category, day_of_week)`, con *fallback* a mediana por categoría. **Las medianas se calculan exclusivamente sobre la ventana TRAIN (`date ≤ 2023-11-30`)** y luego se aplican a validación y holdout, evitando que estadísticas del futuro se filtren hacia la historia imputada (anti-leakage — ver `impute_pos_failures(train_end=...)`).
5. **Grilla Cartesiana Completa:** Reindexación cartesiana de $425 \text{ días} \times 80 \text{ tiendas} \times 6 \text{ categorías} = 204,000 \text{ filas}$ marcando `pos_outage_flag = 1` en días de caída total.

---

## 3. Decisiones de Arquitectura e Ingeniería de Features (Google ML Engineer Standards)

Para garantizar cero **Data Leakage**:
- **Lags de Inercia:** `replenishment_signal_lag_1d, 7d, 14d, 28d` y `units_sold_lag_1d, 7d`.
- **Estadísticas Móviles Desplazadas:** `units_sold_roll_mean_7d, 14d, 28d` y `units_sold_roll_std_7d, 14d` calculados sobre `shift(1)` (garantizando no usar datos contemporáneos del mismo día).
- **Ratios Operativos:** `sales_per_sqm`, `sales_per_checkout` y `cash_ratio_roll_30d` (dependencia histórica del efectivo a 30 días).
- **Proximidad a Quincena:** `is_payday_window` y `days_to_next_payday`.
- **Target Encodings Seguros:** Target encodings de `category` y `store_format` ajustados **únicamente en el split de TRAIN**.

### 3.1 Selección de Features "Gap-Safe" (alineada al caso de uso real)

El problema de negocio no es predecir *mañana* teniendo el dato de *hoy*: es predecir una **ventana de varios días consecutivos con `replenishment_signal` nula** al cierre de mes. Durante ese hueco, cualquier feature que dependa de datos *dentro* de la ventana (el `lag_1`, las medias móviles con `shift(1)`, los ratios sobre `amount_total_lag_1d`) **no existe**.

Por eso el modelo evaluado usa únicamente features **gap-safe**: cuyo insumo más reciente tiene ≥ `GAP_HORIZON` (=7) días de antigüedad. En la práctica: features estáticas de tienda/calendario/target-encoding + `lag_7d / 14d / 28d`. Se **excluyen** `lag_1`, rolling `shift(1)` y ratios contemporáneos. Esto convierte la evaluación en un *forecast honesto a 7 días* en vez de una persistencia de 1 paso.

---

## 4. Estrategia de Validación y Resultados (WAPE, Impacto MXN)

### Split Temporal Cronológico
- **`TRAIN` Split:** 2023-01-01 a 2023-11-30 (160,320 filas)
- **`VALIDATION` Split:** 2023-12-01 a 2024-01-31 (29,760 filas)
- **`HOLDOUT` Split (Prueba a Ciegas):** 2024-02-01 a 2024-02-29 (13,920 filas)

### Tarea de evaluación

Forecast a **7 días** con features **gap-safe**, comparado contra un baseline **seasonal-naive (lag-7)** — el baseline justo para un hueco de ≤7 días (el `lag-1` no es utilizable ahí, ver §3.1). El baseline se calcula, no se inventa.

### Resultados de Evaluación (VALIDATION)

| Modelo | WAPE (%) | RMSE | Pérdida Total (MXN) |
|--------|----------|------|---------------------|
| **Baseline seasonal-naive (lag-7)** | **22.64%** | — | **$170.5M** |
| LightGBM (gap-safe) | 32.62% | 406.62 | $244.1M |
| XGBoost (gap-safe) | 37.26% | 503.14 | $244.5M |

### Hallazgo honesto: el modelo aún NO supera al baseline

En el escenario realista, **ambos GBM pierden contra el seasonal-naive** (ahorro real ≈ **–$73.6M MXN**). La razón es diagnosticada, no misteriosa: al remover `lag_1` (que en la versión anterior concentraba >90% de la importancia y **no existe** durante el hueco), el modelo se queda con `lag_7/14/28` + calendario e intenta reaprender lo que `lag_7` ya codifica, agregando ruido.

> El WAPE de ~8% reportado en una iteración previa era un **espejismo de persistencia de 1 paso** apoyado en "el valor de ayer", más una métrica de ahorro con un factor arbitrario (`×0.45`). Ambos fueron **eliminados** por deshonestos. Se prefiere un resultado negativo auditable a una métrica inflada.

**Próximo paso para superar el baseline** (features multi-step reales, pendiente en otra iteración):
- Media estacional multi-semana (promedio de `lag_7, 14, 21, 28`) para suavizar el ruido del lag-7 único.
- Rolling con `shift(7)` (media/std de las últimas 4 semanas, gap-safe).
- Perfil `día-de-semana × categoría` y tendencia local.

Un **gate de CI** (`savings_best_model_vs_naive_mxn >= 0`) quedará documentado en el plan MLOps para impedir que un modelo peor que el baseline llegue a producción.

---

## 5. MLOps: Tracking, Model Registry y Gate de CI/CD

Dependencia única añadida: **`mlflow-skinny`** (no `mlflow` completo — ver nota técnica abajo). Backend local de archivos (`mlruns/`, gitignoreado y regenerable con `python src/pipeline.py`), sin servidor de tracking.

### 5.1 Tracking (`src/pipeline.py`)
Cada corrida de `run_ml_pipeline()` se envuelve en `mlflow.start_run()` bajo el experimento `walmart-replenishment` y registra:
- **Params:** hiperparámetros de LightGBM/XGBoost (`lgb_*`, `xgb_*`), `gap_horizon_days`, fechas de los splits, `n_features`.
- **Metrics:** `wape`/`rmse`/pérdida financiera por modelo y por el baseline, y `savings_best_model_vs_naive_mxn`.
- **Artifacts:** `ml_results.json`, las 5 figuras del EDA, `feature_cols.json`, feature importance del mejor LightGBM.

### 5.2 Model Registry
Ambos modelos (LightGBM y XGBoost) se loguean como artifacts de la corrida con firma (`infer_signature`) y `input_example`. El que gana en pérdida financiera de VALIDATION se registra en el Model Registry como **`walmart-replenishment`**:
- Alias **`@staging`**: siempre apunta a la versión más reciente registrada.
- Alias **`@production`**: solo se mueve si el modelo **supera al baseline seasonal-naive** (mismo criterio que el gate de CI, §5.3). Hoy no se mueve — el modelo actual pierde contra el baseline (§4).

**Nota técnica — por qué `mlflow-skinny` y no `mlflow`:** al 2026-07-21, incluso el release más reciente de `mlflow` (3.14.0) fija `pandas<3`, incompatible con `pandas>=3.0.3` de este proyecto. `mlflow-skinny` no fija pandas y expone la misma API de tracking/registry usada aquí (`mlflow`, `mlflow.lightgbm`, `mlflow.xgboost`, `MlflowClient`). Los modelos se serializan con `serialization_format="pickle"` en vez del default `"skops"` para no sumar una dependencia extra solo por eso.

**Para apuntar a un backend remoto (S3/GCS) en vez del filestore local:** basta con setear `MLFLOW_TRACKING_URI` (p. ej. `databricks`, `http://<servidor>`) y, para artifacts, `MLFLOW_ARTIFACT_LOCATION=s3://bucket/path` o `gs://bucket/path` al crear el experimento — sin tocar código.

### 5.3 CI/CD (`.github/workflows/ci.yml`)
En cada PR y push a `main`/`master`: `uv sync` → `ruff check` → `python src/pipeline.py` (entrena, trackea en MLflow, registra el modelo) → `pytest` → sube `ml_results.json` como artifact.

**Gate champion/challenger** (`tests/test_model_gate.py`): `assert savings_best_model_vs_naive_mxn >= 0`. Si el mejor modelo no le gana al seasonal-naive, CI queda en rojo. **Hoy falla a propósito** — es el guardrail que impide que se repita el espejismo del §4 (una métrica inflada pasando a producción sin más control que "se ve bien").

---

## 6. Asistencia de Herramientas de IA (Antigravity AI Agent)

Se utilizó el sistema agéntico **Antigravity AI** (Google DeepMind team) como pareja de Pair Programming para:
1. **Fase 1 (EDA Autónomo en 4 Fases):** Creación del pipeline modular en `eda/` con profiling, fact-checking y generación de reportes (`EDA_Report.md` y `EDA_Report.html`).
2. **Fase 2 (Data Engineering):** Construcción de la matriz cartesiana maestra en `src/data_processing.py`.
3. **Fase 3 (ML Pipeline):** Implementación de la arquitectura de Machine Learning sin data leakage, validación temporal y cálculo del impacto financiero en MXN.

Todo el código generado fue validado mediante pruebas de ejecución automatizadas y auditoría fáctica con 0 alucinaciones (`outputs/audit_log.md`).

### Herramientas de Apoyo al Flujo de Trabajo

Además del agente principal, el desarrollo se apoyó en dos herramientas que dejan claro **de qué lado masca la iguana** 🦎 — cero over-engineering y trazabilidad total:

- **graphify** (`/graphify`): convierte todo el repositorio (código, docs y figuras del EDA) en un **grafo de conocimiento** navegable con detección de comunidades y auditoría honesta de relaciones (EXTRACTED / INFERRED / AMBIGUOUS). Se usó para mapear las dependencias reales entre el pipeline de datos, la ingeniería de features y los agentes de EDA, y para responder preguntas sobre la arquitectura sin releer archivo por archivo. Sus salidas viven en `graphify-out/` (regenerables, por eso están en `.gitignore`).
- **ponytail** (`/ponytail`): guardián anti-sobreingeniería. Fuerza la solución más simple que funciona — biblioteca estándar antes que dependencias, una línea antes que cincuenta, y cuestiona si el código siquiera necesita existir (YAGNI). Se aplicó para mantener el pipeline limpio y sin abstracciones especulativas.
