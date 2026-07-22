# AGENTS.md

Instrucciones para cualquier asistente de código (Claude Code, Cursor, Copilot, Aider, etc.)
que trabaje en este repositorio. Complemento generalista de `PROCESS.md` (metodología) y de los
subagentes en `agents/` (definiciones de rol tool-neutral, usables por cualquier asistente — el
equivalente portable de un `.claude/agents/`).

## Proyecto

**Walmart Supply Chain Resilience** — cadena de retail en México. Dos problemas:
1. Imputar fallas de POS (≈5.94% de nulos en efectivo) con identidades contables.
2. Predecir `replenishment_signal` durante la ceguera de inventario al cierre de mes (varios días
   consecutivos con la señal en nulo).

Datos en `data/` (`transactions.csv`, `stores.csv`, `calendar.csv`); diccionario en
`data_dictionary.md`.

## Comandos

Gestor de entorno: **uv** (Python 3.11, pin en `.python-version`, deps en `uv.lock`).

```bash
uv sync                          # instala deps (incluye el grupo dev: pytest)
uv run python eda/main.py        # pipeline EDA multi-agente -> reports/, figures/, outputs/
uv run python src/pipeline.py    # pipeline ML -> outputs/ml_results.json + tracking MLflow (mlruns/)
uv run pytest -q                 # suite de tests (incluye el gate champion/challenger)
uv run ruff check .              # lint (debe pasar limpio)
```

## Arquitectura

- `src/data_processing.py` — carga, grilla cartesiana (204k filas), imputación POS. `merge_master_dataset(train_end=...)`.
- `src/features.py` — lags, rolling, ratios, target encoders. Punto de entrada `build_features_pipeline`.
- `src/models.py` — `DemandForecaster` (wrapper LightGBM/XGBoost, investigación tabular).
- `src/forecasting.py` — **AutoETS** (statsforecast), el forecaster de **producción** que supera al baseline (`evaluate_production`, `fit_production`).
- `src/metrics.py` — `wape`, `rmse`, `business_impact_mxn`.
- `src/pipeline.py` — orquestador ML end-to-end + MLflow. Constantes fuente-de-verdad: `TRAIN_END`, `VAL_*`, `HOLDOUT_*`, `GAP_HORIZON`.
- `eda/` — pipeline EDA de 9 agentes en 4 fases (ver `agents/1_eda_multiagent.md`).
- `agents/` — definiciones de subagentes del proyecto (tool-neutral), en orden de construcción:
  `1_eda_multiagent`, `2_feature_engineer`, `3_ml_pipeline`, `4_leakage_auditor`, `5_mlops_ci`,
  `6_test_suite`.
- `tests/` — `pytest`; `.github/workflows/ci.yml` — CI (lint + pipeline + tests).

## Reglas no negociables (así se decidió y así se mantiene)

1. **Cero data leakage.** Lags y rolling siempre desplazados (`shift(1)+`); target encoders y
   medianas de imputación ajustados **solo con TRAIN** (`train_end`). Hay tests que lo prueban
   (`tests/test_data_leakage.py`) — no los rompas.
2. **Evaluación gap-aware.** El caso real es un hueco multi-día; el modelo solo usa features
   **gap-safe** (estáticas + lags ≥ `GAP_HORIZON`), nunca `lag_1`, rolling ni ratios
   contemporáneos. Baseline honesto: seasonal-naive `lag-7`.
3. **Honestidad sobre métricas.** Nada de números inventados ni factores arbitrarios. El GBM tabular
   **pierde** contra el baseline (los árboles no extrapolan); el modelo de producción **AutoETS lo
   supera** por ~7 pts WAPE (`PROCESS.md §4`). Se reporta tal cual — el criterio de model selection
   fue el que resolvió, no inflar una métrica.
4. **MLOps.** Tracking + Model Registry con MLflow (`mlruns/`, local). El modelo de producción es
   **AutoETS**; `@production` solo si supera al baseline (`passes_baseline_gate`). El gate de CI es un
   `assert` duro y hoy está en **verde** (AutoETS gana).
5. **Política de artefactos.** Los generados no se versionan (`outputs/ml_results.json`, `mlruns/`,
   `graphify-out/`, `outputs/draft_analysis.md`) — **excepto** `reports/`, `figures/` y
   `stats_raw.json`, commiteados a propósito para el revisor (ver `PROCESS.md §5.4`).
6. **Ponytail / YAGNI.** La solución mínima que funciona; biblioteca estándar antes que
   dependencias; una sola dependencia nueva por necesidad real (p. ej. `mlflow-skinny`).

## Convenciones

- Docs, comentarios y mensajes de commit en **español**; un commit por unidad lógica.
- Todo número en un reporte debe trazarse a `outputs/stats_raw.json` (auditado por el fact-checker).
- Antes de dar algo por terminado: `uv run ruff check .` y `uv run pytest -q` en verde.
