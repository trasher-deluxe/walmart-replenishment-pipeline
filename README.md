# Retail Demand Forecasting & POS Data Recovery

## Problema de Negocio

Cadena de retail con presencia en múltiples regiones de México. El dataset de
transacciones diarias por tienda/categoría presenta huecos causados por
fallas de POS y problemas de conectividad, lo que degrada la calidad de las
decisiones de reposición de inventario. Este proyecto:

1. Reconstruye una serie confiable a partir de datos con fallas de captura.
2. Entrena un modelo de forecast de demanda (`amount_total`) por
   tienda/categoría/día.
3. Traduce el error de forecast a impacto financiero en MXN (costo de
   sobre-stock vs. quiebre de stock).

## Estructura del Repositorio

```
.
├── data/                 # transactions.csv, stores.csv, calendar.csv
├── data_dictionary.md    # diccionario de datos
├── notebooks/            # EDA
├── src/
│   ├── data_processing.py  # carga, merge, imputación de nulos por fallas de POS
│   ├── features.py         # lags, rolling windows, variables de calendario
│   ├── models.py           # entrenamiento/predicción con LightGBM/XGBoost
│   └── metrics.py          # WAPE, RMSE, impacto de negocio en MXN
├── outputs/              # artefactos: modelos, gráficos, métricas
├── main.py               # pipeline end-to-end
├── PROCESS.md             # proceso de trabajo y uso de herramientas de IA
└── pyproject.toml
```

## Reproducibilidad (con `uv`)

```bash
# 1. Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Instalar dependencias (crea .venv automáticamente)
uv sync

# 3. Correr el pipeline end-to-end
uv run main.py

# (Opcional) Levantar Jupyter para el EDA
uv run jupyter notebook notebooks/
```

## Resultados Principales e Impacto Financiero

_Pendiente de completar tras ejecutar el pipeline: WAPE, RMSE, y costo
estimado en MXN de sobre-stock/quiebre de stock evitado._

| Métrica | Valor |
|---------|-------|
| WAPE | — |
| RMSE | — |
| Impacto financiero (MXN) | — |
