# Proceso de Trabajo

## 1. Definición del Problema de Negocio

_Qué problema de negocio se eligió resolver y por qué (ej. forecast de
demanda para optimizar reposición de inventario, reducción de quiebres de
stock, etc.)._

## 2. Limpieza e Imputación de Datos (Fallas de POS)

_Cómo se identificaron y trataron los nulos causados por fallas de POS y
conectividad (`cash_transactions`, `amount_cash`, `units_sold`,
`avg_ticket`), y tiendas con periodos completos sin datos._

## 3. Decisiones de Arquitectura e Ingeniería de Features

_Justificación de las features construidas (lags, rolling windows, variables
de calendario) y del modelo elegido (LightGBM/XGBoost)._

## 4. Estrategia de Validación y Resultados (WAPE, Impacto MXN)

_Esquema de validación (split temporal, cross-validation por tienda/fecha),
métricas obtenidas (WAPE, RMSE) y su traducción a impacto financiero en MXN._

## 5. Asistencia de Herramientas de IA (Uso de LLMs/Claude Code)

_Qué partes del proyecto se generaron o apoyaron con LLMs/Claude Code, y qué
se validó/ajustó manualmente._
