# Laboratorio 7 — Forecasting con LSTM y Transformer desde cero

## Summary

Sistema de forecasting sobre el dataset ETTh1 (Electricity Transformer Temperature) implementado con tensores PyTorch puros — sin `nn.LSTM` ni `nn.MultiheadAttention`. Se predice la temperatura del aceite (`OT`) a dos horizontes (`H=24` y `H=48` horas) con un LSTM many-to-one manual y un Transformer encoder con self-attention manual, comparando ambos contra un baseline naive.

- **Task 1 — Preprocesamiento y ACF**: descarga de ETTh1, normalización, split temporal contiguo 60/20/20, `make_windows` para construir pares `(X, y)`, y `acf_manual` para calcular la autocorrelación y justificar la elección de `L=96`.
- **Task 2 — LSTMForecaster**: LSTM manual (forward paso a paso, sin `nn.LSTM`) con conexión residual al último valor observado (`pred = x_last + delta`), entrenado con Adam + MSE. Incluye una demostración explícita (`demo_reg_*`) comparando una versión sin regularización (aprende más, generaliza peor) contra la versión oficial con `weight_decay=0.01` (generaliza mejor y pasa la verificación).
- **Task 3 — TransformerForecaster**: positional encoding sinusoidal, multi-head self-attention manual (`WQ`/`WK`/`WV`/`WO` como `nn.Parameter`), LayerNorm y FFN manuales, entrenamiento para ambos horizontes, y extracción/visualización de los mapas de atención promedio.
- **Task 4 — Análisis comparativo**: comparación numérica entre los picos de la ACF y los lags con mayor atención, tabla final LSTM vs. Transformer vs. naive, y preguntas de análisis sobre acumulación de error y flujo de gradiente.

## Deliverables

| File | Description |
| :--- | :---------- |
| `TimeSeries.ipynb` | Notebook completo: Task 1 a Task 4, celdas de verificación provistas, demo de regularización, y respuestas de análisis. |
| `hparam_search.py` | Script standalone para comparar distintas configuraciones de `weight_decay`/`lr` del LSTM y graficar sus curvas de pérdida. |

## Resultados

- **Task 1**: `rho(24)=0.9250` confirma estacionalidad diaria; naive MAE en H=24 = `0.1996`.
- **Task 2**: LSTM con conexión residual + `weight_decay=0.01` — H=24 MAE=`0.2022` (ratio=`1.013`), H=48 MAE=`0.2693` (ratio=`1.034`). La versión sin regularización aprende más en train pero generaliza peor (ratio=`1.293`, no pasa la verificación).
- **Task 3**: Transformer — H=24 MAE=`0.2357` (ratio=`1.180`), H=48 MAE=`0.2705` (ratio=`1.039`). Su atención se concentra en los lags 1-6, no en los picos estacionales de la ACF (22/24/47/48/72).
- **Task 4**: el LSTM supera claramente al Transformer en H=24, pero la brecha casi desaparece en H=48 — consistente con el argumento de flujo de gradiente (camino O(1) en atención vs. producto de jacobianos en la recurrencia).

## AI Usage

Ver [`AI.md`](./AI.md) para el detalle de uso de IA en este laboratorio (prompts y justificaciones).

## Execution

Requiere Python 3.10+ con PyTorch, NumPy, Matplotlib y Jupyter (ver `.venv` / `requirements.txt`). El dataset ETTh1 (~2.5MB) se descarga automáticamente a `./ETTh1.csv` en la primera ejecución.

```bash
jupyter notebook TimeSeries.ipynb
```
