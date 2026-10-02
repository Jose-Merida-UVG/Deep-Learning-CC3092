# Proyecto 2: Detección de lavado de dinero en remesas

CC3092 Deep Learning y Sistemas Inteligentes, UVG.

**Demo (MVP):** https://claude.ai/artifact/TKe6FkAd5X1zkJtM2yKPfa

## Resumen

Sistema de dos etapas que ordena a los remitentes de una remesadora según qué tan probable es que estén lavando dinero, y que señala las transacciones que provocaron cada alerta.

- **Etapa A.** Un autoencoder aprende cómo se ve el comportamiento normal, sin etiquetas. Se entrena solo con remitentes que nunca lavaron, y el error de reconstrucción de una secuencia es su puntaje de anomalía. Se compararon tres encoders: MLP, LSTM y Transformer.
- **Etapa B.** Un clasificador supervisado parte del encoder de la Etapa A. Se probaron tres estrategias de transfer learning (desde cero, congelado y ajustado) con LSTM y con Transformer. La pérdida es focal loss, con 40 negativos por cada positivo en cada época.
- **Interpretabilidad.** El attention pooling da un peso por transacción, que se usa como mapa de calor de la alerta.

## Datos

IBM AML (HI-Small), un conjunto sintético con 5,078,345 transacciones. Solo se usan `HI-Small_Trans.csv` y `HI-Small_Patterns.txt`, disponibles en https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Cada remitente se divide en ventanas de 32 transacciones. La división es 70/15/15 por remitente, para que ventanas solapadas no queden en conjuntos distintos.

## Resultados principales (conjunto de prueba)

| Modelo | PR-AUC | Recall @5% | Precisión @5% |
|---|---|---|---|
| Etapa A sola (mejor, MLP) | 0.243 | 0.47 | 0.09 |
| LSTM desde cero | 0.582 | 0.79 | 0.15 |
| **Transformer ajustado (final)** | **0.572** | **0.78** | **0.15** |

Al revisar el 5% de los remitentes más riesgosos, el modelo final encuentra 78% de los lavadores, y alrededor de 15% de esas alertas son reales. El sistema de dos etapas no supera a un clasificador entrenado desde cero. Lo iguala, y el efecto de preentrenar con la Etapa A es inconsistente entre arquitecturas. Los datos son sintéticos, así que las cifras no se trasladan directamente a remesas reales.

## Archivos

| Archivo | Contenido |
|---|---|
| `Notebook.ipynb` | Ingeniería de datos, Etapa A, Etapa B con ablación e interpretabilidad |
| `Informe.tex`, `Informe.pdf` | Reporte ejecutivo |
| `referencias.bib` | Referencias del reporte |
| `AI.md` | Uso de IA generativa |
| `artifact.txt`, `MVP.txt` | Enlace al demo |
| `figs/` | Figuras de los casos de estudio usadas en el reporte |
| `aml_outputs/` | Salidas procesadas del notebook |
| `requirements.txt` | Dependencias |

## Cómo correr el notebook

1. Instalar las dependencias con `pip install -r requirements.txt`, o usar Colab con GPU.
2. Conseguir los dos archivos del dataset. En Colab, subirlos a `/content`. Localmente, definir `AML_DATA_DIR` con la carpeta que los contiene.
3. Ejecutar el notebook de principio a fin. Las salidas se guardan en `aml_outputs/` (en Colab, en `/content/aml_outputs`).
