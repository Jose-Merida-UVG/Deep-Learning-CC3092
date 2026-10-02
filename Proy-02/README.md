# Proyecto 2: Detección de lavado de dinero en remesas

CC3092 Deep Learning y Sistemas Inteligentes, UVG.

**Demo (MVP):** https://claude.ai/artifact/TKe6FkAd5X1zkJtM2yKPfa

## Resumen

Sistema de dos etapas que ordena a los remitentes de una remesadora según qué tan probable es que estén lavando dinero, y que señala las transacciones que provocaron cada alerta.

- **Etapa A.** Un autoencoder aprende cómo se ve el comportamiento normal, sin etiquetas. Se entrena solo con remitentes que nunca lavaron, y el error de reconstrucción de una secuencia es su puntaje de anomalía. Se compararon tres encoders: MLP, LSTM y Transformer.
- **Etapa B.** Un clasificador supervisado parte del encoder de la Etapa A. Se probaron tres estrategias de transfer learning (desde cero, congelado y ajustado) con LSTM y con Transformer. La pérdida es focal loss, con 40 negativos por cada positivo en cada época.
- **Interpretabilidad.** El attention pooling da un peso por transacción, que se usa como mapa de calor de la alerta.

## Datos

IBM AML (HI-Small), un conjunto sintético con 5,078,345 transacciones. Se usa la variante HI-Small completa, sin submuestreo adicional. Es suficiente para entrenar, con más de cinco millones de transacciones y 3,376 remitentes que lavan en algún momento. Solo se usan `HI-Small_Trans.csv` y `HI-Small_Patterns.txt`, disponibles en https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

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
| `Notebook.ipynb` | Ingeniería de datos, Etapa A, Etapa B con ablación e interpretabilidad. Versión corrida en local |
| `Notebook-Colab.ipynb` | El mismo notebook, con la celda de carga de datos ajustada para Colab y con las salidas de la corrida en una T4 |
| `Informe.tex`, `Informe.pdf` | Reporte ejecutivo |
| `referencias.bib` | Referencias del reporte |
| `AI.md` | Uso de IA generativa |
| `MVP.txt` | Enlace al demo |
| `figs/` | Figuras de los casos de estudio usadas en el reporte |
| `aml_outputs/` | Salidas procesadas del notebook |
| `requirements.txt` | Dependencias |

## Tiempo de ejecución

| Entorno | GPU | Tiempo total |
|---|---|---|
| Local (`Notebook.ipynb`) | NVIDIA GeForce RTX 4070 SUPER | 12.3 min |
| Colab (`Notebook-Colab.ipynb`) | Tesla T4 | 36.9 min (cerca de 38 min medidos a mano) |

El enunciado pide que el notebook corra en menos de 30 minutos en una T4 gratuita de Colab. En Colab tardó más que eso. Se esperaba que cupiera en el límite porque en el equipo local, que es más potente que una T4, tardó menos de 15 minutos. La mayor parte del tiempo es el entrenamiento de la Etapa A, que en la corrida local terminó cerca del minuto 10 de 12.

## Cómo correr el notebook

La única diferencia entre las dos versiones es la celda que carga los datos (la celda 4, la que empieza con `DATASET = ...`). En `Notebook-Colab.ipynb` esa celda busca los archivos en `AML_DATA_DIR`, `/kaggle/input`, `/content` y la carpeta actual, y solo si no los encuentra intenta descargarlos con `kagglehub`. La descarga con `kagglehub` falló en Colab, por eso en Colab hay que subir los archivos.

### En Colab

1. Abrir `Notebook-Colab.ipynb` y elegir un entorno con GPU (Entorno de ejecución, Cambiar tipo de entorno, T4).
2. Descargar `HI-Small_Trans.csv` y `HI-Small_Patterns.txt` desde https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (hay que iniciar sesión en Kaggle).
3. Subir los dos archivos al panel de archivos de Colab (el ícono de carpeta a la izquierda). Quedan en `/content`, que es donde la celda los busca. Esperar a que termine la subida, porque el CSV pesa cerca de 475 MB.
4. Ejecutar el notebook de principio a fin. Las salidas se guardan en `/content/aml_outputs`. Colab borra esa carpeta al cerrar la sesión, así que hay que descargarla si se necesita.

Los archivos subidos también se borran al cerrar la sesión, así que hay que subirlos de nuevo cada vez.

### En local

1. Instalar las dependencias con `pip install -r requirements.txt`.
2. Conseguir los dos archivos del dataset y definir `AML_DATA_DIR` con la carpeta que los contiene, o dejarlos en la carpeta del notebook.
3. Ejecutar `Notebook.ipynb` de principio a fin. Las salidas se guardan en `aml_outputs/`.
