# Laboratorio 8 — Mini-GPT, estrategias de muestreo y embeddings contextuales

## Summary

Laboratorio en dos partes. En la primera se entrena un Mini-GPT a nivel de carácter sobre `tiny_shakespeare` (reutilizando el `MiniGPT` de la Semana 6, hecho con tensores PyTorch puros) y se implementan desde cero tres estrategias de muestreo. En la segunda se extraen embeddings contextuales de BERT multilingüe y se comparan contra vectores estáticos de Word2Vec.

- **Task 1 — Mini-GPT y muestreo**: vocabulario de 65 caracteres, partición 90/10 en orden temporal, batches donde el objetivo es la entrada desplazada un paso, entrenamiento con AdamW (`lr=3e-4`, warmup lineal, recorte de gradiente), curvas de pérdida y perplejidad de validación. Muestreo *greedy*, *top-k* con temperatura y *top-p* con temperatura, escritos desde cero.
- **Task 2 — Generación y análisis**: `generate(prompt, max_new_tokens, strategy, **kwargs)` y 5 muestras de 200 caracteres para cada una de las configuraciones A–E partiendo de `ROMEO:`, con un análisis de las muestras (ciclos de greedy, efecto de `k` y `τ`, adaptatividad de top-p).
- **Task 3 — Embeddings contextuales**: embeddings de la última capa de `bert-base-multilingual-cased` para `banco`, `copa` y `vela` en 7 oraciones, similitud coseno entre contextos, comparación contra Word2Vec en español (`gensim`) y visualización con PCA.
- **Task 4 — Preguntas conceptuales**: por qué GPT necesita la máscara causal y cómo el paradigma de preentrenar y adaptar permite elegir entre GPT y BERT según la tarea.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Notebook.ipynb` | Notebook completo: Task 1 a Task 4, con celdas de verificación (vocabulario, causalidad, perplejidad, funciones de muestreo) y las respuestas de análisis. |
| `AI.md` | Detalle del uso de IA generativa en este laboratorio. |
| `requirements.txt` | Dependencias de Python. |
| `input.txt` | Corpus `tiny_shakespeare` (se descarga automáticamente; está en `.gitignore`). |

## Resultados

- **Entrenamiento**: una época es una pasada completa sobre el conjunto de entrenamiento (123 pasos con `batch_size=64`, bloques no solapados barajados en cada época). Con las 10 épocas del enunciado la perplejidad de validación fue ≈8.24, fuera del rango esperado [3.5, 8.0] y con la pérdida aún bajando. Como el enunciado llama *mínimos* a los hiperparámetros, se entrenó con **35 épocas**: perplejidad de validación **5.844**, dentro del rango.
- **BERT (similitud coseno entre contextos de la misma palabra)**: *banco* 0.58 / 0.66 / 0.65, *copa* 0.57, *vela* 0.80. Word2Vec da 1.0 en todos los pares, porque asigna un único vector por palabra.
- **PCA**: los puntos se agrupan principalmente por palabra; los significados de una misma palabra se separan solo parcialmente (los dos de *vela* casi se superponen).

## AI Usage

Ver [`AI.md`](./AI.md) para el detalle de uso de IA en este laboratorio (alcance, prompts reconstruidos y justificaciones).

## Execution

Requiere Python 3.10+ (se recomienda GPU, pero funciona en CPU). Instalar dependencias y abrir el notebook:

```bash
pip install -r requirements.txt
jupyter notebook Notebook.ipynb
```

En la primera ejecución se descargan automáticamente: el corpus `tiny_shakespeare`, el modelo `bert-base-multilingual-cased` (~700 MB) y los vectores Word2Vec en español `Word2vec/wikipedia2vec_eswiki_20180420_300d` desde el Hub de HuggingFace (~4 GB), por lo que esa primera ejecución requiere conexión a internet y espacio en disco.
