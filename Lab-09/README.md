# Laboratorio 9 — VAE y modelos de difusión (Fashion-MNIST)

## Summary

Laboratorio sobre dos familias de modelos generativos implementadas desde cero con tensores PyTorch sobre Fashion-MNIST: un autoencoder variacional y un modelo de difusión condicional con guía sin clasificador. Además se verifican numéricamente resultados derivados en clase: la forma cerrada del proceso forward, la KL entre gaussianas, la guía sin clasificador y el estimador de pass@k.

- **Task 1 — VAE**: VAE completamente conectado (`latent_dim = 2`) entrenado con $\beta \in \{0.1, 1, 10\}$; reparametrización, KL cerrada y pérdida escritas a mano. Dispersión del espacio latente, cuadrícula decodificada, muestras, interpolaciones en píxeles vs. espacio latente, KL en bits, verificación Monte Carlo de la KL y análisis de nitidez.
- **Task 2 — Proceso forward**: calendario lineal ($T = 1000$), fórmula cerrada de $x_t$ verificada empíricamente con 5 000 copias, curvas de $\bar\alpha_t$ y SNR, un segundo calendario (coseno, Nichol & Dhariwal 2021) y análisis de la aproximación de $\bar\alpha_T$, del rango útil de SNR y de la varianza de $x_t$.
- **Task 3 — Difusión condicional**: U-Net pequeña (28×28 / 14×14, embedding sinusoidal de $t$ + `nn.Embedding(11, d)` para la clase), entrenamiento con `p_uncond = 0.1`, muestreo ancestral, guía sin clasificador con $w \in \{1, 3, 7\}$, y fidelidad/diversidad medidas con un clasificador auxiliar.
- **Task 4 — Comparación y pass@k**: nitidez y tiempo de generación VAE vs. difusión, y estimador insesgado de pass@k vs. la versión ingenua.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Notebook.ipynb` | Notebook completo: Task 1 a Task 4, con encabezados de Pregunta/Respuesta para cada pregunta del enunciado. |
| `Informe.tex` / `Informe.pdf` | Informe con las respuestas a cada task, generado a partir del notebook. |
| `referencias.bib` | Referencias del informe. |
| `figs/` | Figuras del notebook usadas en el informe. |
| `AI.md` | Detalle del uso de IA generativa en este laboratorio. |
| `requirements.txt` | Dependencias de Python. |
| `S14 - Lab9.pdf` | Enunciado del laboratorio. |

`data/` (Fashion-MNIST) y `ckpt/` (pesos entrenados) se generan al correr el notebook y están en `.gitignore`.

## Resultados

- **VAE**: al subir $\beta$ la reconstrucción empeora (19.8 → 21.2 → 36.8) y la KL baja (11.9 → 7.7 → 2.5 bits); con $\beta = 10$ la KL queda por debajo de los $\log_2 10 \approx 3.32$ bits necesarios para identificar la clase. La KL Monte Carlo coincide con la cerrada (error relativo $2.4\times10^{-5}$).
- **Proceso forward**: el error máximo de la media empírica es ≈3 SE (lo esperado para el máximo de 784 píxeles). El calendario coseno tiene 880 pasos con SNR en $[0.01, 100]$ contra 647 del lineal.
- **Difusión**: U-Net de 364,257 parámetros. La fidelidad sube de 0.784 ($w = 1$) a 0.952 ($w = 3$) y baja a 0.878 ($w = 7$), mientras la diversidad cae de forma monótona (7.94 → 7.88 → 7.59).
- **VAE vs. difusión**: la difusión alcanza el 83 % de la nitidez real contra el 57 % del VAE, pero es ~6300× más lenta (1000 evaluaciones de red contra 1).
- **pass@k**: promedio de pass@1 = 0.417 y de pass@10 = 0.791; para P2 el menor $k$ con pass@$k > 0.9$ es 14.

## AI Usage

Ver [`AI.md`](./AI.md) para el detalle de uso de IA en este laboratorio (alcance, prompts y justificaciones).

## Execution

Requiere Python 3.10+ (se recomienda GPU). Instalar dependencias y abrir el notebook:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
jupyter lab Notebook.ipynb
```

Fashion-MNIST se descarga automáticamente en `data/` en la primera ejecución. Los modelos entrenados se guardan en `ckpt/` y se reutilizan en ejecuciones posteriores; para reentrenar, borre `ckpt/` o ponga `FORCE_RETRAIN = True`.

Para compilar el informe:

```bash
latexmk -pdf -outdir=.tex Informe.tex && cp .tex/Informe.pdf .
```
