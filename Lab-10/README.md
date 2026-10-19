# Laboratorio 10 — Explicabilidad (Grad-CAM / CAM, SHAP / LIME / contrafactuales)

## Summary

Laboratorio sobre explicabilidad de modelos en dos dominios: visión (ResNet-18 preentrenado sobre tres imágenes de prueba) y datos tabulares (gradient boosting sobre German Credit). Los métodos centrales se implementan desde cero y se verifican numéricamente; la última parte valida los mapas de Grad-CAM con pruebas causales y de aleatorización.

- **Task 1 — Grad-CAM y CAM**: Grad-CAM sobre `layer4` con hooks (retropropagando desde el logit), CAM desde los pesos de `fc` y verificación de que ambos coinciden salvo por la constante $Z = 49$. Discriminación entre dos clases en una misma imagen (IoU de las regiones top-20 %) y prueba causal con el fondo en gris sobre un objeto fuera de contexto.
- **Task 2 — SHAP**: TreeSHAP intervencional sobre German Credit (eficiencia y valor base verificados), comparación de la importancia global con la de permutación, efecto de duplicar una variable con ruido, KernelSHAP vs. TreeSHAP, Shapley exacto escrito a mano (6 variables) y análisis de costo computacional.
- **Task 3 — LIME y contrafactuales**: LIME tabular desde cero (muestreo, pesos, ridge ponderado en forma cerrada, $R^2_w$, $N_{ef}$), estabilidad con Jaccard, comparación LIME vs. SHAP y búsqueda de contrafactuales por muestreo con variables inmutables y accionables.
- **Task 4 — Validación de Grad-CAM**: curvas de deletion e insertion contra órdenes aleatorios, aleatorización de parámetros en cascada (sanity check de Adebayo et al.) y discusión de qué explicación entregar a cada audiencia.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Lab10_XAI.ipynb` | Notebook completo: Task 1 a Task 4, con encabezados de Pregunta/Respuesta para cada pregunta del enunciado. |
| `Informe.tex` / `Informe.pdf` | Informe con las respuestas a cada task, generado a partir del notebook. |
| `referencias.bib` | Referencias del informe. |
| `figs/` | Figuras del notebook usadas en el informe. |
| `AI.md` | Detalle del uso de IA generativa en este laboratorio. |
| `requirements.txt` | Dependencias de Python. |
| `S15 -Lab10.pdf` | Enunciado del laboratorio. |

Los pesos de ResNet-18 y el dataset `credit-g` se descargan en la primera ejecución (requiere internet).

## Resultados

- **Grad-CAM vs. CAM**: $\alpha_k^c = w_k^c/Z$ con error relativo máximo de $2.4\times10^{-7}$ (precisión de float32). En la imagen con dos objetos, las regiones top-20 % de cada clase no se tocan (IoU = 0).
- **Atajo de contexto**: en la imagen (c) solo el ≈4 % de la masa del mapa cae dentro del recuadro (que ocupa el 16 %), y al poner el fondo en gris la probabilidad de *bell cote* pasa de 0.27 a ≈0.
- **Modelo tabular**: exactitud 0.775 y AUC 0.765. TreeSHAP suma exacto en log-odds (error $9.5\times10^{-8}$) y su ranking global tiene Spearman 0.70 con la importancia por permutación; `age` es la mayor discrepancia (puesto 6 en SHAP, 20 en permutación).
- **Variable duplicada**: al copiar `credit_amount`, cada gemela recibe ≈0.30 y juntas ≈0.50 (0.49 sin copia). El Shapley exacto coincide con `shap` hasta $2\times10^{-8}$.
- **Costo**: calcular Shapley exacto con 20 variables tomaría ≈12 min por instancia contra ≈0.14 s de KernelSHAP; con 60 variables serían ≈37 000 años.
- **LIME**: $R^2_w \approx 0.3$ con $K = 5$ (0.37 con las 20 variables); el top-5 es idéntico entre corridas con $N \geq 2000$ (Jaccard 1.0). Spearman LIME–SHAP entre 0.68 y 0.90.
- **Contrafactuales**: solo el ≈0.5 % de los 20 000 candidatos es válido; el mejor cambia `checking_status` y `credit_amount` ($P$ de 0.05 a 0.75).
- **Validación de Grad-CAM**: en insertion supera al azar en las tres imágenes ($+6\sigma$ a $+30\sigma$). Al aleatorizar `fc` el Spearman con el mapa original cae por debajo de 0.3 en las tres imágenes.

## AI Usage

Ver [`AI.md`](./AI.md) para el detalle de uso de IA en este laboratorio (alcance, prompts y justificaciones).

## Execution

Requiere Python 3.10+. Instalar dependencias y abrir el notebook:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
python -m ipykernel install --user --name lab10 --display-name "Lab-10 (.venv)"
jupyter lab Lab10_XAI.ipynb   # seleccionar el kernel "Lab-10 (.venv)"
```

Semilla fija: `SEED = 42`. El notebook corre en CPU.

Para compilar el informe:

```bash
latexmk -pdf -outdir=.tex Informe.tex && cp .tex/Informe.pdf .
```
