# Lab 10 — Explicabilidad (Grad-CAM / CAM, SHAP / LIME / contrafactuales)

## Entorno

```bash
cd Lab-10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name lab10 --display-name "Lab-10 (.venv)"
jupyter lab Lab10_XAI.ipynb   # seleccionar el kernel "Lab-10 (.venv)"
```

Semilla: `SEED = 42`. Los pesos de ResNet-18 y `credit-g` se descargan en la primera ejecución (requiere internet).

## Contenido

El código, las salidas y las respuestas de todos los tasks están en `Lab10_XAI.ipynb`; el informe es `Informe.tex` / `Informe.pdf` (figuras en `figs/`).
