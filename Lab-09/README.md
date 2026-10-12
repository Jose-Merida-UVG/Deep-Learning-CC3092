# Laboratorio 9 — VAE y modelos de difusión (Fashion-MNIST)

## Cómo correr
```bash
source .venv/bin/activate        # el entorno ya está creado (torch + CUDA, jupyter, pandas, ...)
jupyter lab Notebook.ipynb       # kernel "python3" del .venv
```
Recrear el entorno: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Estado del notebook
- **Armado:** datos, semilla, ciclos de entrenamiento/evaluación, checkpoints (`ckpt/`, borre o ponga `FORCE_RETRAIN=True` para reentrenar), todas las figuras, tablas y métricas (nitidez, KL Monte Carlo, verificación del proceso forward, SNR, fidelidad/diversidad, matriz de confusión, tiempos).
- **Para completar (`# TODO (core)`):** `reparameterize`, `kl_divergence`, `vae_loss`, `VAE` · `q_sample`, `custom_betas` · `sinusoidal_embedding`, `UNet`, `diffusion_loss`, `guided_eps`, `ancestral_sample` · `pass_at_k`, `pass_at_k_naive`.
- **Para escribir:** las celdas `Respuesta:` y las demostraciones/cálculos a mano (en el informe).
- Task 1 y 2 = entrega parcial; Task 3 y 4 = entrega final.
