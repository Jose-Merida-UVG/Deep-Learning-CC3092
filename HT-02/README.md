# Hoja de Trabajo 2: GAN

## Summary

A DCGAN implemented from scratch in PyTorch to generate 64x64 Pokémon sprites, following the architecture/training guidelines from Radford et al. (2015), followed by two experiments probing GAN training dynamics: an artificially induced mode collapse and an application of the Two Time-Scale Update Rule (TTUR) from Heusel et al. (2017).

- **Task 0 — Dataset**: Custom `Dataset`/`DataLoader` over ~900 Pokémon sprites (`data/pokemon`), downloaded via `download_data.py` from the public PokeAPI sprite repo, composited onto a solid background to remove transparency, and resized to 64x64 RGB.
- **Task 1 — DCGAN**: Generator (`z ∈ R^100 -> (batch, 100, 1, 1) -> transposed-conv stack -> 64x64x3 image`) and Discriminator (`64x64x3 image -> conv stack -> scalar`) built per Radford et al.'s architectural rules, with output-shape verification.
  - **Task 1.2 — Alternating training**: `train` implements the standard alternating min-max training loop (configurable `disc_steps`), with a fixed-noise sample grid saved per epoch.
  - **Task 1.3 — Visualizations**: 50-epoch baseline run, with generator/discriminator loss curves (`figs/loss/normal_losses.png`) and per-epoch sample grids (`figs/samples/`).
- **Task 2 — Inducing mode collapse**:
  - **Task 2.1**: Re-trained for 20 epochs with `disc_steps=5`, deliberately over-training the discriminator to trigger mode collapse; written analysis (with derivations) on why an overpowered $D$ zeroes out $\nabla_{\theta_g}\mathcal{L}_G$, the value $D^*(x)$ takes on the collapsed outputs, and a proposed fix (asymmetric learning rates) grounded in the gradient math.
  - **Task 2.2**: Empirical Jensen-Shannon divergence estimate per epoch, derived from $V(D,G) = -\log 4 + 2\cdot\text{JSD}$ and the discriminator loss, compared between the normal and collapsed runs (`figs/jsd/jsd_evolution.png`); written analysis on the estimator's bias (underestimates JSD, since $D$ is never truly optimal) and how the empirical curves compare to the theoretical $\text{JSD}\to 0$ at Nash equilibrium.
- **Task 3 — Paper application (TTUR)**: Summary of Heusel et al. (2017), "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," followed by `train_ttur` — a variant of `train` using two distinct Adam learning rates ($a=10^{-4}$ for $G$, $b=4\times10^{-4}$ for $D$) — run for 200 epochs, with loss curves and a written comparison against the baseline run.

## Deliverables

| File | Description |
| :--- | :---------- |
| `GAN.py` | Jupytext-paired source of the notebook (see **A note on `GAN.py`** below). |
| `GAN.ipynb` | Full notebook: dataset loading, DCGAN generator/discriminator, alternating training, mode-collapse experiment with written analysis, JSD estimation, and the TTUR experiment with written analysis. |
| `download_data.py` | Standalone script (AI-generated) that builds `data/pokemon` from the PokeAPI sprite repository. |
| `data/pokemon` | Downloaded/preprocessed 64x64 RGB Pokémon sprites used as training data. |
| `figs/loss/` | Generator/discriminator loss curves for the normal, mode-collapse, and TTUR runs. |
| `figs/jsd/jsd_evolution.png` | Estimated JSD per epoch, normal vs. mode-collapse training. |
| `samples/` | Fixed-noise sample grids saved per epoch of the baseline (Task 1.3) training run. |

## A note on `GAN.py`

This assignment was written in Neovim rather than a Jupyter frontend, using [Jupytext](https://jupytext.readthedocs.io/) (`hydrogen` format) to pair `GAN.py` with `GAN.ipynb` — cells are delimited by `# %%`/`# %% [markdown]` comments, edited and run as a normal `.py` file from the editor, and synced to the notebook automatically. `GAN.ipynb` was then opened in Jupyter to run the full pipeline end-to-end and generate the figures, and final formatting/presentation touches (markdown rendering, LaTeX, embedded images) were verified in VS Code.

## Execution

Requires Python 3.10+ with PyTorch, torchvision, NumPy, Matplotlib, Pillow, requests, and Jupyter (see `.venv` / `requirements.txt`).

```bash
python download_data.py --out data/pokemon
jupyter notebook GAN.ipynb
```
