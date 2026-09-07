# Laboratorio 6 — Semana 9: Transfer Learning y Fine-Tuning

## Summary

Starter notebook (data pipeline, training/eval loops and plotting given, not to be modified) where ResNet-18 pretrained on ImageNet is adapted to CIFAR-10 under three transfer-learning strategies of increasing scope, then compared experimentally — both by validation accuracy and by the geometry of the representations each one produces.

- **Block 0 (given)**: CIFAR-10 with ImageNet normalization and `Resize(64)` + horizontal-flip augmentation, subsampled to 2000 train / 500 val so the lab runs on CPU; the `train_epoch` / `eval_epoch` / `plot_curves` helpers; seeds fixed at 42.
- **Block 1 — Feature extraction (20 pts)**: Filled in `build_feature_extractor` — load `resnet18(weights='IMAGENET1K_V1')`, freeze every parameter, and swap `fc` for a fresh `nn.Linear(512, 10)`, leaving exactly 5,130 trainable parameters against 11,176,512 frozen ones. Trained 10 epochs with Adam at `lr=1e-3`.
- **Block 2 — Partial fine-tuning (20 pts)**: Filled in `build_partial_finetune` — same setup but with `layer4` unfrozen alongside the new head (~8.4M trainable), trained at the lower `lr=1e-4` so the pretrained weights are not destroyed.
- **Block 3 — Full fine-tuning with differential LR (20 pts)**: Filled in `build_full_finetune` and `make_param_groups`, assigning one learning rate per depth group (stem `1e-6`, `layer1` `1e-5`, `layer2` `5e-5`, `layer3` `1e-4`, `layer4` `5e-4`, head `1e-3`) with every model parameter covered by exactly one group.
- **Block 4 — Representation analysis (15 pts)**: Filled in `extract_embeddings` (backbone rebuilt as `nn.Sequential(*list(model.children())[:-1])`, 512-d output after `avgpool`) and `separability_ratio` (inter-class over intra-class scatter), applied to all three trained models, plus a manual SVD-based PCA projection of the embeddings.
- **Block 5 (given, answered)**: Written analysis answers — (1) why `requires_grad=False` on `layer3` means the gradient is never computed rather than mathematically zero (and what that saves in compute and memory), how Adam's normalized step turns a 1000× LR gap into a 1000× step-size gap, and how the three convergence curves order themselves; (2) LoRA parameter counts for the `fc` layer at `r ∈ {1,2,4,8}` and why initializing both `A=0` and `B=0` makes the origin a fixed point of training; (3) what the separability jump implies about what `layer4` relearned for CIFAR-10, and a concrete experiment (per-layer relative L2 drift vs. a fresh ImageNet checkpoint, plus `conv1` cosine similarity) to test whether the `1e-6` early layers moved at all.

## Deliverables

| File | Description |
| :--- | :---------- |
| `S9_-_Lab6_Semana9.ipynb` | Full notebook: the three transfer-learning strategies, embedding extraction and separability metric, PCA visualization, autograder cell, and written answers to the three analysis questions. |
| `Comparacion_estrategias.png` | Validation loss and accuracy curves for feature extraction / partial fine-tuning / differential LR over 10 epochs. |
| `representaciones_pca.png` | 2D PCA projections of the 512-d validation embeddings for the three models, annotated with each one's separability ratio. |

## Notebook Walkthrough

- **Block 1 results**: 5,130 trainable / 11,176,512 frozen, exactly as required; val accuracy climbs 52.2% → 60.6% over 10 epochs while val loss flattens from epoch 4 on (~1.27), the frozen backbone capping what a linear head can extract.
- **Block 2 results**: ~8.4M trainable parameters, val accuracy 68.6% → 73.8% (best 75.0%), with train accuracy hitting 99.95% by epoch 10 — the small 2000-image subset is memorized well before the validation curve stops improving.
- **Block 3 results**: LRs verified strictly increasing with depth and every parameter covered by exactly one group; best val accuracy 78.8%, reaching 77.2% by epoch 2 — far faster than either of the other two strategies.
- **Block 4 results**: separability ratio 0.0580 (FE) → 0.2120 (partial, 3.7×) → 0.7178 (differential LR, 12.4×). The PCA plots track that ordering: feature extraction is one undifferentiated cloud, partial fine-tuning shows visible clusters, and differential LR cleanly splits the vehicle classes from the animal ones — CIFAR-10's coarsest division, and the first thing the leading components pick up.
- **Loss vs. accuracy divergence**: the differential-LR model bottoms out in val loss at epoch 1 (~0.75) and then rises past the partial model (1.07 vs. 0.80 by epoch 6) while still winning on accuracy — increasingly confident predictions on a memorized training subset.
- **One modification to given code**: two lines in the PCA cell changed from `.numpy()` to `.detach().cpu().numpy()`, since the tensors live on CUDA.
- **Automatic grading cell**: 75/75 on the code section (B1 20, B2 20, B3 20, B4 15); the 25 analysis-question points are graded manually.

## AI Usage

See [`AI.md`](./AI.md) for details on AI usage in this project (prompts and explanations).

## Execution

Requires Python 3.10+ with PyTorch, torchvision, NumPy, Matplotlib, and Jupyter (see `.venv` / `requirements.txt`). CIFAR-10 (~170MB) downloads into `./data` on first run.

```bash
jupyter notebook S9_-_Lab6_Semana9.ipynb
```
