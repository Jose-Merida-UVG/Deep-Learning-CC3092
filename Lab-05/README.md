# Laboratorio Semana 8: Funciones de Pérdida, Regularización y Optimización

## Summary

Starter notebook (dataset generators and model architectures given, not to be modified) where, unlike previous weeks, the architecture is fixed and the challenge is implementing the loss functions and optimizers correctly with raw PyTorch tensors, then connecting the experimental results to the math derived in class.

- **Block 0 (given)**: Three synthetic datasets — an imbalanced 3-class classification set (70/20/10 split), a linear regression set with 15% heavy-noise outliers, and a small noisy-XOR set (40 train / 200 val) built to overfit visibly — plus the `MLP`, `MLPReg`, and `MLPBig` (128-hidden, 2-layer, optional dropout) architectures.
- **Block 1 — Classification losses (25 pts)**: Filled in manual categorical cross-entropy (numerically-stable log-softmax), manual binary cross-entropy, and the manual CE gradient w.r.t. logits (checked against autograd), plus an experiment comparing standard CE vs. class-weighted CE on the imbalanced dataset.
- **Block 2 — Regression losses (15 pts)**: Filled in manual MSE, MAE, and Huber loss, then trained three models on the outlier-heavy regression set to compare how each loss's optimum shifts under contamination.
- **Block 3 — Optimizers (20 pts)**: Filled in manual `sgd_step` and `momentum_step` (both verified against a hand-computed single update), then compared SGD, SGD+momentum, and `torch.optim.Adam` convergence on the classification set.
- **Block 4 — Regularization (15 pts)**: Filled in manual L2 (added directly to the loss instead of via `weight_decay`) and an `EarlyStopping` class (patience-based, triggered on validation loss), and ran a 4-way comparison (none / L2 / dropout / both) on the deliberately over-parameterized `MLPBig` against the small XOR dataset.
- **Block 5 (given, answered)**: Written analysis answers — (1) why MSE vs. MAE diverge under outliers, worked through both analytically and via per-example gradient magnitude; (2) the momentum velocity amplification factor after 10 consistent-direction steps, and the Adam bias-correction factor at t=1/5/20/100; (3) why L2 inside Adam's adaptive normalization gives a per-parameter-dependent effective decay (motivating AdamW), and a hypothesis for why L2 outperformed Dropout on this specific small-XOR overfitting case.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Losses.ipynb` | Full notebook: manual loss functions (CE, BCE, MSE, MAE, Huber), manual CE gradient vs. autograd check, manual optimizers (SGD, momentum), L2/Dropout/early-stopping regularization experiments, autograder cell, and written answers to the three analysis questions. |
| `Losses.py` | Script export of the notebook. |
| `regresion_perdidas.png` | Convergence curves and fitted predictions for MSE/MAE/Huber on the outlier-contaminated regression set. |
| `optimizadores.png` | Loss-vs-epoch convergence comparison for SGD, SGD+momentum, and Adam. |
| `regularizacion.png` | Train/val loss and accuracy curves for the none/L2/dropout/L2+dropout regularization comparison. |

## Notebook Walkthrough

- **Block 1 verification**: manual CE and BCE match `F.cross_entropy` / `F.binary_cross_entropy_with_logits` to within 1e-5/1e-4, and the manual CE gradient (`softmax - one_hot`, averaged) matches autograd to <1e-5. Class-weighted CE substantially raises the minority class's (10% of the dataset) per-class accuracy over standard CE.
- **Block 2 verification**: converged predictions at `x=0` (true value ≈1.0) were MSE=0.798, MAE=1.096, Huber(δ=1)=1.008 — MSE pulled furthest from the true value by the 15% outlier contamination, consistent with its gradient scaling linearly with error (a 16x larger gradient contribution from an outlier vs. a normal point, vs. 1x for MAE).
- **Block 3 verification**: single-step SGD and momentum updates match hand-computed expected values to <1e-6. In the full comparison, SGD and momentum converge almost identically since the classification task's gradients are consistent/low-noise (momentum's advantage shows up under noisy or oscillating gradients, not here); Adam converges fastest.
- **Block 4 verification**: L2 both lowers the trained weight norm (13.08 → 4.48) and shrinks the train/val loss gap (1.274 → 0.206) relative to no regularization; early stopping triggers well before the 500-epoch cap. Dropout was less effective than L2 on this dataset — the analysis argues XOR's overfitting is weight-magnitude memorization (which L2 directly targets) rather than unit co-adaptation (which is what dropout targets), compounded by too few gradient steps (40 examples) for dropout's implicit-ensemble averaging to pay off.
- **Automatic grading cell**: self-reports the code section's 75 pts (Block 1: 25, Block 2: 15, Block 3: 20, Block 4: 15); the 25 analysis-question points are graded manually.

## AI Usage

See [`AI.md`](./AI.md) for details on AI usage in this project (prompts and explanations).

## Execution

Requires Python 3.10+ with PyTorch, NumPy, Matplotlib, and Jupyter (see `.venv`).

```bash
jupyter notebook Losses.ipynb
```
