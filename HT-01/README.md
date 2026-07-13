# Hoja de Trabajo 1: Redes Neuronales

## Summary

This assignment covers the mathematical foundations of neural networks — from why non-linearity is essential, through the probabilistic origin of the loss function, to deriving and implementing gradient descent for logistic regression by hand.

- **Task 1 — Linear Networks & Neuron Mechanics**: Algebraic proof that any stack of purely linear layers collapses into a single linear transformation ($W' = W_2 W_1$, $b' = W_2 b_1 + b_2$), showing why non-linear activations are what give depth its expressive power. Followed by a worked numerical example computing a neuron's preactivation $z$ and sigmoid activation $a = \sigma(z)$, with interpretation of the resulting probability.

- **Task 2 — Binary Cross-Entropy from First Principles**: Full derivation of binary cross-entropy starting from the Bernoulli likelihood, applying the log-likelihood trick, and negating/averaging to reach the standard loss form. Includes a numerical loss comparison at $\hat{y} = 0.1, 0.5, 0.9$ for $y=0$, showing the non-linear penalty growth as predictions diverge from the true label.

- **Task 3 — Backpropagation & Gradient Descent**: Step-by-step chain-rule derivation of $\frac{\partial \mathcal{L}}{\partial w}$ through the logistic regression computation graph ($x \rightarrow z \rightarrow \hat{y} \rightarrow \mathcal{L}$), showing how the sigmoid derivative cancels with the BCE derivative to yield the clean $(\hat{y} - y)x$ gradient. Backed by a from-scratch NumPy implementation trained via gradient descent on a synthetic 2D dataset, with cost curve reported at epochs 0, 250, and 499.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Informe.pdf` | Full report: linear-collapse proof, BCE derivation, chain-rule backprop derivation, and gradient descent results with loss curve. |
| `Informe.tex` | Original LaTeX source for the report. |
| `Semana 1-class.ipynb` | From-scratch NumPy logistic regression: synthetic data generation, gradient descent training loop, and loss curve plotting. |
| `figs/loss.png` | Cost $J$ vs. epoch plot, referenced in the report. |
| `referencias.bib` | Bibliography for cited sources. |

## Notebook Walkthrough (`Semana 1-class.ipynb`)

- **Data generation**: 200 synthetic 2D samples (`X`, shape `(2, 200)`) with labels derived from a noisy linear decision boundary ($0.8x_1 - 1.2x_2 + \text{noise} > 0$).
- **Training loop**: Weights `w` and bias `b` initialized to zero, updated via vanilla gradient descent with learning rate $\alpha = 0.1$ using the closed-form gradient $\frac{\partial \mathcal{L}}{\partial w} = (\hat{y} - y)x$ derived in Task 3.
- **Results**: Cost drops from $J = 0.693147$ (epoch 0, equivalent to $-\log(0.5)$ at random initialization) to $J = 0.275725$ (epoch 250) and $J = 0.248261$ (final epoch), with the steepest descent occurring in the first ~50–100 epochs before flattening — characteristic, well-behaved gradient descent convergence.
- **Plotting**: Loss curve rendered with Matplotlib, marking epochs 0, 250, and 499 with reference lines.

## Execution

Requires Python 3.10+ with NumPy, Matplotlib, and Jupyter (see `.venv`).

```bash
jupyter notebook "Semana 1-class.ipynb"
```
