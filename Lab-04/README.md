# Laboratorio Semana 5: Mecanismo de Atención desde cero

## Summary

Starter notebook (corpus, vocabularies and initial weights given, not to be modified) extending the Week-4 Seq2Seq with scaled dot-product attention: instead of a fixed context vector $\mathbf{c} = \mathbf{h}_T^{enc}$, the decoder computes a **dynamic** context $\tilde{\mathbf{c}}_s$ at every step as a weighted combination of all encoder hidden states. Only PyTorch tensors are allowed — no `nn.MultiheadAttention` or any high-level layer — and the backward pass is fully manual (`loss.backward()` is not used).

- **Block 0 (given)**: Same 78-pair EN/ES corpus as Week 4 (58 train / 20 test), vocabularies (EN 158, ES 163), `lstm_cell`, and fixed initial weights — with the new $W_Q, W_K, W_V \in \mathbb{R}^{16 \times 32}$ attention matrices and a widened $W_{out}$ that now takes the concatenation $[\mathbf{h}_s^{dec}; \tilde{\mathbf{c}}_s]$ of dimension 48.
- **Block 1 — Q, K, V projections (8 pts)**: Filled in the query $\mathbf{q}_0 = W_Q \mathbf{h}_0^{dec}$ and the encoder-wide keys/values $K = H_{enc}W_K^\top$, $V = H_{enc}W_V^\top$, computed once and reused across every decoder step.
- **Block 2 — Attention scores (8 pts)**: Filled in the scaled dot product $e_{0,t} = (K \mathbf{q}_0)_t / \sqrt{d_k}$ as a single matrix-vector product.
- **Block 3 — Attention weights (6 pts)**: Filled in the softmax over the $T$ scores.
- **Block 4 — Dynamic context (6 pts)**: Filled in $\tilde{\mathbf{c}}_0 = V^\top \boldsymbol{\alpha}_0$.
- **Block 5 — Full decoder forward (14 pts)**: Filled in the per-step loop — LSTM cell, query, scores, weights, dynamic context, concatenate-and-project through $W_{out}$, cross-entropy — caching `q`, `scores`, `alpha`, `c_tilde` and `h_dec` for the backward pass.
- **Block 6 — Attention backward (10 pts)**: Filled in the manual backward through both gradient routes out of $\tilde{\mathbf{c}}_s$ — one to the values ($\partial L/\partial V \mathrel{+}= \boldsymbol{\alpha}_s \otimes \partial L/\partial\tilde{\mathbf{c}}_s$) and one through the softmax to the scores and on to $\mathbf{q}_s$/$K$ — accumulating into $W_Q$, $W_K$, $W_V$ and $H_{enc}$.
- **Block 7 — Encoder backward with attention gradient (given structure)**: BPTT through the encoder where $\mathbf{h}_T^{enc}$ now receives gradient from two sources (the decoder's initial context plus `dH_enc[T-1]`) and earlier steps receive `dH_enc[t]` alone.
- **Block 8 — Parameter update (5 pts)**: Filled in the gradient-descent step for every parameter, including the three new attention matrices.
- **Block 9 — Training loop (3 pts)**: Filled in 5 full-corpus iterations with attention, plus the same loop **without** attention (the Week-4 seq2seq) for a like-for-like comparison from identical seeds.
- **Block 10 (given)**: Convergence plot (with vs. without attention) and an attention-map heatmap over a greedy decode.
- **Block 11 (given, answered)**: Written analysis answers — (1) the two backward routes and what each teaches the model, how attention adds a one-hop shortcut into the computation graph alongside the recurrent path, and why $W_Q$ and $W_K$ learn different projections despite the same $d_k$ target space; (2) reading the attention map for `she sings well -> canta bien` (subject elision, agreement, and the near-uniform weights after only 5 iterations), whether the map should sharpen over 100 iterations, and a corpus case (`the cat sleeps -> el gato está durmiendo`) where dot-product attention has no source token to align `está` to; (3) self-attention as preparation for Week 6 — exactly which Block-6 lines change, why $W_Q$'s and $W_K$'s gradients differ structurally (row vs. column reuse in the score matrix), and why 8 heads sharing one $W_Q/W_K/W_V$ would collapse to a single head.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Attention.ipynb` | Full notebook: Q/K/V projections, scaled dot-product attention, decoder forward with dynamic context, manual backward through both gradient routes, 5-iteration training loop with and without attention, autograder cell, and written answers to the three analysis questions. |
| `convergencia_atencion.png` | Loss-vs-iteration curves comparing the with-attention and without-attention runs over the 5 iterations. |
| `mapa_atencion.png` | Attention-weight heatmap (decoder step × encoder token) for a greedy decode of a test sentence. |

## Notebook Walkthrough

- **Autograder pair**: `they play soccer -> juegan futbol` — 3 encoder tokens, 3 decoder steps; every block checks its output against a precomputed SHA-256 hash of the rounded tensor.
- **Attention at initialization**: with weights scaled by 0.1, the raw scores are tiny (order $10^{-5}$), so $\boldsymbol{\alpha} \approx 1/3$ across all three tokens — the attention distribution starts essentially uniform, which is what gives every encoder token comparable gradient before the model learns to discriminate.
- **Forward result (Block 5)**: `loss_mean = 5.0975` on the verification pair, close to $-\log(1/163)$ as expected at initialization.
- **Training (Block 9)**: with attention, loss over 5 full-corpus iterations went `5.0725 -> 5.0282 -> 4.9842 -> 4.9405 -> 4.8970` (3.46% reduction); the no-attention control from the same seed went `5.0730 -> 4.9032` — a marginally slower descent, the gap being small because 5 passes over 58 pairs is far too few for the alignment to actually be learned.
- **Attention map (Block 10)**: weights remain near-uniform after 5 iterations, with only a faint gradient across tokens — the analysis notes this explicitly rather than over-reading the colors.
- **Automatic grading cell (Block 12)**: 60/60 on the code section (blocks 1–6, 8 and 9 all pass their hash/assertion checks); the 25 analysis-question points and 15 code-comment points are graded manually.

## Execution

Requires Python 3.10+ with PyTorch, NumPy, Matplotlib, and Jupyter (see `.venv`).

```bash
jupyter notebook Attention.ipynb
```
