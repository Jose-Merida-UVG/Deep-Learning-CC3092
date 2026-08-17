# Proyecto 1 — Semana 6: Transformer desde cero

## Summary

Group project (8% of course grade) implementing a Transformer Encoder and a decoder-only Mini-GPT from raw PyTorch tensors (`nn.Parameter` + `torch.optim.Adam` allowed, but no `nn.MultiheadAttention`/`nn.TransformerEncoder` or other high-level architecture layers), plus an interactive visualization artifact and a 3-minute video. Both models share `layer_norm` and the multi-head attention logic; the Mini-GPT reuses the encoder's building blocks but adds a causal mask and swaps the CLS classifier head for a full vocabulary projection.

- **Block 0 (given)**: Corpus (SST-2 movie reviews, downloaded from GitHub), vocab builders for both the classification task (`vocab_cls`, with a `<CLS>` token) and the language-modeling task (`vocab_gpt`), and fixed model dimensions (`d_model=32`, `d_ff=64`, `n_heads=2`, `d_k=16`).
- **Part 1 — Transformer Encoder (40 pts)**: Implemented `layer_norm` and a full one-block `TransformerEncoder` (`nn.Module`) — embeddings + positional encoding, multi-head self-attention, Add & LayerNorm, position-wise FFN, Add & LayerNorm, and a classifier head on the `[CLS]` token's final representation — trained with Adam for sentiment classification (positive/negative) on SST-2.
- **Part 2 — Mini-GPT (25 pts)**: Implemented a decoder-only `MiniGPT`, reusing the encoder's attention/FFN logic but adding a causal mask (`-inf` on the upper triangle of the attention scores before softmax) and replacing the CLS classifier with a language-modeling head (`Wlm`/`blm`) projecting to the full GPT vocabulary, trained for next-token prediction.
- **Part 3 — Interactive artifact (15 pts)**: Self-contained HTML page (no server, no network calls) reimplementing both models' inference (forward pass, softmax, attention) in JavaScript from the exported JSON weights. Three tabs: *The Writer* (Mini-GPT text generation across multiple temperatures side-by-side, with clickable tokens to inspect/redraw from the sampling distribution), *The Critic* (live sentiment classification with per-head attention maps and word-level occlusion importance), and *The Notes* (technical write-up of the artifact itself).
- **Part 4 — Analysis questions + video (20 pts)**: Written answers (in the notebook) plus a 3-minute video covering: (1) how information flows from a token to `[CLS]` through multi-head attention, interpretation of attention maps on positive/negative examples, and what a second stacked encoder layer could capture that one layer can't; (2) why `exp(-inf)=0` is exact for the causal mask (vs. a large negative number), how gradient signal richness differs between position 0 and position T-1 under causal masking, and the softmax/temperature relationship behind repetitive vs. varied generations; (3) the code changes needed to turn the Mini-GPT's masked self-attention into cross-attention (connecting back to the Week 5 seq2seq decoder), a parameter-count comparison against the original Vaswani et al. Transformer (~1,584x larger), and what linguistic capabilities a 400-sentence corpus fundamentally can't teach a language model.

## Deliverables

| File | Description |
| :--- | :---------- |
| `S6 - Proyecto1_Semana6.ipynb` | Full notebook: `TransformerEncoder` and `MiniGPT` implementations, training loops, verification/autograder cells, weight export, artifact description, and written answers to the three analysis questions. |
| `convergencia_encoder.png` | Training loss / accuracy curves for the `TransformerEncoder` over 25 epochs. |
| `convergencia_gpt.png` | Training loss curve for the `MiniGPT` over 50 epochs. |
| `encoder_weights.json` / `gpt_weights.json` | Trained parameters exported to JSON, embedded in the artifact for in-browser inference. |
| `sst2_train.tsv` / `sst2_dev.tsv` | SST-2 corpus (downloaded automatically by Block 0). |

## Notebook Walkthrough

- **Part 1 verification**: forward pass produces the expected `[2]` logits and `[2, 16, 16]` attention shapes, attention rows sum to 1.0, and initial loss ≈ 1.46 (close to the untrained baseline). Trained for 25 epochs, reaching **66.00% dev accuracy** (train accuracy ~98.7% by epoch 12) — passes the ≥60% autograder threshold.
- **Part 2 verification**: causal mask confirmed correct (row 0 attends only to position 0, row 1 to positions 0–1, etc.; all future positions are exactly 0). Trained for 50 epochs, loss going **4.3562 → 1.0082** (a >75% reduction, well past the ≥15% autograder threshold). Sample generations at temperature 0.8 show short locally-plausible fragments heavily peppered with `<UNK>`, reflecting the tiny corpus.
- **Automatic grading cell**: reports **65/65** on the code section — Part 1 (40/40, dev_acc=66.00%) and Part 2 (25/25, loss 4.3562→1.0082) — with Parts 3 (artifact, 15 pts) and 4 (video, 20 pts) left for manual grading.
- **Artifact**: published at https://claude.ai/public/artifacts/05c8421e-c929-457f-ab62-f0a9907ff4db — **100% AI-generated**, built with Claude from the exported weights and technical details, with all inference logic reimplemented in JavaScript.
- **Video**: https://youtu.be/ogbKwlHrfCU

## AI Usage

See [`AI.md`](./AI.md) for details on AI usage in this project (prompts and explanations).

## Execution

Requires Python 3.10+ with PyTorch, NumPy, Matplotlib, and Jupyter (see `.venv` / `requirements.txt`). The corpus is downloaded automatically from GitHub when Block 0 runs.

```bash
jupyter notebook "S6 - Proyecto1_Semana6.ipynb"
```
