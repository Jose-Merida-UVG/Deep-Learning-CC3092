# Prompts usados para generar el codigo — Multi-Head Attention (Proyecto 1, Semana 6)

Prompts que se pudieron haber usado para que una IA generara cada bloque de codigo trabajado en esta sesion, con el "por que funciono" y donde se uso en el notebook.

---

## 1. Separar cabezas para multi-head attention

**Donde se uso:** Paso 2 de `multi_head_attention`.

```python
Q = Q.view(T, self.n_heads, self.d_k).transpose(0, 1)
K = K.view(T, self.n_heads, self.d_k).transpose(0, 1)
V = V.view(T, self.n_heads, self.d_k).transpose(0, 1)
```

**Prompt:**

> I'm implementing multi-head attention from scratch in PyTorch (without `nn.MultiheadAttention`). I already have `Q`, `K`, `V` with shape `(T, d_model)`, the result of projecting `x` with `self.WQ`, `self.WK`, `self.WV`. I have `self.n_heads` heads
> and `self.d_k = d_model // n_heads`. Write the PyTorch code to split `Q`, `K`, `V` into the `n_heads` heads, leaving each one with final shape `(n_heads, T, d_k)`, so that each head processes a different slice of the vector
> for each token. Don't use `nn.MultiheadAttention` or high-level functions, only basic tensor operations.

**Por que funciono:** Dar las formas de entrada y salida exactas (`(T, d_model)` → `(n_heads, T, d_k)`) le da al modelo un objetivo verificable en vez de pedir "codigo de attention" en general.

---

## 2. Enmascarar posiciones PAD antes del softmax

**Donde se uso:** Paso 4 de `multi_head_attention`.

```python
key_mask = pad_mask.view(1, 1, T)
scores = scores.masked_fill(~key_mask, float('-inf'))
```

**Prompt:**

> I have `scores` with shape `(n_heads, T, T)` (attention scores before softmax) and `pad_mask`, a boolean tensor of shape `(T,)` where `True` marks valid positions and `False` marks padding positions (`<PAD>`). Write the
> PyTorch code so that no token can attend to PAD positions, setting those scores to `-inf` before applying softmax (softmax is applied afterward, don't include it). The mask must apply the same way for all heads and
> all rows (queries), it only depends on the key's position.

**Por que funciono:** Especificar el significado semantico de `True`/`False` en la mascara evita que el modelo invierta la logica por accidente.

---

## 3. Recombinar las cabezas y proyectar con WO

**Donde se uso:** Paso 7 de `multi_head_attention`.

```python
out = out.transpose(0, 1).contiguous().view(T, self.d_model)
out = out @ self.WO
```

**Prompt:**

> After applying attention per head, I have `out = attn_w @ V` with shape `(n_heads, T, d_k)`. Write the PyTorch code to: (1) recombine all the heads back into a single vector per token, leaving the final shape as
> `(T, d_model)` with `d_model = n_heads * d_k`, and (2) apply a final linear projection with a weight matrix `self.WO` of shape `(d_model, d_model)` to mix information across heads.

**Por que funciono:** Nuevamente, detallar los tamaños y dar información sobre el código correspondiente genera mejores respuestas.
