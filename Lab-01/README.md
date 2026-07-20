# Laboratorio Semana 2: CNN Completa con Forward y Backward Pass

## Summary

This lab implements a complete convolutional neural network pipeline — `Conv(3x3) + bias -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(9->1) + sigmoid -> BCE Loss` — entirely from scratch in NumPy, covering both the forward pass and the manual backward pass (backpropagation) through every layer. The network is trained for a single gradient descent iteration on one image (digit "0" vs. not) from scikit-learn's `load_digits` dataset.

- **Forward pass**: 2D convolution (stride 1, no padding) over an 8x8 image with a fixed 3x3 diagonal kernel, ReLU activation, 2x2 max pooling with stride 2 (tracking argmax indices for backward), flatten, a dense layer down to a single logit, sigmoid, and binary cross-entropy loss.
- **Backward pass**: Chain-rule gradients computed in reverse order — dense layer gradients (using the closed-form $\hat{y}-y$ simplification from BCE+sigmoid), backward through flatten and max pooling (gradient routed only to the max index per window), backward through ReLU (masking by the pre-activation sign), and backward through the convolution (gradient w.r.t. kernel and bias accumulated over all sliding-window positions).
- **Weight update**: One step of vanilla gradient descent ($\alpha = 0.01$) applied to the kernel, conv bias, dense weights, and dense bias, followed by a fresh forward pass verifying the loss decreases.
- **Analysis questions**: Written discussion connecting the numeric results to underlying concepts — why early predictions are essentially random, what a kernel gradient value means to the optimizer, the "dead ReLU" failure mode, and why max pooling's backward pass differs from average pooling's.

## Deliverables

| File | Description |
| :--- | :---------- |
| `CNN.ipynb` | Full from-scratch NumPy CNN: forward pass, backward pass, gradient descent update, pipeline visualization, and analysis question answers. |
| `b0_datos.png` | Visualization of the input image (digit 0, 8x8) and the fixed diagonal convolution kernel. |
| `b11_pipeline.png` | Visualization of the full forward pipeline: input, pre-ReLU feature map, post-ReLU feature map, and pooled output, annotated with `y_hat` and loss. |
| `requirements.txt` | Python dependencies (NumPy, Matplotlib, scikit-learn, Jupyter). |

## Notebook Walkthrough (`CNN.ipynb`)

- **Data & fixed weights**: First "0" digit from `load_digits`, normalized to `[0,1]`. Kernel is a fixed 3x3 identity/diagonal matrix with `bias_conv = -1.0`; dense weights `W_dense` (1,9) are seeded (`np.random.seed(0)`) small random values, `b_dense = 0`, learning rate `alpha = 0.01`.
- **Forward pass results**: 6x6 pre-ReLU feature map -> 6x6 post-ReLU -> 3x3 max-pooled -> flattened to 9 -> dense logit `z_dense ≈ 0.204` -> `y_hat ≈ 0.5509` -> `loss (BCE) ≈ 0.8004` (label is 0, so the model starts out wrong).
- **Backward pass results**: `dL_dz ≈ 0.5509`, gradients propagated back through the dense layer, unpooled via stored argmax indices, masked through ReLU, and accumulated into `dL_dkernel` (3x3) and `dL_dbias_conv ≈ 0.0661`.
- **Gradient descent step**: Updated weights (`W_new`, `b_new`, `kernel_new`, `bias_conv_new`) reduce the loss from `0.800438` to `0.786452` (1.75% improvement) in a single iteration, confirming correct gradient signs and magnitudes.
- **Analysis answers**: Cover random-init behavior, the meaning of individual kernel gradient entries, the dead ReLU phenomenon, and how max pooling's winner-take-all backward pass contrasts with average pooling's uniform gradient split.

## Execution

Requires Python 3.10+ with NumPy, Matplotlib, scikit-learn, and Jupyter (see `.venv` / `requirements.txt`).

```bash
jupyter notebook CNN.ipynb
```
