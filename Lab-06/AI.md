# Uso de IA Generativa

En este laboratorio se utilizó IA generativa para resolver un error de CUDA en el bloque de visualización, además se utilizó para escribir algunas ecuaciones utilizando LaTeX.

## Error de CUDA en la proyección PCA

La celda de PCA del Bloque 4 venía dada y usaba `.numpy()` directamente sobre los tensores. Como el entrenamiento corrió en GPU, los embeddings estan en CUDA y la conversión falla.

*Prompt:* I'm running this given cell in a notebook and it throws:

`TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`

[Codigo de la celda]

The embeddings come from a model trained on GPU, and the labels tensor comes from the DataLoader. What's the minimal change here?

*Por qué funciona:* Pegar el mensaje de error completo junto con la celda exacta le da todo el contexto necesario, y al pedir explícitamente el cambio mínimo evito que reescriba la celda dada. La respuesta (`.detach().cpu().numpy()` en las dos líneas) es trivial de verificar corriendo la celda de nuevo.

## Ecuaciones LaTeX

Aquí más que todo la utilicé para traducir de "pseudo latex", en los bloques de Markdown escribí por ejemplo:

`Delta theta = -eta * mhat_t / (sqrt(vhat_t) + eps)`

Luego, le pedí que me ayudara a formatearlo.

*Prompt:* I've got the following Markdown block, can you fix the LaTeX "equations" I've written?

*Por qué funciona:* Puedo verificar rápidamente que las ecuaciones si sean de lo que tuve intención, además puedo arreglarlo fácilmente en caso que esté mal. Más que todo es por evitar perder tiempo buscando símbolos específicos o arreglando issues de formateo.
