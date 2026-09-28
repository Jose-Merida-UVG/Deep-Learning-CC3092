# Uso de IA Generativa

Esta entrega corresponde a la entrega parcial (Task 1 y Task 2). Ya había implementado un LSTM manual con tensores puros en un laboratorio anterior, y esa implementación fue bastante escrutinada y revisada a fondo en su momento, así que ya sabía cómo se resolvía este tipo de problema. Por eso usé Claude Code sobre todo para no hacer el "trabajo grueso" de escribir el scaffolding, tanto del notebook de esta entrega como de la estructura del resto del laboratorio (Task 3 y Task 4), y también para formatear Markdown y ayudarme a poner algunas ideas en orden.

## Scaffolding del notebook e implementaciones de Task 1 y Task 2

Se utilizó Claude Code (Anthropic) para generar el scaffolding del notebook y completar las implementaciones de Task 1 y Task 2 a partir del enunciado del PDF.

*Prompt utilizado (resumido):* "Crea un notebook con encabezados markdown por sub-tarea siguiendo el PDF del lab, e implementa make_windows, acf_manual y LSTMForecaster (LSTM manual con tensores puros, sin nn.LSTM) según los pasos detallados en el enunciado."

*Por qué funcionó:* El PDF ya especifica fórmulas exactas (ACF, positional encoding) y pasos numerados para el forward del LSTM, dejando poco margen de ambigüedad de diseño; el resultado se validó con las celdas de verificación provistas. Además, ya tenía experiencia previa implementando y depurando un LSTM manual desde cero, por lo que pude revisar el código generado con criterio propio en vez de confiar ciegamente en él.

## Scaffolding del resto del laboratorio (Task 3 y Task 4)

También se usó para generar el scaffolding de las secciones de Task 3 (Transformer) y Task 4 (análisis comparativo), siguiendo la misma estructura de encabezados por sub-tarea y reusando la implementación de un Transformer encoder manual (self-attention, LayerNorm, FFN) hecha en un proyecto anterior del curso.

## Formato de Markdown y organización de ideas

También se usó para dar formato a bloques de Markdown (encabezados, tablas, LaTeX de las fórmulas) y para ayudar a poner en orden algunas ideas antes de escribirlas. El contenido y las conclusiones de las respuestas de análisis (Task 1.2, Task 3.3, Task 4.1 y Task 4.2) son propios; el uso de IA en esas celdas se limitó a redacción y formato, no a generar el análisis en sí.

## Funciones puntuales: tabla comparativa y mapas de atención

Se pidió ayuda para un par de funciones puntuales dentro del código: la celda que arma la tabla comparativa final de MAE/RMSE/ratio entre LSTM, Transformer y naive (Task 4.2), y la celda que extrae y grafica los mapas de atención promedio del Transformer (Task 3.3c). En ambos casos el cálculo de fondo (qué comparar, qué promediar) ya estaba definido por el enunciado; el uso de IA fue para el código de armado/graficado en sí.
