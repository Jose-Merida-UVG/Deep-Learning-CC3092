# Uso de IA Generativa

Se utilizó Claude Code (Anthropic) desde la terminal sobre el repositorio del laboratorio. La IA se usó para el trabajo de ingeniería y presentación (estructura del notebook, formato, carga de datos, gráficas y tablas), para la implementación completa del Task 4 y para revisar el texto. Las implementaciones de los algoritmos centrales de los Tasks 1 a 3 y el análisis de resultados son de mi autoría.

## Resumen del alcance

| Parte | Quién | Detalle |
| --- | --- | --- |
| Scaffolding, datos y gráficas | IA | Estructura del notebook por sub-tarea, carga y partición de datos (imágenes, German Credit), semilla, funciones auxiliares, tablas (`DataFrame`) y todas las figuras (superposiciones de mapas, cascadas, beeswarm, curvas, etc.). |
| Algoritmos centrales, Tasks 1 a 3 | Yo | Grad-CAM con hooks, CAM, TreeSHAP y sus verificaciones, Shapley exacto, LIME desde cero (muestreo, pesos, ridge ponderado, $R^2_w$, $N_{ef}$), contrafactuales por muestreo. |
| Task 4 completo (código) | IA | Todo el código del Task 4: deletion/insertion (4.1), aleatorización en cascada (4.2) y el código de apoyo de las preguntas de 4.3. Yo solo escribí las respuestas. |
| Análisis y respuestas | Yo | Redacción de las respuestas de todos los Tasks, interpretación de los números e hipótesis. |
| Revisión | IA | Revisión de números y redacción contra los outputs del notebook, y limpieza de restos de plantilla (ver abajo). |
| Informe | IA | `Informe.tex` a partir de la plantilla del repositorio, transcribiendo las respuestas del notebook. |

## Scaffolding, datos y gráficas

*Prompt:* "I'm working on Lab 10, instructions are in @S15 - Lab10.pdf. I need the notebook scaffolding: data loading, seed, helper functions, tables and figures for each task. Do NOT implement the core algorithms (Grad-CAM, CAM, TreeSHAP checks, exact Shapley, LIME, counterfactuals) and do NOT write any of the analysis; leave the core functions as `# TODO (core)` and the answer cells empty, I'll fill those in myself."

*Alcance:* Fue la parte más grande del uso de IA. En el prompt excluí a propósito lo que el laboratorio evalúa (los algoritmos y el análisis), para delegar solo trabajo de ingeniería y presentación. Incluye armar las tres imágenes de prueba, cargar y codificar German Credit con la misma partición para 2.3 y 2.5, y todo el código de gráficas. Las funciones centrales quedaron vacías para que las escribiera yo.

*Por qué funcionó:* El enunciado es muy específico y el trabajo era repetitivo y fácil de verificar: si un dataset no carga o una figura sale rara, se nota de inmediato.

*Cómo lo verifiqué:* Ejecutando el notebook completo, comparando las figuras con los números impresos (por ejemplo, la fracción de masa del recuadro en 1.4(c) con el mapa dibujado) y revisando que las particiones y semillas fueran las mismas entre tasks.

## Task 4 (4.1, 4.2 y apoyo de 4.3)

*Prompt:* "Please implement Task 4: deletion and insertion curves for the Grad-CAM maps (P=20 steps, gray reference for deletion, Gaussian blur sigma=10 for insertion, 10 random orders), the cascading parameter randomization with PyTorch default init, and the supporting code for the 4.3 questions. Only code: don't answer the 4.3 questions or interpret the results, I'll write the analysis."

*Alcance:* La IA escribió todo el código del Task 4, es decir `deletion_insertion`, el desenfoque gaussiano separable, la comparación contra órdenes aleatorios (media, desviación y $z$), la cascada con `reset_parameters` y las verificaciones de apoyo de 4.3 (referencia desenfocada, azar por bloques de $32\times32$, comprobación de $\alpha_k^c = w'^c_k/Z$ con `fc` aleatorio). Yo escribí el análisis de todas las preguntas de 4.3.

*Por qué funcionó:* Las definiciones del PDF (AUC como promedio de $c_k$, fracción $k/P$, orden por valor del mapa) son precisas y los resultados se pueden contrastar con los de los Tasks anteriores.

*Cómo lo verifiqué:* Revisando el código contra las definiciones del PDF, comprobando que las curvas tengan sentido (deletion cae y insertion sube, ambas entre 0 y 1) y que el $z$ concuerde con la media y desviación impresas. Al revisar las respuestas, volví a calcular las curvas de la imagen (a) y los valores citados (≈ 0 al quitar el 20 % más importante, ≈ 0.94 al insertarlo) coinciden. El comportamiento raro de deletion en (c) lo contrasté con el azar por bloques, no solo con el azar por píxeles.

## Revisión del notebook y del informe

*Prompt:* "Check the numbers and approximations in the answers against the notebook outputs, and remove leftovers from the template ('el código aquí', 'pendiente', 'se puede cambiar'). Keep my answers, you're only allowed to make small modifications that come from typos / short imprecisions." 

*Alcance:* Fueron correcciones menores:
- Números que no coincidían con los outputs: tiempos de KernelSHAP y del cálculo exacto (2.4 y 2.6(d)), el número de puntos saturados en 3.5(a).
- Restos de plantilla en el notebook y el README.
- Formato inconsistente en 2.2(c).

*Por qué funcionó:* Cada número de las respuestas se puede comparar directamente con los outputs del notebook, y las correcciones no cambiaron el código de los algoritmos ni los resultados existentes.

*Cómo lo verifiqué:* Contra los outputs guardados, ejecutando las celdas de verificación nuevas y recompilando el informe.

## Informe en LaTeX

*Prompt:* "Can you make a .tex doc? I've got a template in the repo, just pull it up and copy 1:1 what the notebook says."

*Por qué funcionó:* El contenido ya estaba escrito y verificado en el notebook, así que la tarea era de transcripción y formato sobre una plantilla existente. Es fácil de verificar comparando el PDF compilado con las celdas del notebook.
