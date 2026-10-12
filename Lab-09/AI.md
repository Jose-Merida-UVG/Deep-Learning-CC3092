# Uso de IA Generativa

Se utilizó Claude Code (Anthropic) desde la terminal sobre el repositorio del laboratorio, principalmente como herramienta para el trabajo mecánico: armar el notebook como "scaffolding" y ayudar con el formato en LaTeX y Markdown. La lógica principal, el análisis y las demostraciones son de mi autoría, salvo lo indicado en la tabla. Este mismo documento (`AI.md`) también fue redactado con ayuda de la IA.

## Resumen del alcance

| Parte | Uso de la IA |
| --- | --- |
| Scaffolding | Estructura del notebook con encabezados por sub-tarea, carga de datos, ciclos de entrenamiento/evaluación, figuras y tablas. Las funciones core quedaron como `# TODO (core)`. |
| Funciones core | Implementé la lógica principal (Tasks 1, 2 y 4). **Solo la Task 2.4** (segundo calendario, incluida su descripción) **y la Task 3.1** (U-Net condicional) fueron implementadas por la IA. Las revisé contra el PDF y los outputs del notebook. |
| Redacción | Formato de Markdown y LaTeX de las respuestas analíticas y demostraciones (redactadas previamente por mí). |
| Revisión | Encabezados de Pregunta/Respuesta en el notebook, versión más concisa de las respuestas de la Task 4, corrección de dos inconsistencias en la Task 3.5 y contraste del notebook contra el PDF del laboratorio. |
| Informe | Generación de `Informe.tex` a partir de la plantilla del repositorio, transcribiendo las respuestas del notebook. |

## Scaffolding del notebook

*Prompt:* "I'm working on Lab 9, instructions are in @S14 - Lab9.pdf. I need the notebook scaffolding: data loading, seed, training and evaluation loops, checkpoints, figures, tables and metrics for each task. Leave the core functions as `# TODO (core)` and the answer cells empty so I can fill them in."

*Por qué funcionó:* El enunciado es muy específico y el trabajo delegado era repetitivo, también son secciones bastante fáciles de verificar. Por ejemplo, si el dataset no cargó es bastante obvio que el Notebook no corre o si observo comportamientos muy fuera de la norma algo salió mal en la implementación.

## Funciones core (Task 2.4)

*Prompt:* "Please implement `custom_betas` for Task 2.4. Take a popular schedule from the literature, cite it, and implement it with some modification. It must satisfy ᾱ_T < 1e-3 and β_t ∈ (0,1) for all t."

*Cómo lo verifiqué:* Con los `assert` del notebook (β_t ∈ (0,1) y ᾱ_T < 1e-3), comparando su curva de SNR con la del calendario lineal y revisando la fórmula contra el paper citado (Nichol & Dhariwal, 2021).

## Funciones core (Task 3.1)

*Prompt:* "Please implement the missing Conditional U-Net for Task 3.1. Follow the PDF architecture strictly (channels, bottleneck, concatenated skips). Include the sinusoidal time embedding and class conditioning, and briefly explain how they are injected."

*Cómo lo verifiqué:* Contra las fórmulas del PDF, revisando la explicación que generó, y verificando empíricamente que los tensores pasaran todos los `assert` de dimensiones del notebook.

## Revisión del notebook

*Prompt:* "Can you check it out, just add markdown cells to be more indicative of where the questions and answers are please?"

*Por qué funcionó:* Los cambios eran de estructura y redacción, sin tocar código ni salidas, y cada número de las respuestas se puede verificar directamente contra los outputs del notebook.

## Informe en LaTeX

*Prompt (reconstruido):* "Can you make a .tex doc? I've got a template in the repo, just pull it up and copy 1:1 what the notebook says."

*Por qué funcionó:* El contenido ya estaba escrito y verificado en el notebook, así que la tarea era solo de transcripción y formato sobre una plantilla existente. Es fácil de verificar comparando el PDF compilado con las celdas del notebook.
