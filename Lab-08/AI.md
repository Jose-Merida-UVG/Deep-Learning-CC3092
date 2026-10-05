# Uso de IA Generativa

Se utilizó IA generativa para realizar las implementaciones de la parte de ingeniería de datos (bajar el dataset, construir el vocabulario, etc.) al igual que copiar el Mini-GPT que se había realizado anteriormente
para tener un "scaffolding" del proyecto. De cierta manera simplemente lo estoy acercando un poco más a como fueron los primeros labs, más que todo llenar código de ciertas partes clave de alguna implementación. Siempre
intento dejar la parte "relevante" al curso para implementarla a manita.

También utilicé la versión web de Claude para consultas de sintaxis de torch o fragmentos de las implementaciones, estas no las trackee tan a detalle pero más que todo fue apoyo bastante pequeño.

## Resumen del alcance

Herramienta: Claude Code (Anthropic), usado desde la terminal sobre el repositorio del laboratorio.

La IA se usó para **implementar la mayoría del código del laboratorio** y para **traer/portar código existente** (el Mini-GPT de la Semana 6). Mi rol fue principalmente de supervisión y no de escribir el código, revisé las salidas del notebook, revisé el código, comprobé que las validaciones y verificaciones pasaran, y pedí cambios más generales mientras avanzaba el trabajo (por ejemplo, la definición de época y el número de épocas). El flujo fue iterativo en su mayoría, sin embargo también fui bastante específico con las partes que decidí implementar a mano (las más relevantes al curso y el contenido, en mi opinión). En concreto, la IA generó:


| Parte | Qué hizo la IA |
|---|---|
| Scaffolding | Estructura del notebook con un encabezado por sub-tarea según el PDF; `requirements.txt`. |
| Task 1.1 | Descarga de `tiny_shakespeare`, vocabulario de 65 caracteres, codificación, partición 90/10 en orden temporal, `get_batch` (objetivo = entrada desplazada un paso). |
| Mini-GPT | Port del `MiniGPT` de la Semana 6 (atención causal multi-cabeza, LayerNorm, FFN, embeddings posicionales). |
| Task 1.2 | Ciclo de entrenamiento y curvas de pérdida. |
| Task 3 | Extracción de embeddings de `bert-base-multilingual-cased` y carga de modelos. |



## Scaffolding / Tasks 1 y 2

*Prompt:* "I'm working on a Deep Learning project, instructions are in @(archivo).pdf. I need to implement basic scaffolding for the first two tasks, meaning the following:

Task 1:

- Dataset download, vocab construction, test / train set split as detailed in the PDF. Also implement validation with assert cells on vocabulary size.

- 'Mini-GPT' code ported from the Proyecto 1 subdirectory (../Proy-01)

- Training loop with the specified hyperparameters for the Mini-GPT over the dataset.

Note: Leave perplexity & sampling functions as placeholders so I can implement manually.

Task 2:

- Scaffolding for questions / answers, as well as a dictionary with the specific configurations / parameters.

Note: Leave the actual implementation of the generate function to be done manually."

*Por qué funcionó*: El enunciado de por si es bastante específico, antes que nada lo primero que hice fue verificar la información presente y si debía complementar con algo adicional para el prompt. Luego, se delimitó el scope más que todo
a tareas fácilmente verificables y que es bastante difícil que salgan mal. Por ejemplo, la descarga del dataset es trivial de verificar, lo mismo para la partición del dataset y un loop de entrenamiento se puede leer en muy poco tiempo. Además, el código del Mini-GPT ya existía así que solo era verificar que matcheara.


## Task 3 (BERT, Word2Vec y PCA)

*Prompt (reconstruido):* "I'm working on a Deep Learning project, instructions are in @(archivo).pdf. I've already implemented the first 2 tasks, I'll need you to implement what's pending in Task 3. ie:

- Pull bert model specified from HuggingFace, extract the contextual embeddings from the sentences specified.

- Pull a word2vec pre-trained model in gensim (that would work well w/ spanish), obtain the static vector of each of the target words.

- Create a visualization for BERT embeddings using 2D PCA as specified in the PDF

- Build the cells with the questions in Task 4

Note: It's crucial that you do NOT answer the questions in Task 4, I'll also implement the vector comparisons / cosine sim part manually.

*Por qué funcionó:* Nuevamente, los tasks realmente son bastante simples / verificables (el modelo o se descarga, o no se descarga. No hay manera que falle sin que me de cuenta). También fui explícito con las partes que necesitaba y le di una tarea bastante delimitada a las partes repetitivas / sencillas (pero tediosas) de implementar.

