# Uso de IA generativa: Proyecto 2

Reflexión sobre cómo usé IA (Claude, a través de Claude Code) en este proyecto. Trabajé a un nivel de abstracción alto: en vez de pedir fragmentos de código, delegaba tareas completas y revisaba lo que volvía.

## El flujo

1. Entrego una tarea o una exploración inicial simple, y pido que vuelva con hallazgos.
2. Con esos hallazgos pido experimentos más específicos.
3. Repito. Los experimentos de este proyecto no requieren mucho cómputo, así que podía iterar muchas veces seguidas en lugar de pensar cada experimento de antemano.

Lo que cambia respecto a escribir el código yo mismo es dónde pongo la atención. Casi no escribo código; decido qué pregunta vale la pena, leo los resultados y decido el siguiente paso.

## Por componente

**EDA.** Fue lo más fácil de delegar. Pedí muchas exploraciones a la vez y que la IA regresara con hallazgos interesantes, para generar preguntas mejores que las que yo tenía al empezar. Varios hallazgos terminaron siendo decisiones de diseño:

- Una cola de actividad después del 10 de septiembre concentra 12.7% del lavado, y el simulador la genera aunque el resto de la actividad se detenga. De ahí salió la decisión de no darle tiempo absoluto al modelo.
- El pico de medianoche no es un batch: solo 2.8% de la hora 0 está marcada exactamente a las 00:00. Se descartó el feature de hora del día.
- Wire y Reinvestment no tienen lavado, y ACH concentra 87% de este. De ahí salió el submuestreo de esos remitentes en entrenamiento.
- Cuatro ids de cuenta aparecen con bancos distintos, así que una cuenta se identifica por banco y número.

**Ingeniería de datos.** Aquí usé la IA sobre todo para correr experimentos. El mejor ejemplo es la configuración de ventanas: en lugar de elegir largo mínimo, tamaño de ventana, paso y tope a ojo, pedí una tabla comparando configuraciones. Mostró que el largo mínimo importaba mucho más que el tamaño de ventana (de 81% a 96% de positivos conservados al pasar de 3 a 2 transacciones), y eso cambió la decisión.

**Modelos.** Es donde más se nota el ciclo de hallazgo, experimento y ajuste. En una versión anterior, la Etapa A (un Transformer) rankeaba bien en apariencia, pero al pedir un análisis salió que medía cuánta actividad tenía el remitente y no qué tan raro era: el error crecía con el largo de la ventana. Eso llevó a normalizar el puntaje por largo. En esta etapa final pedí cambiar la Etapa A, comparar MLP, LSTM y Transformer, y repetir la ablación para ambas arquitecturas.

## Qué decidí yo y qué hizo la IA

Mías: usar un autoencoder encoder-decoder para la Etapa A; agregar un MLP como baseline; comparar LSTM y Transformer en las dos etapas; fijar el umbral en el 5% de remitentes alertados porque es en lo que se basa el análisis de negocio; cuestionar el umbral de mejor F1 que se había implementado; recortar gráficas y métricas del notebook; interpretar los resultados y plantear las hipótesis sobre ellos (por qué el MLP gana en la Etapa A y por qué el preentrenamiento no ayudó); revisar el análisis de los casos y decidir cómo presentarlos; y el formato del reporte y de la documentación, que seguí del enunciado.

De la IA: la implementación, las pruebas, la redacción y borradores del reporte y notebook (verificados detenidamente y modificados), y el demo web (incluida la traducción de los dos modelos a JavaScript, que se verificó contra las salidas de PyTorch). 

## Cómo verifiqué lo que hizo

- Los chequeos del pipeline (ningún remitente en dos splits, padding en cero, etiquetas consistentes) son duros: si fallan, el notebook se detiene.
- Antes de entrenar, tests de cordura: el padding no cambia la representación, y el modelo puede sobreajustar un batch pequeño.
- Los números del reporte vienen de mis corridas del notebook, con intervalos bootstrap, no de lo que la IA diga de memoria.
- Cuando una salida del notebook era muy extensa, pedí que las tablas más largas se tradujeran a tablas en markdown para analizarlas poco a poco, y así revisar los resultados con calma en lugar de leerlos de golpe.
- Las referencias y la ley vigente se verificaron con búsquedas web (lo hizo la IA al preparar el reporte). Eso encontró que la ley que menciona el enunciado (Decreto 67-2001) fue derogada por el Decreto 15-2026. Igual hay que confirmarlo contra una fuente oficial, porque la verificación vino de prensa y blogs legales.

## Dónde falló la IA

- Usó una función `elapsed()` en 15 lugares del notebook sin definirla: el notebook falló al correrlo y lo tuve que reportar.
- Escribió en las limitaciones que el fan-in y el bipartite "se ven como pagos ordinarios" desde un solo remitente, y los resultados mostraron que se detectan al 90% y 87%. Se corrigió al ver los datos.
- Dejó cifras de una corrida anterior en el texto del notebook. Hay que revisar que los números del markdown coincidan con la última corrida.

## Lo que aprendí y lo que cambiaría

Funciona bien delegar a un nivel alto cuando los experimentos son baratos y hay criterios que se pueden medir (PR-AUC, recall con presupuesto de alertas), porque el resultado de cada iteración se puede juzgar sin leer todo el código. Funciona peor para entender: en varios momentos tuve que detenerme y preguntar qué era un GRU, cómo se entrena el autoencoder o por qué ganaba el MLP, porque a ese nivel de abstracción es fácil usar algo sin entenderlo.

Lo que haría distinto:

- Pedir desde el inicio que cada hipótesis venga con un experimento corto para probarla, en lugar de dejarla como explicación posible. Hoy hay dos sin verificar.
- Revisar con más cuidado la primera versión de cada resultado antes de construir encima. Varias correcciones vinieron de mirar los datos después y no antes.
- Correr más de una semilla por modelo. Con una sola, varias diferencias entre modelos quedan dentro del ruido y no se puede decir mucho.

El resultado principal, que preentrenar con la Etapa A no mejoró al clasificador entrenado desde cero, viene de los datos, no del flujo de trabajo. Lo que sí cambió fue la velocidad con la que llegué a verlo.
