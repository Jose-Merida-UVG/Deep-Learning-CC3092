# Uso de IA Generativa
En este laboratorio se utilizó IA generativa en la implementación de CE Loss, además se utilizó para escribir algunas ecuaciones utilizando LaTeX.

## CE Loss

*Prompt:* I've got the following code where I'm trying to implement CE loss:

[Codigo]

I've calculated softmax on the logits already and simply need to get the probabilities depending on the label. How would I implement this using PyTorch / what functions do I need to work with these
tensors?

*Por qué funciona:* Siempre intento ser específico, dejando poco espacio para interpretación. Además, las consultas fueron cosas pequeñas y es poco probable que se pierda dentro del contexto o genere
código poco coherente entre si el LLM.

## Ecuaciones LaTeX
Aquí más que todo la utilicé para traducir de "pseudo latex", en los bloques de Markdown escribí por ejemplo:

`partial L / partial yhat_i= 2/N (yhat_i - y_i).`

Luego, le pedí que me ayudara a formatearlo.

*Prompt:* I've got the following Markdown block, can you fix the LaTeX "equations" I've written?

*Por qué funciona:* Puedo verificar rápidamente que las ecuaciones si sean de lo que tuve intención, además puedo arreglarlo fácilmente en caso que esté mal. Más que todo es por evitar perder tiempo
buscando símbolos específicos o arreglando issues de formateo.
