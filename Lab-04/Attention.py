# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Laboratorio Semana 5: Mecanismo de Atención desde cero
#
# **Curso:** Deep Learning  
# **Valor:** 4% de la nota del curso  
# **Entrega:** Notebook ejecutado (.ipynb) con todas las celdas con salida visible
#
# ---
#
# ## Contexto
#
# Esta semana extiende el Seq2Seq de la Semana 4 con el mecanismo de atencion de producto punto escalado. En lugar de un vector de contexto fijo $\mathbf{c} = \mathbf{h}_T^{enc}$, el decoder calcula en cada paso un vector de contexto **dinamico** $\tilde{\mathbf{c}}_s$ que es una combinacion ponderada de todos los hidden states del encoder.
#
# Pipeline del modelo con atencion:
# ```
# oracion_EN -> [E_enc] -> [Encoder LSTM] -> H_enc  (T, d_hid)
#                                             |
#                           K = H_enc W_K^T  |  V = H_enc W_V^T
#                                             |
# <SOS>+ES -> [E_dec] -> [Decoder LSTM] -> h_s^dec -> q_s = W_Q h_s^dec
#                                             |
#                           scores = K q_s / sqrt(d_k)
#                           alpha  = softmax(scores)
#                           c_tilde = V^T alpha
#                                             |
#                           [h_s^dec ; c_tilde] -> W_out -> prediccion
# ```
#
# ## Reglas
#
# - Use unicamente PyTorch. No use `nn.MultiheadAttention` ni ninguna capa de alto nivel.
# - El backward debe ser **manual**. No use `loss.backward()`.
# - No modifique los pesos iniciales ni el corpus.
# - La celda final calcula su nota automatica sobre los **60 puntos** de codigo.

# %%
import torch
import torch.nn.functional as F
import hashlib, numpy as np, random, time
import matplotlib.pyplot as plt

def _hash_tensor(t, decimals=5):
    arr = np.round(t.detach().numpy().astype(np.float64), decimals)
    return hashlib.sha256(arr.tobytes()).hexdigest()

_resultados = {}
print(f'PyTorch: {torch.__version__}')

# %% [markdown]
# ---
# ## Bloque 0: Corpus, vocabularios y pesos (dado, no modificar)

# %%
CORPUS_COMPLETO = [
    ("i love you", "te amo"),
    ("i drink water", "bebo agua"),
    ("i read the news", "leo las noticias"),
    ("i write a book", "escribo un libro"),
    ("i hear music", "escucho musica"),
    ("i call my mother", "llamo a mi madre"),
    ("i take a photo", "tomo una foto"),
    ("i finish my work", "termino mi trabajo"),
    ("i open the book", "abro el libro"),
    ("i run every morning", "corro cada manana"),
    ("i learn spanish", "aprendo espanol"),
    ("i like coffee", "me gusta el cafe"),
    ("i work every day", "trabajo todos los dias"),
    ("i buy fresh bread", "compro pan fresco"),
    ("she reads books", "lee libros"),
    ("she sings well", "canta bien"),
    ("she cooks dinner", "cocina la cena"),
    ("she opens the door", "abre la puerta"),
    ("she sleeps early", "duerme temprano"),
    ("she buys flowers", "compra flores"),
    ("she draws pictures", "dibuja imagenes"),
    ("she visits her friend", "visita a su amiga"),
    ("she teaches math", "ensena matematica"),
    ("she paints a picture", "pinta un cuadro"),
    ("she writes a poem", "escribe un poema"),
    ("she enjoys the music", "disfruta la musica"),
    ("she prepares the meal", "prepara la comida"),
    ("he runs fast", "corre rapido"),
    ("he writes a letter", "escribe una carta"),
    ("he closes the window", "cierra la ventana"),
    ("he eats an apple", "come una manzana"),
    ("he drives a car", "conduce un carro"),
    ("he fixes the bike", "arregla la bicicleta"),
    ("he plays the guitar", "toca la guitarra"),
    ("he studies history", "estudia historia"),
    ("he answers the phone", "contesta el telefono"),
    ("he repairs the chair", "repara la silla"),
    ("he teaches the class", "ensena la clase"),
    ("we eat bread", "comemos pan"),
    ("we walk together", "caminamos juntos"),
    ("we leave early", "salimos temprano"),
    ("we cook together", "cocinamos juntos"),
    ("we swim in the sea", "nadamos en el mar"),
    ("we watch the stars", "miramos las estrellas"),
    ("we visit the museum", "visitamos el museo"),
    ("we celebrate together", "celebramos juntos"),
    ("we enjoy the summer", "disfrutamos el verano"),
    ("they play soccer", "juegan futbol"),
    ("they arrive late", "llegan tarde"),
    ("they build a house", "construyen una casa"),
    ("they clean the room", "limpian el cuarto"),
    ("they watch the movie", "ven la pelicula"),
    ("they travel by train", "viajan en tren"),
    ("they eat together", "comen juntos"),
    ("they sing a song", "cantan una cancion"),
    ("they dance all night", "bailan toda la noche"),
    ("they plant the seeds", "plantan las semillas"),
    ("they clean the street", "limpian la calle"),
    ("the cat sleeps", "el gato esta durmiendo"),
    ("the dog barks", "el perro esta ladrando"),
    ("the bird flies", "el pajaro esta volando"),
    ("the sun shines", "el sol esta brillando"),
    ("the fish swims", "el pez esta nadando"),
    ("the baby laughs", "el bebe esta riendo"),
    ("the teacher explains", "el profesor esta explicando"),
    ("the rain falls", "la lluvia esta cayendo"),
    ("the moon rises", "la luna esta subiendo"),
    ("the wind blows", "el viento esta soplando"),
    ("the fire burns", "el fuego esta ardiendo"),
    ("the clock ticks", "el reloj esta sonando"),
    ("we study english", "estudiamos ingles"),
    ("we paint the wall", "pintamos la pared"),
    ("we drink hot tea", "bebemos te caliente"),
    ("we meet every week", "nos reunimos cada semana"),
    ("the cat drinks milk", "el gato bebe leche"),
    ("the child plays", "el nino juega"),
    ("he reads the newspaper", "lee el periodico"),
    ("she closes her eyes", "cierra los ojos"),
]

random.seed(42)
indices = list(range(len(CORPUS_COMPLETO)))
random.shuffle(indices)
n_train = int(len(CORPUS_COMPLETO) * 0.75)
TRAIN_DATA = [CORPUS_COMPLETO[i] for i in indices[:n_train]]
TEST_DATA  = [CORPUS_COMPLETO[i] for i in indices[n_train:]]
print(f'Train: {len(TRAIN_DATA)} pares, Test: {len(TEST_DATA)} pares')
print(f'Par autograder (TRAIN_DATA[0]): {TRAIN_DATA[0]}')


# %%
SOS, EOS, PAD, UNK = '<SOS>', '<EOS>', '<PAD>', '<UNK>'
SPECIAL = [PAD, UNK, SOS, EOS]

def build_vocab(sentences):
    words = set()
    for s in sentences: words.update(s.lower().split())
    vocab = SPECIAL + sorted(words)
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w

src_vocab, src_w2i, src_i2w = build_vocab([p[0] for p in CORPUS_COMPLETO])
tgt_vocab, tgt_w2i, tgt_i2w = build_vocab([p[1] for p in CORPUS_COMPLETO])
src_V = len(src_vocab); tgt_V = len(tgt_vocab)
print(f'Vocabulario EN: {src_V}, ES: {tgt_V}')

def tokenize_src(s): return [src_w2i.get(w.lower(), src_w2i[UNK]) for w in s.split()]
def tokenize_tgt(s): return ([tgt_w2i[SOS]] +
                              [tgt_w2i.get(w.lower(), tgt_w2i[UNK]) for w in s.split()] +
                              [tgt_w2i[EOS]])


# %%
# Dimensiones
d_emb = 16   # dimension de embeddings
d_hid = 32   # dimension del hidden state LSTM
d_k   = 16   # dimension del espacio de queries y keys
d_v   = 16   # dimension del espacio de values
alpha_lr = 0.01  # tasa de aprendizaje

# Pesos seq2seq base
torch.manual_seed(42)
E_enc = torch.randn(d_emb, src_V) * 0.1
E_dec = torch.randn(d_emb, tgt_V) * 0.1
Wf_enc=torch.randn(d_hid,d_hid+d_emb)*0.1; bf_enc=torch.zeros(d_hid)
Wi_enc=torch.randn(d_hid,d_hid+d_emb)*0.1; bi_enc=torch.zeros(d_hid)
Wc_enc=torch.randn(d_hid,d_hid+d_emb)*0.1; bc_enc=torch.zeros(d_hid)
Wo_enc=torch.randn(d_hid,d_hid+d_emb)*0.1; bo_enc=torch.zeros(d_hid)
Wf_dec=torch.randn(d_hid,d_hid+d_emb)*0.1; bf_dec=torch.zeros(d_hid)
Wi_dec=torch.randn(d_hid,d_hid+d_emb)*0.1; bi_dec=torch.zeros(d_hid)
Wc_dec=torch.randn(d_hid,d_hid+d_emb)*0.1; bc_dec=torch.zeros(d_hid)
Wo_dec=torch.randn(d_hid,d_hid+d_emb)*0.1; bo_dec=torch.zeros(d_hid)
# NOTA: W_out ahora recibe [h_dec ; c_tilde] de dimension (d_hid + d_v,)
W_out = torch.randn(tgt_V, d_hid + d_v) * 0.1
b_out = torch.zeros(tgt_V)

# Matrices de atencion (nuevas esta semana)
torch.manual_seed(7)
W_Q = torch.randn(d_k, d_hid) * 0.1   # (d_k, d_hid)
W_K = torch.randn(d_k, d_hid) * 0.1   # (d_k, d_hid)
W_V = torch.randn(d_v, d_hid) * 0.1   # (d_v, d_hid)

print(f'W_Q: {W_Q.shape}, W_K: {W_K.shape}, W_V: {W_V.shape}')
print(f'W_out: {W_out.shape}  <- recibe [h_dec; c_tilde] de dim {d_hid+d_v}')


# %%
def lstm_cell(h, c, x, Wf, bf, Wi, bi, Wc, bc, Wo, bo):
    concat = torch.cat([h, x])
    f = torch.sigmoid(Wf @ concat + bf)
    i = torch.sigmoid(Wi @ concat + bi)
    ct = torch.tanh(Wc @ concat + bc)
    cn = f * c + i * ct
    o = torch.sigmoid(Wo @ concat + bo)
    hn = o * torch.tanh(cn)
    return hn, cn, {'concat': concat, 'h_prev': h, 'c_prev': c,
                    'f': f, 'i': i, 'ct': ct, 'c_t': cn, 'o': o, 'h_t': hn}

# Forward encoder sobre el par autograder
src_test = tokenize_src(TRAIN_DATA[0][0])
h = torch.zeros(d_hid); c = torch.zeros(d_hid); enc_caches = []
for idx in src_test:
    emb = E_enc[:, idx]
    h, c, cache = lstm_cell(h, c, emb, Wf_enc, bf_enc, Wi_enc, bi_enc,
                             Wc_enc, bc_enc, Wo_enc, bo_enc)
    cache['emb_idx'] = idx; enc_caches.append(cache)

ctx_h = h.clone(); ctx_c = c.clone()

# Apilar hidden states del encoder
H_enc = torch.stack([cc['h_t'] for cc in enc_caches])  # (T, d_hid)
T_enc = H_enc.shape[0]
print(f'Par autograder: {TRAIN_DATA[0]}')
print(f'H_enc forma: {H_enc.shape}  (T={T_enc} tokens, d_hid={d_hid})')

# %%
# VERIFICACION H_enc
_H = 'f161442cb5995364216cce9ed8bd200741b3c08c1ca014ddcd1b07755e6be18f'
try:
    assert _hash_tensor(H_enc) == _H, 'H_enc incorrecto. Verifique el encoder.'
    print('H_enc: CORRECTO')
except AssertionError as e:
    print(f'H_enc: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 1: Proyecciones Q, K, V
#
# Implemente las tres proyecciones sobre los hidden states del encoder y el estado inicial del decoder.
#
# **Query del paso inicial** (usamos $\mathbf{h}_0^{dec} = \mathbf{c} = \mathbf{h}_T^{enc}$):
# $$\mathbf{q}_0 = W_Q \, \mathbf{h}_0^{dec} \in \mathbb{R}^{d_k}$$
#
# **Keys de todos los tokens del encoder** (apiladas en matriz):
# $$K = H_{enc} W_K^\top \in \mathbb{R}^{T \times d_k}$$
#
# **Values de todos los tokens del encoder** (apilados en matriz):
# $$V = H_{enc} W_V^\top \in \mathbb{R}^{T \times d_v}$$
#
# Donde:
# - $H_{enc} \in \mathbb{R}^{T \times d_{hid}}$ es la matriz de hidden states del encoder
# - $W_Q \in \mathbb{R}^{d_k \times d_{hid}}$, $W_K \in \mathbb{R}^{d_k \times d_{hid}}$, $W_V \in \mathbb{R}^{d_v \times d_{hid}}$ son las matrices de proyeccion aprendibles
# - Las keys y values se calculan **una sola vez** para toda la secuencia del encoder y se reutilizan en cada paso del decoder

# %%
# Query del paso inicial
q_0   = W_Q @ ctx_h        # (d_k,)

# Keys de todos los tokens del encoder
K_mat = H_enc @ W_K.T      # (T, d_k)

# Values de todos los tokens del encoder
V_mat = H_enc @ W_V.T      # (T, d_v)

print(f'q_0 forma:    {q_0.shape if q_0 is not None else None}')
print(f'K_mat forma:  {K_mat.shape if K_mat is not None else None}')
print(f'V_mat forma:  {V_mat.shape if V_mat is not None else None}')

# %%
# VERIFICACION BLOQUE 1
_H1 = {
    'q_0':   '31ec5d935ed2adf0be415cd7217c7d7a901612ffc5e2d86da18ac72c33862273',
    'K_mat': '5b3da92cab10e65abc309b68fd4e42e0ceec5b5f45679402ee18a10818ab52de',
    'V_mat': 'de5e3910bb598ab9291017b6615b4c21b0a0ad8d03c6746cb66304e8139843ed',
}
try:
    assert q_0 is not None and q_0.shape==(d_k,), f'q_0 debe ser ({d_k},)'
    assert K_mat is not None and K_mat.shape==(T_enc,d_k), f'K_mat debe ser ({T_enc},{d_k})'
    assert V_mat is not None and V_mat.shape==(T_enc,d_v), f'V_mat debe ser ({T_enc},{d_v})'
    assert _hash_tensor(q_0)==_H1['q_0'], 'q_0 incorrecto.'
    assert _hash_tensor(K_mat)==_H1['K_mat'], 'K_mat incorrecto.'
    assert _hash_tensor(V_mat)==_H1['V_mat'], 'V_mat incorrecto.'
    _resultados['b1'] = True; print('BLOQUE 1: CORRECTO')
except AssertionError as e:
    _resultados['b1'] = False; print(f'BLOQUE 1: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 2: Attention scores
#
# Calcule el attention score de producto punto escalado entre el query $\mathbf{q}_0$ y cada key:
#
# $$e_{0,t} = \frac{\mathbf{q}_0^\top \mathbf{k}_t}{\sqrt{d_k}} = \frac{(K_{mat} \, \mathbf{q}_0)_t}{\sqrt{d_k}}$$
#
# Donde:
# - $(K_{mat} \, \mathbf{q}_0)_t$ es el producto punto entre la fila $t$ de $K_{mat}$ y el vector $\mathbf{q}_0$
# - $\sqrt{d_k}$ es el factor de escala que estabiliza el entrenamiento
# - El resultado `scores` debe ser un vector de forma $(T,)$ con un score por token del encoder
#
# **Importante:** usar la multiplicacion matricial `K_mat @ q_0` produce los $T$ productos punto simultaneamente en una sola operacion.

# %%
# Attention scores con multiplicación matricial
scores = (K_mat @ q_0) / (d_k ** 0.5)   # forma (T,)

print(f'scores forma: {scores.shape if scores is not None else None}')
print(f'scores valores: {scores}')

# %%
# VERIFICACION BLOQUE 2
_H2 = 'b23b61ce5d29de76770c89cf0f7065012d20023f226f129c4cef2ced8f60b428'
try:
    assert scores is not None and scores.shape==(T_enc,), f'scores debe ser ({T_enc},)'
    assert _hash_tensor(scores)==_H2, 'scores incorrecto. Verifique la formula de escala.'
    _resultados['b2'] = True; print('BLOQUE 2: CORRECTO')
except AssertionError as e:
    _resultados['b2'] = False; print(f'BLOQUE 2: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 3: Attention weights
#
# Convierta los scores en una distribucion de probabilidad aplicando softmax:
#
# $$\alpha_{0,t} = \frac{\exp(e_{0,t})}{\sum_{t'=1}^{T} \exp(e_{0,t'})}$$
#
# Donde:
# - `alpha` debe ser un vector de forma $(T,)$ con $\alpha_{0,t} \in (0,1)$ para todo $t$
# - $\sum_{t=1}^{T} \alpha_{0,t} = 1$: es una distribucion de probabilidad valida
# - Use `F.softmax(scores, dim=0)` para aplicar softmax sobre los $T$ scores
#
# El vector `alpha` representa cuanta atencion pone el decoder sobre cada token del encoder en este paso de generacion.

# %%
# Attention weights, softmax sobre los scores
alpha = F.softmax(scores, dim=0)   # forma (T,)

print(f'alpha forma: {alpha.shape if alpha is not None else None}')
print(f'alpha valores: {alpha}')
print(f'alpha suma: {alpha.sum().item():.6f} (debe ser 1.0)')

# %%
# VERIFICACION BLOQUE 3
_H3 = 'd873c9019af3384980335d62131f67a1b6f55ccbb2c495e5b03a8c6a26bd0e04'
try:
    assert alpha is not None and alpha.shape==(T_enc,), f'alpha debe ser ({T_enc},)'
    assert abs(alpha.sum().item()-1.0)<1e-5, 'alpha no suma 1.'
    assert _hash_tensor(alpha)==_H3, 'alpha incorrecto.'
    _resultados['b3'] = True; print('BLOQUE 3: CORRECTO')
except AssertionError as e:
    _resultados['b3'] = False; print(f'BLOQUE 3: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 4: Vector de contexto dinamico
#
# Calcule el vector de contexto dinamico como combinacion ponderada de los values:
#
# $$\tilde{\mathbf{c}}_0 = \sum_{t=1}^{T} \alpha_{0,t} \, \mathbf{v}_t = V_{mat}^\top \boldsymbol{\alpha}_0$$
#
# Donde:
# - $V_{mat}^\top \in \mathbb{R}^{d_v \times T}$ multiplicado por $\boldsymbol{\alpha}_0 \in \mathbb{R}^T$ produce $\tilde{\mathbf{c}}_0 \in \mathbb{R}^{d_v}$
# - El resultado es la suma ponderada de los values usando los attention weights como coeficientes
# - Este vector **reemplaza** al vector de contexto fijo $\mathbf{c}$ de la Semana 4 y **cambia en cada paso** del decoder

# %%
# Vector de contexto, suma ponderada de los valores
# en base a los attention weights
c_tilde = V_mat.T @ alpha   # forma (d_v,)

print(f'c_tilde forma: {c_tilde.shape if c_tilde is not None else None}')

# %%
# VERIFICACION BLOQUE 4
_H4 = 'ec840db9cef7a0695ce91e0ab2a54284646f8b50e8c98a2ea59297aa4f0eebb1'
try:
    assert c_tilde is not None and c_tilde.shape==(d_v,), f'c_tilde debe ser ({d_v},)'
    assert _hash_tensor(c_tilde)==_H4, 'c_tilde incorrecto.'
    _resultados['b4'] = True; print('BLOQUE 4: CORRECTO')
except AssertionError as e:
    _resultados['b4'] = False; print(f'BLOQUE 4: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 5: Forward completo del decoder con atencion
#
# Implemente el forward pass completo del decoder con atencion. En cada paso $s$ del decoder:
#
# 1. Correr la celda LSTM del decoder para obtener $\mathbf{h}_s^{dec}$
# 2. Calcular el query: $\mathbf{q}_s = W_Q \mathbf{h}_s^{dec}$
# 3. Calcular scores: $\mathbf{e}_s = K_{mat} \, \mathbf{q}_s / \sqrt{d_k}$
# 4. Calcular pesos: $\boldsymbol{\alpha}_s = \text{softmax}(\mathbf{e}_s)$
# 5. Calcular contexto dinamico: $\tilde{\mathbf{c}}_s = V_{mat}^\top \boldsymbol{\alpha}_s$
# 6. Concatenar y proyectar: $\mathbf{z}_s = W_{out} [\mathbf{h}_s^{dec}; \tilde{\mathbf{c}}_s] + \mathbf{b}_{out}$
# 7. Calcular perdida: $L_s = -\log(\text{softmax}(\mathbf{z}_s)[k_s^*])$
#
# **Cambio clave respecto a la Semana 4:** $W_{out}$ ahora recibe la concatenacion $[\mathbf{h}_s^{dec}; \tilde{\mathbf{c}}_s]$ de dimension $(d_{hid} + d_v,)$ en lugar de solo $\mathbf{h}_s^{dec}$.
#
# Guarde en `attn_caches` para cada paso: `q`, `scores`, `alpha`, `c_tilde`, `h_dec`.

# %%
tgt_full = tokenize_tgt(TRAIN_DATA[0][1])
dec_input_idx  = tgt_full[:-1]
dec_target_idx = tgt_full[1:]
S = len(dec_target_idx)

h2 = ctx_h.clone(); c2 = ctx_c.clone()
dec_caches = []; logits_list = []; attn_caches = []
loss_total = torch.tensor(0.0)

for s, (in_idx, ti) in enumerate(zip(dec_input_idx, dec_target_idx)):
    emb = E_dec[:, in_idx]
    h2, c2, dc = lstm_cell(h2, c2, emb, Wf_dec, bf_dec, Wi_dec, bi_dec,
                            Wc_dec, bc_dec, Wo_dec, bo_dec)
    dc['emb_idx'] = in_idx; dc['tgt_idx'] = ti

    # Calcular query
    q_s     = W_Q @ h2   # (d_k,)

    # Calcular scores
    scores_s = (K_mat @ q_s) / (d_k ** 0.5) # (T,)

    # Calcular pesos
    alpha_s  = F.softmax(scores_s, dim=0)  # (T,)

    # Calcular contexto dinámico
    c_tilde_s = V_mat.T @ alpha_s # (d_v,)

    # Concatenar y proyectar
    z_s      = W_out @ torch.cat([h2, c_tilde_s]) + b_out  # (tgt_V,)

    # Actualizar cache
    attn_caches.append({'q': q_s, 'scores': scores_s, 'alpha': alpha_s,
                        'c_tilde': c_tilde_s, 'h_dec': h2})
    
    dec_caches.append(dc); logits_list.append(z_s)

    # Calcular pérdida
    loss_total = loss_total - F.log_softmax(z_s, dim=0)[ti]

loss_mean = loss_total / S
print(f'S (pasos decoder): {S}')
print(f'loss_mean: {loss_mean.item():.4f}')

# %%
# VERIFICACION BLOQUE 5
try:
    assert abs(loss_mean.item()-5.097456)<1e-3, \
        f'loss_mean incorrecto: {loss_mean.item():.6f}, esperado ~5.097'
    _resultados['b5'] = True; print('BLOQUE 5: CORRECTO')
except AssertionError as e:
    _resultados['b5'] = False; print(f'BLOQUE 5: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 6: Backward del mecanismo de atencion
#
# Implemente el backward pass. El gradiente fluye en **dos rutas** desde $\tilde{\mathbf{c}}_s$:
#
# **Ruta 1: hacia los values** (gradiente directo proporcional al peso de atencion)
# $$\frac{\partial L}{\partial \boldsymbol{\alpha}_s} = V_{mat} \frac{\partial L}{\partial \tilde{\mathbf{c}}_s}$$
# $$\frac{\partial L}{\partial V_{mat}} \mathrel{+}= \boldsymbol{\alpha}_s \otimes \frac{\partial L}{\partial \tilde{\mathbf{c}}_s} \quad \text{(producto exterior)}$$
#
# **Ruta 2: hacia los scores a traves del softmax**
# $$\frac{\partial L}{\partial \mathbf{e}_s} = \frac{1}{\sqrt{d_k}} \boldsymbol{\alpha}_s \odot \left(\frac{\partial L}{\partial \boldsymbol{\alpha}_s} - \left(\frac{\partial L}{\partial \boldsymbol{\alpha}_s}^\top \boldsymbol{\alpha}_s\right) \mathbf{1}\right)$$
#
# **Desde los scores hacia queries y keys:**
# $$\frac{\partial L}{\partial \mathbf{q}_s} = K_{mat}^\top \frac{\partial L}{\partial \mathbf{e}_s}$$
# $$\frac{\partial L}{\partial K_{mat}} \mathrel{+}= \frac{\partial L}{\partial \mathbf{e}_s} \otimes \mathbf{q}_s$$
#
# **Acumular en $W_Q$, $W_K$, $W_V$ y en $H_{enc}$:**
# $$\frac{\partial L}{\partial W_Q} \mathrel{+}= \mathbf{q}_s^{grad} \otimes \mathbf{h}_s^{dec}$$
# $$\frac{\partial L}{\partial W_K} \mathrel{+}= (\frac{\partial L}{\partial K_{mat}})^\top H_{enc}$$
# $$\frac{\partial L}{\partial W_V} \mathrel{+}= (\frac{\partial L}{\partial V_{mat}})^\top H_{enc}$$
# $$\frac{\partial L}{\partial H_{enc}} \mathrel{+}= \frac{\partial L}{\partial K_{mat}} W_K + \frac{\partial L}{\partial V_{mat}} W_V$$

# %%
dh_dn = torch.zeros(d_hid); dc_dn = torch.zeros(d_hid)
dWf_dec=torch.zeros_like(Wf_dec); dbf_dec=torch.zeros_like(bf_dec)
dWi_dec=torch.zeros_like(Wi_dec); dbi_dec=torch.zeros_like(bi_dec)
dWc_dec=torch.zeros_like(Wc_dec); dbc_dec=torch.zeros_like(bc_dec)
dWo_dec=torch.zeros_like(Wo_dec); dbo_dec=torch.zeros_like(bo_dec)
dW_out = torch.zeros_like(W_out); db_out_g = torch.zeros_like(b_out)
dE_dec = torch.zeros_like(E_dec)
dW_Q = torch.zeros_like(W_Q)
dW_K = torch.zeros_like(W_K)
dW_V = torch.zeros_like(W_V)
dH_enc = torch.zeros_like(H_enc)   # (T, d_hid)

for s in reversed(range(S)):
    cc = dec_caches[s]; ac = attn_caches[s]
    z  = logits_list[s]; ti = cc['tgt_idx']

    # Gradiente softmax+CE
    p = F.softmax(z, dim=0); dz = p.clone(); dz[ti] -= 1.0; dz = dz / S

    # Gradiente hacia W_out y h_dec via [h_dec; c_tilde]
    h_cat = torch.cat([ac['h_dec'], ac['c_tilde']])
    dW_out += torch.outer(dz, h_cat); db_out_g += dz
    dh_from_Wout = W_out.T @ dz
    dh_s = dh_from_Wout[:d_hid] + dh_dn  # de W_out + paso siguiente
    dc_tilde = dh_from_Wout[d_hid:]       # gradiente hacia c_tilde

    # Ruta 1: gradiente hacia alpha y V_mat
    dal    = V_mat @ dc_tilde  # (T,)    dL/d alpha
    dV_s   = torch.outer(ac['alpha'], dc_tilde)   # (T, d_v) dL/dV_mat en este paso

    # Ruta 2: backward a traves del softmax hacia scores
    dsc    = ac['alpha'] * (dal - (dal @ ac['alpha']))   # (T,)    gradiente del softmax
    dscr   = dsc / (d_k ** 0.5)   # (T,)    escalado por 1/sqrt(d_k)

    # Gradiente hacia q_s y K_mat
    dq_s   = K_mat.T @ dscr   # (d_k,)
    dK_s   = torch.outer(dscr, ac['q'])   # (T, d_k)

    # Acumular en W_Q, W_K, W_V y H_enc
    dW_Q  += torch.outer(dq_s, ac['h_dec'])
    dW_K  += dK_s.T @ H_enc
    dW_V  += dV_s.T @ H_enc
    dH_enc += dK_s @ W_K
    dH_enc += dV_s @ W_V

    # Gradiente de atencion hacia h_dec
    dh_s = dh_s + W_Q.T @ dq_s

    # BPTT decoder LSTM (identico a Semana 4)
    f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
    c_p=cc['c_prev']; conc=cc['concat']
    do=dh_s*torch.tanh(c_t); dcc=dc_dn+dh_s*o*(1-torch.tanh(c_t)**2)
    df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
    zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
    dWf_dec+=torch.outer(zf,conc); dbf_dec+=zf
    dWi_dec+=torch.outer(zi,conc); dbi_dec+=zi
    dWc_dec+=torch.outer(zc,conc); dbc_dec+=zc
    dWo_dec+=torch.outer(zo,conc); dbo_dec+=zo
    dc2=Wf_dec.T@zf+Wi_dec.T@zi+Wc_dec.T@zc+Wo_dec.T@zo
    dh_dn=dc2[:d_hid]; dc_dn=dcc*f; dE_dec[:,cc['emb_idx']]+=dc2[d_hid:]

print(f'dW_Q forma: {dW_Q.shape}')
print(f'dH_enc forma: {dH_enc.shape}')

# %%
# VERIFICACION BLOQUE 6
_H6 = {
    'dW_Q': '98cae91ab905ed0cd4f3303ee7c688449a53055274c3db863f64a6a4b0455da1',
    'dW_K': '6fc36c560c0f7b9ae303f01d2d93ea832020cd450450ffefa0813e9b277293ae',
    'dW_V': '051551903856b47f13742154b5e33793022d92147f93b091e1a6f0df550b74bb',
}
try:
    assert _hash_tensor(dW_Q)==_H6['dW_Q'], 'dW_Q incorrecto.'
    assert _hash_tensor(dW_K)==_H6['dW_K'], 'dW_K incorrecto.'
    assert _hash_tensor(dW_V)==_H6['dW_V'], 'dW_V incorrecto.'
    _resultados['b6'] = True; print('BLOQUE 6: CORRECTO')
except AssertionError as e:
    _resultados['b6'] = False; print(f'BLOQUE 6: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 7: Backward del encoder con gradiente de atencion
#
# El gradiente hacia el encoder ahora tiene **dos fuentes**:
#
# 1. El gradiente del contexto inicial $\mathbf{c} = \mathbf{h}_T^{enc}$ (igual que en Semana 4)
# 2. El gradiente acumulado en `dH_enc` desde el mecanismo de atencion: cada hidden state del encoder recibe gradiente directamente desde los pasos donde fue atendido
#
# Para el ultimo paso del encoder ($t = T-1$), el gradiente total es la suma de ambas fuentes:
# $$\frac{\partial L}{\partial \mathbf{h}_T^{enc}} = \frac{\partial L}{\partial \mathbf{h}_0^{dec}} + dH_{enc}[T-1]$$
#
# Para los demas pasos ($t < T-1$): solo el gradiente de atencion $dH_{enc}[t]$.

# %%
dh_ctx = dh_dn.clone(); dc_ctx = dc_dn.clone()
dh_en = dh_ctx.clone(); dc_en = dc_ctx.clone()
dWf_enc=torch.zeros_like(Wf_enc); dbf_enc=torch.zeros_like(bf_enc)
dWi_enc=torch.zeros_like(Wi_enc); dbi_enc=torch.zeros_like(bi_enc)
dWc_enc=torch.zeros_like(Wc_enc); dbc_enc=torch.zeros_like(bc_enc)
dWo_enc=torch.zeros_like(Wo_enc); dbo_enc=torch.zeros_like(bo_enc)
dE_enc = torch.zeros_like(E_enc)

for t in reversed(range(T_enc)):
    cc = enc_caches[t]
    
    # Combinar gradiente del contexto inicial (solo en t==T_enc-1) con dH_enc[t]
    dh_tot = dH_enc[t] + (dh_ctx if t == T_enc - 1 else dh_en)

    # BPTT encoder LSTM (identico a Semana 4 pero con dh_tot en lugar de dh_en)
    f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
    c_p=cc['c_prev']; conc=cc['concat']
    do=dh_tot*torch.tanh(c_t); dcc=dc_en+dh_tot*o*(1-torch.tanh(c_t)**2)
    df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
    zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
    dWf_enc+=torch.outer(zf,conc); dbf_enc+=zf
    dWi_enc+=torch.outer(zi,conc); dbi_enc+=zi
    dWc_enc+=torch.outer(zc,conc); dbc_enc+=zc
    dWo_enc+=torch.outer(zo,conc); dbo_enc+=zo
    dc3=Wf_enc.T@zf+Wi_enc.T@zi+Wc_enc.T@zc+Wo_enc.T@zo
    dh_en=dc3[:d_hid]; dc_en=dcc*f; dE_enc[:,cc['emb_idx']]+=dc3[d_hid:]

print('Backward encoder completado.')

# %% [markdown]
# ---
# ## Bloque 8: Actualizacion de parametros
#
# Actualice todos los parametros del modelo incluyendo las nuevas matrices de atencion:
# $$\theta \leftarrow \theta - \alpha \cdot \frac{\partial L}{\partial \theta}$$
#
# Actualice: `E_enc`, `E_dec`, pesos del encoder, pesos del decoder, `W_out`, `b_out`, **y ademas** `W_Q`, `W_K`, `W_V`.

# %%
# Atencion
W_Q_new = W_Q - alpha_lr * dW_Q
W_K_new = W_K - alpha_lr * dW_K
W_V_new = W_V - alpha_lr * dW_V

# Proyeccion de salida
W_out_new = W_out - alpha_lr * dW_out
b_out_new = b_out - alpha_lr * db_out_g

# Embeddings
E_enc_new = E_enc - alpha_lr * dE_enc
E_dec_new = E_dec - alpha_lr * dE_dec

# Encoder
Wf_enc_new = Wf_enc - alpha_lr * dWf_enc; bf_enc_new = bf_enc - alpha_lr * dbf_enc
Wi_enc_new = Wi_enc - alpha_lr * dWi_enc; bi_enc_new = bi_enc - alpha_lr * dbi_enc
Wc_enc_new = Wc_enc - alpha_lr * dWc_enc; bc_enc_new = bc_enc - alpha_lr * dbc_enc
Wo_enc_new = Wo_enc - alpha_lr * dWo_enc; bo_enc_new = bo_enc - alpha_lr * dbo_enc

# Decoder
Wf_dec_new = Wf_dec - alpha_lr * dWf_dec; bf_dec_new = bf_dec - alpha_lr * dbf_dec
Wi_dec_new = Wi_dec - alpha_lr * dWi_dec; bi_dec_new = bi_dec - alpha_lr * dbi_dec
Wc_dec_new = Wc_dec - alpha_lr * dWc_dec; bc_dec_new = bc_dec - alpha_lr * dbc_dec
Wo_dec_new = Wo_dec - alpha_lr * dWo_dec; bo_dec_new = bo_dec - alpha_lr * dbo_dec

print(f'W_Q_new: {W_Q_new.shape if W_Q_new is not None else None}')

# %%
# VERIFICACION BLOQUE 8
_H8 = {
    'W_Q_new':   '2424ce6e19f2fbaed1ebfce339441268abc641b87985f3d2f197374d3aa271e5',
    'W_K_new':   '5f6fd6c4d9dd01ef6cad1e662dd907795bfea1aaac73fa984d7380c904d03907',
    'W_out_new': 'fd46068b88ad6ffea4bde689590b4af67daa83abdebc0b3468d47b58f7a1ae6a',
}
try:
    for name, arr in [('W_Q_new',W_Q_new),('W_K_new',W_K_new),('W_out_new',W_out_new)]:
        assert arr is not None, f'{name} no definido.'
        assert _hash_tensor(arr)==_H8[name], f'{name} incorrecto.'
    _resultados['b8'] = True; print('BLOQUE 8: CORRECTO')
except AssertionError as e:
    _resultados['b8'] = False; print(f'BLOQUE 8: INCORRECTO\n  {e}')

# %% [markdown]
# ---
# ## Bloque 9: Loop de entrenamiento y convergencia
#
# Implemente 5 iteraciones de entrenamiento sobre el corpus completo.
# Guarde la loss promedio en `losses_train`.
#
# Adicionalmente, ejecute el mismo loop **sin atencion** (seq2seq de la Semana 4) y guarde las losses en `losses_no_attn`. Usara ambas curvas para la comparacion visual y para las preguntas de analisis.

# %%
# Reinicializar pesos
torch.manual_seed(42)
E_enc_tr=torch.randn(d_emb,src_V)*0.1; E_dec_tr=torch.randn(d_emb,tgt_V)*0.1
Wfe=torch.randn(d_hid,d_hid+d_emb)*0.1; bfe=torch.zeros(d_hid)
Wie=torch.randn(d_hid,d_hid+d_emb)*0.1; bie=torch.zeros(d_hid)
Wce=torch.randn(d_hid,d_hid+d_emb)*0.1; bce=torch.zeros(d_hid)
Woe=torch.randn(d_hid,d_hid+d_emb)*0.1; boe=torch.zeros(d_hid)
Wfd=torch.randn(d_hid,d_hid+d_emb)*0.1; bfd=torch.zeros(d_hid)
Wid=torch.randn(d_hid,d_hid+d_emb)*0.1; bid=torch.zeros(d_hid)
Wcd=torch.randn(d_hid,d_hid+d_emb)*0.1; bcd=torch.zeros(d_hid)
Wod=torch.randn(d_hid,d_hid+d_emb)*0.1; bod=torch.zeros(d_hid)
Wo2=torch.randn(tgt_V,d_hid+d_v)*0.1; bo2=torch.zeros(tgt_V)
torch.manual_seed(7)
WQ=torch.randn(d_k,d_hid)*0.1; WK=torch.randn(d_k,d_hid)*0.1; WV=torch.randn(d_v,d_hid)*0.1

losses_train = []
for it in range(5):
    total_loss = 0.0

    # Recorrer todos los pares del conjunto de entrenamiento
    for src, tgt in TRAIN_DATA:

        # FORWARD ENCODER

        # Tokenizar la oracion fuente
        src_t = tokenize_src(src)

        # Inicializar estados en cero y cache vacio
        h = torch.zeros(d_hid); c = torch.zeros(d_hid); ecs = []

        # Correr la LSTM del encoder token por token
        for idx in src_t:
            emb = E_enc_tr[:, idx]
            h, c, cache = lstm_cell(h, c, emb, Wfe, bfe, Wie, bie,
                                     Wce, bce, Woe, boe)
            cache['emb_idx'] = idx; ecs.append(cache)

        # Contexto inicial del decoder: ultimo estado del encoder
        ctxh = h.clone(); ctxc = c.clone()

        # Apilar hidden states y proyectar keys y values
        H = torch.stack([cc['h_t'] for cc in ecs])   # (T, d_hid)
        T = H.shape[0]
        K2 = H @ WK.T    # (T, d_k)
        V2 = H @ WV.T    # (T, d_v)

        # FORWARD DECODER

        # Tokenizar el objetivo y desplazar para teacher forcing
        tgt_f = tokenize_tgt(tgt)
        din = tgt_f[:-1]; dtg = tgt_f[1:]
        S = len(dtg)

        # Arrancar el decoder desde el contexto del encoder
        h2 = ctxh.clone(); c2 = ctxc.clone()
        dcs = []; zs = []; acs = []
        L = torch.tensor(0.0)

        for in_idx, ti in zip(din, dtg):
            # Celda LSTM del decoder
            emb = E_dec_tr[:, in_idx]
            h2, c2, dc = lstm_cell(h2, c2, emb, Wfd, bfd, Wid, bid,
                                    Wcd, bcd, Wod, bod)
            dc['emb_idx'] = in_idx; dc['tgt_idx'] = ti

            # Calcular query
            q = WQ @ h2                            # (d_k,)

            # Calcular scores
            sc = (K2 @ q) / (d_k ** 0.5)           # (T,)

            # Calcular pesos
            al = F.softmax(sc, dim=0)              # (T,)

            # Calcular contexto dinamico
            ct_ = V2.T @ al                        # (d_v,)

            # Concatenar y proyectar
            z = Wo2 @ torch.cat([h2, ct_]) + bo2   # (tgt_V,)

            # Actualizar caches
            acs.append({'q': q, 'scores': sc, 'alpha': al,
                        'c_tilde': ct_, 'h_dec': h2})
            dcs.append(dc); zs.append(z)

            # Acumular perdida
            L = L - F.log_softmax(z, dim=0)[ti]

        # Perdida promedio de este par
        total_loss += (L / S).item()

        # BACKWARD DECODER

        # Gradientes que vienen del paso siguiente (cero al final)
        dh_dn = torch.zeros(d_hid); dc_dn = torch.zeros(d_hid)

        # Acumuladores de gradiente, en cero para cada par
        gWfd=torch.zeros_like(Wfd); gbfd=torch.zeros_like(bfd)
        gWid=torch.zeros_like(Wid); gbid=torch.zeros_like(bid)
        gWcd=torch.zeros_like(Wcd); gbcd=torch.zeros_like(bcd)
        gWod=torch.zeros_like(Wod); gbod=torch.zeros_like(bod)
        gWo2=torch.zeros_like(Wo2); gbo2=torch.zeros_like(bo2)
        gEdec=torch.zeros_like(E_dec_tr)
        gWQ=torch.zeros_like(WQ); gWK=torch.zeros_like(WK); gWV=torch.zeros_like(WV)
        gH=torch.zeros_like(H)   # (T, d_hid)

        # Recorrer los pasos del decoder en orden inverso
        for s in reversed(range(S)):
            cc = dcs[s]; ac = acs[s]
            z = zs[s]; ti = cc['tgt_idx']

            # Gradiente softmax+CE
            p = F.softmax(z, dim=0); dz = p.clone(); dz[ti] -= 1.0; dz = dz / S

            # Gradiente hacia W_out y h_dec via [h_dec; c_tilde]
            h_cat = torch.cat([ac['h_dec'], ac['c_tilde']])
            gWo2 += torch.outer(dz, h_cat); gbo2 += dz
            dh_from_Wout = Wo2.T @ dz
            dh_s = dh_from_Wout[:d_hid] + dh_dn  # de W_out + paso siguiente
            dc_tilde = dh_from_Wout[d_hid:]      # gradiente hacia c_tilde

            # Ruta 1: gradiente hacia alpha y V_mat
            dal  = V2 @ dc_tilde                          # (T,)
            dV_s = torch.outer(ac['alpha'], dc_tilde)     # (T, d_v)

            # Ruta 2: backward a traves del softmax hacia scores
            dsc  = ac['alpha'] * (dal - (dal @ ac['alpha']))   # (T,)
            dscr = dsc / (d_k ** 0.5)                          # (T,)

            # Gradiente hacia q_s y K_mat
            dq_s = K2.T @ dscr                       # (d_k,)
            dK_s = torch.outer(dscr, ac['q'])        # (T, d_k)

            # Acumular en W_Q, W_K, W_V y H_enc
            gWQ += torch.outer(dq_s, ac['h_dec'])
            gWK += dK_s.T @ H
            gWV += dV_s.T @ H
            gH  += dK_s @ WK
            gH  += dV_s @ WV

            # Gradiente de atencion hacia h_dec
            dh_s = dh_s + WQ.T @ dq_s

            # BPTT decoder LSTM
            f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
            c_p=cc['c_prev']; conc=cc['concat']
            do=dh_s*torch.tanh(c_t); dcc=dc_dn+dh_s*o*(1-torch.tanh(c_t)**2)
            df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
            zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
            gWfd+=torch.outer(zf,conc); gbfd+=zf
            gWid+=torch.outer(zi,conc); gbid+=zi
            gWcd+=torch.outer(zc,conc); gbcd+=zc
            gWod+=torch.outer(zo,conc); gbod+=zo
            dc2=Wfd.T@zf+Wid.T@zi+Wcd.T@zc+Wod.T@zo
            dh_dn=dc2[:d_hid]; dc_dn=dcc*f; gEdec[:,cc['emb_idx']]+=dc2[d_hid:]

        # BACKWARD ENCODER

        # El gradiente que sobra del decoder entra por el contexto inicial
        dh_ctx = dh_dn.clone(); dc_ctx = dc_dn.clone()
        dh_en = dh_ctx.clone(); dc_en = dc_ctx.clone()

        # Acumuladores del encoder, en cero para cada par
        gWfe=torch.zeros_like(Wfe); gbfe=torch.zeros_like(bfe)
        gWie=torch.zeros_like(Wie); gbie=torch.zeros_like(bie)
        gWce=torch.zeros_like(Wce); gbce=torch.zeros_like(bce)
        gWoe=torch.zeros_like(Woe); gboe=torch.zeros_like(boe)
        gEenc=torch.zeros_like(E_enc_tr)

        # Recorrer los pasos del encoder en orden inverso
        for t in reversed(range(T)):
            cc = ecs[t]

            # Combinar gradiente del contexto inicial (solo en t==T-1) con gH[t]
            dh_tot = gH[t] + (dh_ctx if t == T - 1 else dh_en)

            # BPTT encoder LSTM
            f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
            c_p=cc['c_prev']; conc=cc['concat']
            do=dh_tot*torch.tanh(c_t); dcc=dc_en+dh_tot*o*(1-torch.tanh(c_t)**2)
            df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
            zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
            gWfe+=torch.outer(zf,conc); gbfe+=zf
            gWie+=torch.outer(zi,conc); gbie+=zi
            gWce+=torch.outer(zc,conc); gbce+=zc
            gWoe+=torch.outer(zo,conc); gboe+=zo
            dc3=Wfe.T@zf+Wie.T@zi+Wce.T@zc+Woe.T@zo
            dh_en=dc3[:d_hid]; dc_en=dcc*f; gEenc[:,cc['emb_idx']]+=dc3[d_hid:]

        # ACTUALIZACIÓN DE PARÁMETROS

        # Atencion
        WQ -= alpha_lr*gWQ; WK -= alpha_lr*gWK; WV -= alpha_lr*gWV

        # Proyeccion de salida
        Wo2 -= alpha_lr*gWo2; bo2 -= alpha_lr*gbo2

        # Embeddings
        E_enc_tr -= alpha_lr*gEenc; E_dec_tr -= alpha_lr*gEdec

        # Encoder
        Wfe -= alpha_lr*gWfe; bfe -= alpha_lr*gbfe
        Wie -= alpha_lr*gWie; bie -= alpha_lr*gbie
        Wce -= alpha_lr*gWce; bce -= alpha_lr*gbce
        Woe -= alpha_lr*gWoe; boe -= alpha_lr*gboe

        # Decoder
        Wfd -= alpha_lr*gWfd; bfd -= alpha_lr*gbfd
        Wid -= alpha_lr*gWid; bid -= alpha_lr*gbid
        Wcd -= alpha_lr*gWcd; bcd -= alpha_lr*gbcd
        Wod -= alpha_lr*gWod; bod -= alpha_lr*gbod

    # Perdida promedio de la iteracion
    losses_train.append(total_loss / len(TRAIN_DATA))

print('Loss con atencion por iteracion:')
for i, l in enumerate(losses_train): print(f'  Iter {i+1}: {l:.4f}')

# %%
# VERIFICACION BLOQUE 9
try:
    assert len(losses_train)==5, 'losses_train debe tener 5 valores'
    assert all(losses_train[i]>losses_train[i+1] for i in range(4)), \
        f'Loss no decrece: {[round(l,4) for l in losses_train]}'
    assert abs(losses_train[0]-5.0725)<0.05, \
        f'Loss inicial incorrecta: {losses_train[0]:.4f}'
    _resultados['b9'] = True
    print('BLOQUE 9: CORRECTO')
    print(f'  {losses_train[0]:.4f} -> {losses_train[-1]:.4f} '
          f'({(losses_train[0]-losses_train[-1])/losses_train[0]*100:.2f}% reduccion)')
except AssertionError as e:
    _resultados['b9'] = False; print(f'BLOQUE 9: INCORRECTO\n  {e}')

# %%
# Mismo loop sin atencion (seq2seq de la Semana 4)

torch.manual_seed(42)
E_enc_na=torch.randn(d_emb,src_V)*0.1; E_dec_na=torch.randn(d_emb,tgt_V)*0.1
Wfe_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bfe_na=torch.zeros(d_hid)
Wie_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bie_na=torch.zeros(d_hid)
Wce_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bce_na=torch.zeros(d_hid)
Woe_na=torch.randn(d_hid,d_hid+d_emb)*0.1; boe_na=torch.zeros(d_hid)
Wfd_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bfd_na=torch.zeros(d_hid)
Wid_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bid_na=torch.zeros(d_hid)
Wcd_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bcd_na=torch.zeros(d_hid)
Wod_na=torch.randn(d_hid,d_hid+d_emb)*0.1; bod_na=torch.zeros(d_hid)
Wo2_na=torch.randn(tgt_V,d_hid)*0.1; bo2_na=torch.zeros(tgt_V)

losses_no_attn = []
for it in range(5):
    total_loss = 0.0

    for src, tgt in TRAIN_DATA:

        # FORWARD ENCODER

        src_t = tokenize_src(src)
        h = torch.zeros(d_hid); c = torch.zeros(d_hid); ecs = []
        for idx in src_t:
            emb = E_enc_na[:, idx]
            h, c, cache = lstm_cell(h, c, emb, Wfe_na, bfe_na, Wie_na, bie_na,
                                     Wce_na, bce_na, Woe_na, boe_na)
            cache['emb_idx'] = idx; ecs.append(cache)

        # Contexto fijo, no se recalcula por paso
        ctxh = h.clone(); ctxc = c.clone()
        T = len(ecs)

        # FORWARD DECODER

        tgt_f = tokenize_tgt(tgt)
        din = tgt_f[:-1]; dtg = tgt_f[1:]
        S = len(dtg)

        h2 = ctxh.clone(); c2 = ctxc.clone()
        dcs = []; zs = []
        L = torch.tensor(0.0)

        for in_idx, ti in zip(din, dtg):
            emb = E_dec_na[:, in_idx]
            h2, c2, dc = lstm_cell(h2, c2, emb, Wfd_na, bfd_na, Wid_na, bid_na,
                                    Wcd_na, bcd_na, Wod_na, bod_na)
            dc['emb_idx'] = in_idx; dc['tgt_idx'] = ti

            z = Wo2_na @ h2 + bo2_na
            dcs.append(dc); zs.append(z)

            L = L - F.log_softmax(z, dim=0)[ti]

        total_loss += (L / S).item()

        # BACKWARD DECODER

        dh_dn = torch.zeros(d_hid); dc_dn = torch.zeros(d_hid)
        gWfd=torch.zeros_like(Wfd_na); gbfd=torch.zeros_like(bfd_na)
        gWid=torch.zeros_like(Wid_na); gbid=torch.zeros_like(bid_na)
        gWcd=torch.zeros_like(Wcd_na); gbcd=torch.zeros_like(bcd_na)
        gWod=torch.zeros_like(Wod_na); gbod=torch.zeros_like(bod_na)
        gWo2=torch.zeros_like(Wo2_na); gbo2=torch.zeros_like(bo2_na)
        gEdec=torch.zeros_like(E_dec_na)

        for s in reversed(range(S)):
            cc = dcs[s]; z = zs[s]; ti = cc['tgt_idx']

            # Gradiente softmax+CE
            p = F.softmax(z, dim=0); dz = p.clone(); dz[ti] -= 1.0; dz = dz / S

            # Gradiente hacia W_out y h_dec
            gWo2 += torch.outer(dz, cc['h_t']); gbo2 += dz
            dh_s = Wo2_na.T @ dz + dh_dn

            # BPTT decoder LSTM
            f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
            c_p=cc['c_prev']; conc=cc['concat']
            do=dh_s*torch.tanh(c_t); dcc=dc_dn+dh_s*o*(1-torch.tanh(c_t)**2)
            df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
            zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
            gWfd+=torch.outer(zf,conc); gbfd+=zf
            gWid+=torch.outer(zi,conc); gbid+=zi
            gWcd+=torch.outer(zc,conc); gbcd+=zc
            gWod+=torch.outer(zo,conc); gbod+=zo
            dc2=Wfd_na.T@zf+Wid_na.T@zi+Wcd_na.T@zc+Wod_na.T@zo
            dh_dn=dc2[:d_hid]; dc_dn=dcc*f; gEdec[:,cc['emb_idx']]+=dc2[d_hid:]

        # BACKWARD ENCODER

        # Sin atencion el encoder solo recibe gradiente por el contexto
        dh_en = dh_dn.clone(); dc_en = dc_dn.clone()
        gWfe=torch.zeros_like(Wfe_na); gbfe=torch.zeros_like(bfe_na)
        gWie=torch.zeros_like(Wie_na); gbie=torch.zeros_like(bie_na)
        gWce=torch.zeros_like(Wce_na); gbce=torch.zeros_like(bce_na)
        gWoe=torch.zeros_like(Woe_na); gboe=torch.zeros_like(boe_na)
        gEenc=torch.zeros_like(E_enc_na)

        for t in reversed(range(T)):
            cc = ecs[t]

            # BPTT encoder LSTM
            f=cc['f']; i=cc['i']; ct=cc['ct']; c_t=cc['c_t']; o=cc['o']
            c_p=cc['c_prev']; conc=cc['concat']
            do=dh_en*torch.tanh(c_t); dcc=dc_en+dh_en*o*(1-torch.tanh(c_t)**2)
            df=dcc*c_p; di2=dcc*ct; dct2=dcc*i
            zf=df*f*(1-f); zi=di2*i*(1-i); zc=dct2*(1-ct**2); zo=do*o*(1-o)
            gWfe+=torch.outer(zf,conc); gbfe+=zf
            gWie+=torch.outer(zi,conc); gbie+=zi
            gWce+=torch.outer(zc,conc); gbce+=zc
            gWoe+=torch.outer(zo,conc); gboe+=zo
            dc3=Wfe_na.T@zf+Wie_na.T@zi+Wce_na.T@zc+Woe_na.T@zo
            dh_en=dc3[:d_hid]; dc_en=dcc*f; gEenc[:,cc['emb_idx']]+=dc3[d_hid:]

        # ACTUALIZACION DE PARAMETROS

        # Proyeccion de salida
        Wo2_na -= alpha_lr*gWo2; bo2_na -= alpha_lr*gbo2

        # Embeddings
        E_enc_na -= alpha_lr*gEenc; E_dec_na -= alpha_lr*gEdec

        # Encoder
        Wfe_na -= alpha_lr*gWfe; bfe_na -= alpha_lr*gbfe
        Wie_na -= alpha_lr*gWie; bie_na -= alpha_lr*gbie
        Wce_na -= alpha_lr*gWce; bce_na -= alpha_lr*gbce
        Woe_na -= alpha_lr*gWoe; boe_na -= alpha_lr*gboe

        # Decoder
        Wfd_na -= alpha_lr*gWfd; bfd_na -= alpha_lr*gbfd
        Wid_na -= alpha_lr*gWid; bid_na -= alpha_lr*gbid
        Wcd_na -= alpha_lr*gWcd; bcd_na -= alpha_lr*gbcd
        Wod_na -= alpha_lr*gWod; bod_na -= alpha_lr*gbod

    losses_no_attn.append(total_loss / len(TRAIN_DATA))

print('Loss sin atencion por iteracion:')
for i, l in enumerate(losses_no_attn): print(f'  Iter {i+1}: {l:.4f}')


# %% [markdown]
# ---
# ## Bloque 10: Visualizacion de pesos de atencion y comparacion de convergencia

# %%
# Visualizar loss con vs sin atencion
if losses_train and losses_no_attn:
    plt.figure(figsize=(8,3))
    plt.plot(range(1,6), losses_train,   'o-',  color='steelblue', lw=2, label='Con atencion')
    plt.plot(range(1,6), losses_no_attn, 's--', color='indianred', lw=2, label='Sin atencion')
    plt.xlabel('Iteracion'); plt.ylabel('Loss promedio')
    plt.title('Convergencia Seq2Seq EN->ES: con vs sin atencion')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('convergencia_atencion.png', dpi=110, bbox_inches='tight')
    plt.show()

# Visualizar mapa de atencion para un par de prueba
def get_attention_map(src_sentence, trained_params):
    """Calcula los pesos de atencion para cada paso del decoder."""
    Eenc,Edec,Wfe,bfe,Wie,bie,Wce,bce,Woe,boe,Wfd,bfd,Wid,bid,Wcd,bcd,Wod,bod,Wo2,bo2,WQ,WK,WV=trained_params
    src_t=tokenize_src(src_sentence)
    h=torch.zeros(d_hid); c=torch.zeros(d_hid); ec=[]
    for idx in src_t:
        emb=Eenc[:,idx]
        h,c,cache=lstm_cell(h,c,emb,Wfe,bfe,Wie,bie,Wce,bce,Woe,boe)
        cache['emb_idx']=idx; ec.append(cache)
    H=torch.stack([cc['h_t'] for cc in ec])
    K2=H@WK.T; V2=H@WV.T
    # Greedy decode recogiendo alphas
    h2=h.clone(); c2=c.clone()
    cur_idx=tgt_w2i[SOS]; alphas=[]; words_gen=[]
    for _ in range(10):
        emb=Edec[:,cur_idx]
        h2,c2,_=lstm_cell(h2,c2,emb,Wfd,bfd,Wid,bid,Wcd,bcd,Wod,bod)
        qs=WQ@h2; scs=K2@qs/(d_k**0.5); als=F.softmax(scs,dim=0)
        alphas.append(als.detach().numpy())
        cts=V2.T@als; h_cat=torch.cat([h2,cts]); z=Wo2@h_cat+bo2
        cur_idx=int(torch.argmax(z).item())
        word=tgt_i2w[cur_idx]
        if word==EOS: break
        words_gen.append(word)
    return alphas, words_gen, [src_i2w[i] for i in src_t]

# Usar parametros entrenados (si el bloque 9 esta completo)
if losses_train:
    params=(E_enc_tr,E_dec_tr,Wfe,bfe,Wie,bie,Wce,bce,Woe,boe,
            Wfd,bfd,Wid,bid,Wcd,bcd,Wod,bod,Wo2,bo2,WQ,WK,WV)
    src_ex='she sings well'
    alphas,words_gen,src_words=get_attention_map(src_ex,params)
    print(f'Entrada: {src_ex}')
    print(f'Generado: {" ".join(words_gen)}')
    if alphas:
        A=np.array(alphas)
        plt.figure(figsize=(6,4))
        plt.imshow(A,cmap='Blues',aspect='auto')
        plt.xticks(range(len(src_words)),src_words)
        plt.yticks(range(len(words_gen)),words_gen)
        plt.xlabel('Tokens encoder (EN)'); plt.ylabel('Pasos decoder (ES)')
        plt.title(f'Mapa de atencion: "{src_ex}"')
        plt.colorbar(label='Peso de atencion')
        plt.tight_layout()
        plt.savefig('mapa_atencion.png', dpi=110, bbox_inches='tight')
        plt.show()

# %% [markdown]
# ---
# ## Bloque 11: Preguntas de analisis
#
# ---
#
# ### Pregunta 1 (35 pts)
#
# En el Bloque 6, el gradiente de la perdida respecto a los pesos de atencion $\boldsymbol{\alpha}_s$ fluye en dos rutas distintas desde $\tilde{\mathbf{c}}_s$, y desde ahi hacia $W_Q$, $W_K$ y $W_V$.
#
# a) Trace las dos rutas del backward desde $\tilde{\mathbf{c}}_s$ hasta $H_{enc}$, explicando matematicamente que calcula cada ruta y por que son necesarias las dos. ¿Que informacion aprende el modelo a traves de cada ruta que no podria aprender sin ella?
#
# b) En el seq2seq sin atencion, el gradiente viaja $S + T$ pasos desde la perdida hasta los pesos del encoder. Con atencion, el gradiente hacia $\mathbf{h}_t^{enc}$ llega escalado por $\alpha_{s,t}$. Explique en terminos del grafo de computo como cambia el recorrido del gradiente y que implicacion tiene el factor $\alpha_{s,t}$ para el aprendizaje de tokens que raramente reciben atencion.
#
# c) Las matrices $W_Q$, $W_K$ y $W_V$ reciben el mismo hidden state del encoder como entrada pero aprenden proyecciones distintas. Sin que nadie se lo programe, ¿que diferencia esperaria observar en lo que aprende $W_Q$ versus lo que aprende $W_K$, dado que $W_Q$ proyecta el decoder y $W_K$ proyecta el encoder? Justifique en terminos del gradiente que recibe cada una.

# %% [markdown]
# **Su respuesta a la Pregunta 1:**
#
# a) Las dos rutas son las siguientes:
#
# *Ruta 1 (values, el contenido).* El gradiente fluye $\partial L/\partial\tilde{\mathbf{c}}_s \to V_{mat} \to W_V$, con $\boldsymbol{\alpha}_s$ entrando como constante de ponderación. Esta ruta ajusta *qué* información se extrae de cada token una vez que ya se decidió mirarlo.
#
# *Ruta 2 (scores, la alineación).* El gradiente fluye $\partial L/\partial\tilde{\mathbf{c}}_s \to \boldsymbol{\alpha}_s \to \mathbf{e}_s \to \mathbf{q}_s, K_{mat} \to W_Q, W_K$. Esta ruta ajusta *dónde* mirar, es decir qué tan bien coincide el query con cada key.
#
# Ambas rutas terminan sumándose en el mismo lugar, `gH += dK_s @ W_K` y `gH += dV_s @ W_V`, y las dos son necesarias. Sin la Ruta 2 los pesos de atención quedarían congelados y el modelo nunca aprendería a enfocar tokens distintos según el contexto, sin la Ruta 1 podría decidir dónde mirar pero no qué representación vale la pena extraer de ahí.
#
# b) Sin atención el gradiente hacia $\mathbf{h}_t^{enc}$ debe atravesar toda la cadena recurrente del decoder y del encoder, $S+T$ pasos, multiplicándose en cada uno por las compuertas (el problema discutido a fondo en el laboratorio anterior). Con atención el grafo gana un atajo, $L \to \tilde{\mathbf{c}}_s \to \mathbf{h}_t^{enc}$ en un solo salto sin pasar por la recurrencia.
#
# Lo interesante es que los dos caminos no transportan la misma información. El atajo le dice a $\mathbf{h}_t^{enc}$ cómo ser mejor key y mejor value para las queries que lo seleccionaron, pero solo el camino recurrente le dice cómo ser mejor resumen de la secuencia, porque es el único que atraviesa la cadena del encoder. Un token muy atendido sigue necesitando el camino largo para aprender su papel dentro de la oración.
#
# El factor $\alpha_{s,t}$ agrega una dependencia circular. La cantidad de gradiente que recibe un token depende de la atención que el modelo ya le asigna, y esa atención es justamente lo que todavía no ha aprendido. Un token con $\alpha_{s,t} \approx 0$ recibe casi nada por la vía corta, lo que refuerza que siga recibiendo poco peso. Lo que rompe el ciclo al inicio es que la distribución arranca prácticamente uniforme, como se ve en el Bloque 3 donde $\boldsymbol{\alpha} \approx 1/3$ para los tres tokens, de modo que todos reciben gradiente comparable antes de que el modelo empiece a discriminar.
#
# c) Aunque las dos matrices proyectan al mismo espacio de dimensión $d_k$, el gradiente que recibe cada una viene de una fuente distinta. El de $W_Q$ se construye con el estado del decoder, $\mathbf{q}_s^{grad} \otimes \mathbf{h}_s^{dec}$, mientras que el de $W_K$ se construye con los estados del encoder, $(\partial L/\partial K_{mat})^\top H_{enc}$. Ninguna de las dos ve nunca la entrada de la otra.
#
# A eso se suma una asimetría en cuántas veces participa cada vector. Cada query $\mathbf{q}_s$ interviene en un solo paso de generación y recibe gradiente de ese paso únicamente, pero cada key $\mathbf{k}_t$ interviene en los scores de todos los pasos del decoder, así que $W_K$ acumula gradiente promediado sobre muchas consultas distintas antes de cada actualización.
#
# Por eso esperaría comportamientos distintos. $W_Q$ debería especializarse por paso de generación, aprendiendo algo parecido a "qué necesito buscar ahora" en función de lo que el decoder lleva emitido. $W_K$ debería converger a representaciones más genéricas y reutilizables, algo parecido a "qué tengo para ofrecer", porque tiene que servirle bien a todas las queries a la vez y no puede ajustarse a ninguna en particular.

# %% [markdown]
# ---
# ### Pregunta 2 (35 pts)
#
# Observe el mapa de atencion generado en el Bloque 10 para la oracion 'she sings well' -> 'canta bien'.
#
# a) La oracion en ingles tiene sujeto ('she') y la traduccion al espanol lo elide ('canta bien' sin 'ella'). En el paso del decoder que genera 'canta', ¿que tokens del encoder esperaria que recibieran mayor peso de atencion y por que? ¿Es el sujeto 'she' relevante para generar 'canta'? Conecte su respuesta con lo que el mecanismo de atencion esta matematicamente calculando.
#
# b) Si entrenara el mismo modelo durante 100 iteraciones en lugar de 5, ¿esperaria que el mapa de atencion se volviera mas o menos concentrado (pesos mas o menos uniformes)? Justifique en terminos de lo que el modelo aprende progresivamente sobre las correspondencias entre palabras en ingles y espanol.
#
# c) Proponga un caso especifico del corpus donde el mecanismo de atencion de **producto punto escalado** tendria dificultad para capturar la correspondencia correcta, y explique por que. ¿Que alternativa arquitectonica (sin necesidad de implementarla) resolveria ese caso? Justifique matematicamente.

# %% [markdown]
# **Su respuesta a la Pregunta 2:**
#
# a) Al generar 'canta' esperaría casi todo el peso en 'sings', que es donde está el verbo. 'she' debería recibir algo pero poco. En este corpus 'she sings' y 'he sings' dan lo mismo, así que el género no aporta nada. Lo que sí importa del sujeto es la persona y el número, como se ve en 'we eat bread' → 'comemos pan' contra 'he eats an apple' → 'come una manzana'.
#
# El score $e_{s,t} = \mathbf{q}_s^\top \mathbf{k}_t / \sqrt{d_k}$ mide qué tanto se parece lo que el decoder busca a lo que ofrece cada token. Como $\boldsymbol{\alpha}_s$ es una distribución y no una elección de uno solo, $\tilde{\mathbf{c}}_s$ puede mezclar el verbo con algo de concordancia y mandar todo junto a $W_{out}$.
#
# En el mapa se ve un degradado hacia 'well', pero los tres pesos son casi iguales, así que el color no significa mayor cosa. Con 5 iteraciones el modelo no ha aprendido ninguna alineación todavía.
#
# b) Esperaría que se concentre más. Ahora los pesos salen casi uniformes porque el modelo no ha aprendido nada; conforme aprenda qué palabra va con cuál, los productos punto de las parejas correctas crecen más rápido que el resto. Como el softmax es exponencial, una diferencia mediana en los scores se vuelve una diferencia grande en los pesos.
#
# Aún así no todos los pasos se van a concentrar. El factor $1/\sqrt{d_k}$ está justamente para que los scores no se disparen. Además hay pasos que no tienen una palabra fuente a la cual apuntar. Al generar `<EOS>` ninguna palabra del inglés dice "aquí se acaba la oración", eso sale del estado del decoder, así que ahí la atención se queda repartida.
#
# c) Un caso del corpus es 'the cat sleeps' → 'el gato está durmiendo', que se repite en doce pares. El inglés tiene tres tokens y el español cuatro, y 'está' no corresponde a ninguno de la fuente, así que en ese paso $\boldsymbol{\alpha}_s$ se reparte entre tokens que no traen esa información. Además $\mathbf{q}_s^\top\mathbf{k}_t$ da lo mismo si se reordenan las posiciones, así que el modelo tampoco puede apoyarse en el orden.
#
# Una alternativa es multi-head attention. Correr el mismo mecanismo $H$ veces en paralelo, cada una con sus propias $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$, y pegar los $H$ contextos al final. Con una sola cabeza los pesos suman 1, así que mirar 'sings' significa no mirar 'she'; con varias cabezas cada una arma su propia distribución y una puede quedarse con el sujeto mientras otra se queda con el verbo.

# %% [markdown]
# ---
# ### Pregunta 3 (30 pts)
#
# Esta pregunta explora self-attention conceptualmente, como preparacion para la Semana 6.
#
# a) En el mecanismo de atencion que implemento, los queries vienen del decoder y las keys y values del encoder. En self-attention, los tres vienen de la misma secuencia. Identifique exactamente que lineas de codigo del Bloque 6 cambiarian si convirtiera su implementacion de cross-attention a self-attention sobre la secuencia del encoder. ¿Que nueva informacion capturaria el modelo que la arquitectura LSTM no puede capturar directamente?
#
# b) En self-attention, cada posicion calcula su query usando $W_Q$ y cada posicion ofrece su key usando $W_K$. Ambas matrices se inicializan aleatoriamente e identicamente podrian converger al mismo valor. Sin embargo, en la practica aprenden proyecciones distintas. Explique matematicamente, usando la estructura del grafo de computo, por que el gradiente que llega a $W_Q$ es diferente al que llega a $W_K$, incluso cuando reciben los mismos vectores de entrada.
#
# c) Un Transformer con 8 cabezas de atencion ('multi-head attention') aplica 8 mecanismos de atencion en paralelo sobre la misma secuencia. Si todas las cabezas compartieran las mismas matrices $W_Q$, $W_K$, $W_V$, ¿que pasaria con el gradiente durante el entrenamiento y por que ese diseno no funcionaria bien? Justifique en terminos de la diversidad de representaciones que el modelo necesita aprender.

# %% [markdown]
# **Su respuesta a la Pregunta 3:**
#
# a) En el forward, `q_0 = W_Q @ ctx_h` pasaría a ser `Q = H_enc @ W_Q.T`, una query por cada token del encoder en vez de una sola sacada del decoder. Eso hace que los scores dejen de ser un vector $(T,)$ y pasen a ser una matriz $T \times T$, cada posición contra todas las demás. $K_{mat}$ y $V_{mat}$ no cambian porque ya salen de `H_enc`.
#
# En el Bloque 6 cambian tres líneas, porque la query ya no viene del decoder. `dW_Q += torch.outer(dq_s, ac['h_dec'])` usaría `H_enc[s]`; `dh_s = dh_s + W_Q.T @ dq_s` se borra, porque no hay nada que devolverle al decoder; y `dH_enc` sumaría una tercera contribución, la de las queries.
#
# Con self-attention el modelo puede conectar dos tokens de la misma oración en un solo salto y en las dos direcciones, por ejemplo un sujeto con su verbo aunque estén lejos. La LSTM tiene que pasar esa información por toda la cadena y esperar que sobreviva.
#
# b) En el grafo de cómputo, $W_Q$ y $W_K$ están en ramas separadas. Solo se juntan en el producto $e_{s,t} = \mathbf{q}_s^\top \mathbf{k}_t$. $W_Q$ hace que un token pregunte y $W_K$ hace que un token responda. Como el score es un producto de las dos, al derivar cada una queda multiplicada por la otra, el gradiente de $W_Q$ depende de las keys y el de $W_K$ depende de las queries.
#
# Hay otra diferencia. Cada query se usa en una sola fila de la matriz de scores, pero cada key se usa en toda una columna. Entonces $W_K$ junta gradiente de todas las posiciones que la consultaron y $W_Q$ solo de la suya. Aunque las dos arranquen iguales, en la primera actualización ya reciben cosas distintas y se separan.
#
# c) Si las 8 cabezas compartieran $W_Q, W_K, W_V$, las 8 harían la misma cuenta y darían el mismo resultado. Los gradientes no serían iguales, eso sí, porque $W_O$ actúa sobre la concatenación y cada cabeza entra multiplicada por un bloque distinto, así que cada una devuelve un gradiente distinto y las matrices compartidas terminan acumulando la suma de los 8. Esa suma no apunta a nada en particular, es un promedio de lo que 8 bloques querían por separado. Y como las tres matrices son las mismas para todas, por mucho que se actualicen las 8 cabezas siguen dando el mismo vector y nunca se separan.

# %% [markdown]
# ---
# ## Bloque 12: Nota automatica sobre la seccion de codigo

# %%
_PUNTOS = {
    'b1': ('Bloque 1: Proyecciones Q, K, V',        8),
    'b2': ('Bloque 2: Attention scores',             8),
    'b3': ('Bloque 3: Attention weights',            6),
    'b4': ('Bloque 4: Vector de contexto dinamico',  6),
    'b5': ('Bloque 5: Forward decoder con atencion', 14),
    'b6': ('Bloque 6: Backward de atencion',         10),
    'b8': ('Bloque 8: Actualizacion de parametros',  5),
    'b9': ('Bloque 9: Convergencia 5 iteraciones',   3),
}
_TOTAL = 60
print('='*62)
print('  NOTA AUTOMATICA - SECCION DE CODIGO')
print('='*62)
_obtenido=0
for key,(nombre,pts_max) in _PUNTOS.items():
    val=_resultados.get(key,False)
    pts=pts_max if val is True else 0
    _obtenido+=pts
    print(f'  {"CORRECTO" if val is True else "PENDIENTE":10s} | {nombre:38s} | {pts:2d}/{pts_max} pts')
print('-'*62)
print(f'  Subtotal codigo:   {_obtenido}/{_TOTAL} puntos')
print('  Pendiente manual:')
print('    Bloque 11 preguntas : __/25 pts')
print('    Comentarios codigo  : __/15 pts')
print('-'*62)
print('  TOTAL FINAL (sobre 100): __/100 pts')
print('='*62)
