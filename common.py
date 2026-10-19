import os

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEED = 123


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_tsv(path):
    """Lee un archivo TSV y devuelve una lista de tuplas (col1, col2)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def append_tsv(path, a, b):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{a}\t{b}\n")


def edit_distance(a, b):
    """Distancia de Levenshtein entre dos secuencias."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def greedy_decode(logits, length, blank):
    """Decodificacion CTC voraz: argmax, colapsar repetidos, quitar blanco."""
    ids = np.argmax(logits[:length], axis=-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def confidence(logits, length, blank):
    """Confianza aproximada: probabilidad media de los pasos no-blanco."""
    probs = softmax(logits[:length])
    best = probs.argmax(axis=-1)
    mask = best != blank
    if not mask.any():
        return 0.0
    return float(probs.max(axis=-1)[mask].mean())


def ctc_loss(labels, logits, label_len, logit_len):
    """logits: [batch, tiempo, clases]. El blanco es la ultima clase."""
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_len,
        logit_length=logit_len,
        logits_time_major=False,
        blank_index=-1,
    )
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
    return tf.reduce_mean(loss)


def fit_ctc(model, x, y, y_len, x_len, blank, epochs, on_epoch,
            batch_size=16, lr=1e-3):
    """Bucle de entrenamiento CTC con division 80/20 (semilla fija).

    x: [n, ...] float32 | y: [n, max_label] int32
    y_len, x_len: [n] int32 (longitud de etiqueta y de secuencia de salida)
    on_epoch(epoch, metricas) se llama al terminar cada epoca.
    """
    n = len(x)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * 0.2)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    opt.build(model.trainable_variables)

    @tf.function
    def train_step(xb, yb, ylb, xlb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = ctc_loss(yb, logits, ylb, xlb)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def evaluate(ids):
        total_loss, dist, chars = 0.0, 0, 0
        for s in range(0, len(ids), batch_size):
            b = ids[s:s + batch_size]
            logits = model(x[b], training=False)
            total_loss += float(ctc_loss(y[b], logits, y_len[b], x_len[b])) * len(b)
            logits = logits.numpy()
            for k, i in enumerate(b):
                pred = greedy_decode(logits[k], int(x_len[i]), blank)
                ref = y[i][: y_len[i]].tolist()
                dist += edit_distance(pred, ref)
                chars += max(1, len(ref))
        return total_loss / len(ids), dist / chars

    for epoch in range(1, epochs + 1):
        rng.shuffle(tr_idx)
        losses = []
        for s in range(0, len(tr_idx), batch_size):
            b = tr_idx[s:s + batch_size]
            loss = train_step(
                tf.constant(x[b]), tf.constant(y[b]),
                tf.constant(y_len[b]), tf.constant(x_len[b]),
            )
            losses.append(float(loss))
        val_loss, val_cer = evaluate(val_idx)
        on_epoch(epoch, {
            "loss": float(np.mean(losses)),
            "val_loss": float(val_loss),
            "val_cer": float(val_cer),
        })