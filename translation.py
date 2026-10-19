import json
import os
import random
import re
import shutil
from collections import Counter

import numpy as np
import tensorflow as tf

from common import DATA_DIR, MODELS_DIR, SEED, ensure_dir, read_tsv

layers = tf.keras.layers

MAX_LEN = 20
MAX_VOCAB = 8000
MAX_PAIRS = 30000
MIN_PAIRS = 50
EMB, UNITS = 128, 256
PAD, UNK, START, END = 0, 1, 2, 3
SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_cache = {}


def clean_pair(pair):
    pair = re.sub(r"[^a-z0-9_-]", "", (pair or "").lower())
    if not pair:
        raise ValueError("Indica un nombre de par valido (por ejemplo: en-es).")
    return pair


def _data_path(pair):
    return os.path.join(DATA_DIR, "translation", pair, "pairs.tsv")


def _model_dir(pair):
    return os.path.join(MODELS_DIR, "translation", pair)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?)%])", r"\1", text)
    text = re.sub(r"([¿¡(])\s+", r"\1", text)
    return text[:1].upper() + text[1:] if text else text


def list_pairs():
    base = os.path.join(DATA_DIR, "translation")
    result = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            path = _data_path(name)
            if os.path.exists(path):
                result.append({
                    "name": name,
                    "pairs": len(read_tsv(path)),
                    "trained": os.path.exists(os.path.join(_model_dir(name), "model.keras")),
                })
    return result


def add_corpus(pair, raw):
    """Agrega pares desde un TSV (origen<TAB>destino[<TAB>otras columnas])."""
    pair = clean_pair(pair)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("El archivo debe estar codificado en UTF-8.") from None
    ensure_dir(os.path.dirname(_data_path(pair)))
    added = 0
    with open(_data_path(pair), "a", encoding="utf-8") as f:
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src, tgt = " ".join(parts[0].split()), " ".join(parts[1].split())
            if src and tgt:
                f.write(f"{src}\t{tgt}\n")
                added += 1
    if added == 0:
        raise ValueError("No se encontraron pares (formato: origen<TAB>destino por linea).")
    _cache.pop(pair, None)
    return added


def delete_pair(pair):
    pair = clean_pair(pair)
    shutil.rmtree(os.path.dirname(_data_path(pair)), ignore_errors=True)
    shutil.rmtree(_model_dir(pair), ignore_errors=True)
    _cache.pop(pair, None)


def _build_vocab(token_lists):
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, _ in counts.most_common(MAX_VOCAB - len(SPECIAL))]
    return SPECIAL + words


def _build_model(src_vocab, tgt_vocab):
    enc_in = layers.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(src_vocab, EMB, mask_zero=True)(enc_in)
    _, state = layers.GRU(UNITS, return_state=True)(x)

    dec_in = layers.Input(shape=(MAX_LEN + 1,), dtype="int32")
    y = layers.Embedding(tgt_vocab, EMB, mask_zero=True)(dec_in)
    y = layers.GRU(UNITS, return_sequences=True)(y, initial_state=state)
    y = layers.Dropout(0.3)(y)
    out = layers.Dense(tgt_vocab, activation="softmax")(y)
    return tf.keras.Model([enc_in, dec_in], out)


class _Progress(tf.keras.callbacks.Callback):
    def __init__(self, on_epoch):
        super().__init__()
        self.on_epoch = on_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.on_epoch(epoch + 1, {k: float(v) for k, v in (logs or {}).items()})


def train(pair, epochs, on_epoch):
    pair = clean_pair(pair)
    rows = read_tsv(_data_path(pair))
    random.Random(SEED).shuffle(rows)
    rows = rows[:MAX_PAIRS]

    src_toks = [tokenize(s) for s, _ in rows]
    tgt_toks = [tokenize(t) for _, t in rows]
    keep = [i for i in range(len(rows))
            if 0 < len(src_toks[i]) <= MAX_LEN and 0 < len(tgt_toks[i]) <= MAX_LEN]
    if len(keep) < MIN_PAIRS:
        raise ValueError(f"Se necesitan al menos {MIN_PAIRS} pares validos (hay {len(keep)}).")

    src_itos = _build_vocab([src_toks[i] for i in keep])
    tgt_itos = _build_vocab([tgt_toks[i] for i in keep])
    s2i = {w: i for i, w in enumerate(src_itos)}
    t2i = {w: i for i, w in enumerate(tgt_itos)}

    n = len(keep)
    enc = np.zeros((n, MAX_LEN), dtype=np.int32)
    dec_in = np.zeros((n, MAX_LEN + 1), dtype=np.int32)
    dec_out = np.zeros((n, MAX_LEN + 1), dtype=np.int32)
    for k, i in enumerate(keep):
        s = [s2i.get(t, UNK) for t in src_toks[i]]
        t = [t2i.get(w, UNK) for w in tgt_toks[i]]
        enc[k, :len(s)] = s
        dec_in[k, 0] = START
        dec_in[k, 1:len(t) + 1] = t
        dec_out[k, :len(t)] = t
        dec_out[k, len(t)] = END

    model = _build_model(len(src_itos), len(tgt_itos))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit([enc, dec_in], dec_out, batch_size=64, epochs=epochs,
              validation_split=0.2, shuffle=True, verbose=0,
              callbacks=[_Progress(on_epoch)])

    out_dir = ensure_dir(_model_dir(pair))
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"src": src_itos, "tgt": tgt_itos}, f, ensure_ascii=False)
    _cache.pop(pair, None)


def _load(pair):
    if pair not in _cache:
        d = _model_dir(pair)
        mp, vp = os.path.join(d, "model.keras"), os.path.join(d, "vocab.json")
        if not (os.path.exists(mp) and os.path.exists(vp)):
            raise FileNotFoundError("Modelo de traduccion no entrenado.")
        model = tf.keras.models.load_model(mp)
        with open(vp, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        _cache[pair] = (model, vocab["src"], vocab["tgt"])
    return _cache[pair]


def translate(pair, text):
    pair = clean_pair(pair)
    model, src_itos, tgt_itos = _load(pair)
    s2i = {w: i for i, w in enumerate(src_itos)}
    tokens = tokenize(text)[:MAX_LEN]
    if not tokens:
        raise ValueError("Escribe un texto para traducir.")

    enc = np.zeros((1, MAX_LEN), dtype=np.int32)
    enc[0, :len(tokens)] = [s2i.get(t, UNK) for t in tokens]
    dec = np.zeros((1, MAX_LEN + 1), dtype=np.int32)
    dec[0, 0] = START

    result = []
    for t in range(MAX_LEN + 1):
        probs = model.predict([enc, dec], verbose=0)[0, t].copy()
        probs[PAD] = 0.0
        probs[UNK] = 0.0
        nxt = int(np.argmax(probs))
        if nxt == END:
            break
        result.append(tgt_itos[nxt])
        if t + 1 < MAX_LEN + 1:
            dec[0, t + 1] = nxt
    return {"text": _detokenize(result)}