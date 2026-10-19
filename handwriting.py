import io
import json
import os
import shutil
import uuid

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from common import (DATA_DIR, MODELS_DIR, append_tsv, confidence, ensure_dir,
                    fit_ctc, greedy_decode, read_tsv)

layers = tf.keras.layers

IMG_H, IMG_W = 64, 256
TIME_STEPS = IMG_W // 4
MAX_LABEL = 32
MIN_SAMPLES = 10

FOLDER = os.path.join(DATA_DIR, "handwriting")
IMAGES_DIR = os.path.join(FOLDER, "images")
LABELS_PATH = os.path.join(FOLDER, "labels.tsv")
MODEL_DIR = os.path.join(MODELS_DIR, "handwriting")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CHARSET_PATH = os.path.join(MODEL_DIR, "charset.json")

_cache = {}


def _open_gray(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ValueError("No se pudo leer la imagen (JPEG, PNG, BMP, GIF o WebP).") from None
    return ImageOps.exif_transpose(img).convert("L")


def _to_array(img):
    """Escala a alto 64, tinta = 1, fondo = 0, y rellena hasta 256 px de ancho."""
    img = ImageOps.autocontrast(img)
    w, h = img.size
    new_w = max(1, min(IMG_W, int(round(w * IMG_H / h))))
    img = img.resize((new_w, IMG_H), Image.Resampling.LANCZOS)
    arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    canvas = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    canvas[:, :new_w] = arr
    return canvas[..., np.newaxis]


def info():
    return {
        "samples": len(read_tsv(LABELS_PATH)),
        "min_samples": MIN_SAMPLES,
        "trained": os.path.exists(MODEL_PATH),
    }


def add_sample(data, text):
    text = " ".join(text.split())
    if not text:
        raise ValueError("Escribe la transcripcion de la imagen.")
    if len(text) > MAX_LABEL:
        raise ValueError(f"La transcripcion admite hasta {MAX_LABEL} caracteres.")
    img = _open_gray(data)
    ensure_dir(IMAGES_DIR)
    name = f"{uuid.uuid4().hex}.png"
    img.save(os.path.join(IMAGES_DIR, name))
    append_tsv(LABELS_PATH, name, text)


def clear():
    shutil.rmtree(FOLDER, ignore_errors=True)


def build_model(n_chars):
    inp = layers.Input(shape=(IMG_H, IMG_W, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D((2, 2))(x)           # 32 x 128
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)           # 16 x 64
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)           # 8 x 64
    x = layers.Permute((2, 1, 3))(x)             # (ancho, alto, canales)
    x = layers.Reshape((TIME_STEPS, (IMG_H // 8) * 128))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    out = layers.Dense(n_chars + 1)(x)           # +1 = simbolo blanco CTC
    return tf.keras.Model(inp, out)


def train(epochs, on_epoch):
    xs, texts = [], []
    for name, text in read_tsv(LABELS_PATH):
        path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(path) or len(text) > MAX_LABEL:
            continue
        try:
            with open(path, "rb") as f:
                xs.append(_to_array(_open_gray(f.read())))
            texts.append(text)
        except ValueError:
            continue  # archivo invalido: se omite
    if len(xs) < MIN_SAMPLES:
        raise ValueError(f"Se necesitan al menos {MIN_SAMPLES} muestras validas (hay {len(xs)}).")

    charset = sorted(set("".join(texts)))
    c2i = {c: i for i, c in enumerate(charset)}
    n = len(xs)
    max_len = max(len(t) for t in texts)
    y = np.zeros((n, max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        y[i, :len(t)] = [c2i[c] for c in t]
    y_len = np.array([len(t) for t in texts], dtype=np.int32)
    x = np.stack(xs).astype(np.float32)
    x_len = np.full(n, TIME_STEPS, dtype=np.int32)

    model = build_model(len(charset))
    fit_ctc(model, x, y, y_len, x_len, blank=len(charset),
            epochs=epochs, on_epoch=on_epoch, batch_size=16)

    ensure_dir(MODEL_DIR)
    model.save(MODEL_PATH)
    with open(CHARSET_PATH, "w", encoding="utf-8") as f:
        json.dump(charset, f, ensure_ascii=False)
    _cache.clear()


def _load():
    if "model" not in _cache:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(CHARSET_PATH)):
            raise FileNotFoundError("Modelo de manuscrito no entrenado.")
        _cache["model"] = tf.keras.models.load_model(MODEL_PATH)
        with open(CHARSET_PATH, "r", encoding="utf-8") as f:
            _cache["charset"] = json.load(f)
    return _cache["model"], _cache["charset"]


def predict(data):
    model, charset = _load()
    x = _to_array(_open_gray(data))[np.newaxis]
    logits = model(x, training=False).numpy()[0]
    blank = len(charset)
    ids = greedy_decode(logits, TIME_STEPS, blank)
    return {
        "text": "".join(charset[i] for i in ids),
        "confidence": confidence(logits, TIME_STEPS, blank),
    }