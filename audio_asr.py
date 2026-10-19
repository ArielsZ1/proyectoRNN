import json
import os
import shutil
import uuid

import numpy as np
import tensorflow as tf

from common import (DATA_DIR, MODELS_DIR, append_tsv, confidence, ensure_dir,
                    fit_ctc, greedy_decode, read_tsv)

layers = tf.keras.layers

SR = 16000
N_MELS = 64
FRAME_LEN, FRAME_STEP, FFT_LEN = 400, 160, 512
MAX_FRAMES = 800          # ~8 segundos por fragmento
MIN_FRAMES = 30           # ~0.3 segundos
MIN_SAMPLES = 10

FOLDER = os.path.join(DATA_DIR, "audio")
CLIPS_DIR = os.path.join(FOLDER, "clips")
LABELS_PATH = os.path.join(FOLDER, "transcripts.tsv")
MODEL_DIR = os.path.join(MODELS_DIR, "audio")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CHARSET_PATH = os.path.join(MODEL_DIR, "charset.json")

_cache = {}


def _normalize(text):
    return " ".join(text.lower().split())


def _decode_wav(data):
    try:
        audio, sr = tf.audio.decode_wav(data, desired_channels=-1)
    except Exception:
        raise ValueError("Archivo WAV invalido (se requiere WAV PCM de 16 bits).") from None
    if int(sr.numpy()) != SR:
        raise ValueError(f"La frecuencia de muestreo debe ser {SR} Hz (recibido {int(sr.numpy())} Hz).")
    return tf.reduce_mean(audio, axis=1)  # mezcla a mono


def _features(samples):
    stft = tf.signal.stft(samples, frame_length=FRAME_LEN,
                          frame_step=FRAME_STEP, fft_length=FFT_LEN)
    spec = tf.abs(stft) ** 2
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=FFT_LEN // 2 + 1,
        sample_rate=SR, lower_edge_hertz=80.0, upper_edge_hertz=7600.0)
    logmel = tf.math.log(tf.tensordot(spec, mel_w, 1) + 1e-6).numpy()
    if logmel.shape[0] < MIN_FRAMES:
        raise ValueError("El audio es demasiado corto.")
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


def info():
    return {
        "samples": len(read_tsv(LABELS_PATH)),
        "min_samples": MIN_SAMPLES,
        "trained": os.path.exists(MODEL_PATH),
    }


def add_sample(data, text):
    text = _normalize(text)
    if not text:
        raise ValueError("Escribe la transcripcion del audio.")
    feats = _features(_decode_wav(data))
    if feats.shape[0] > MAX_FRAMES:
        raise ValueError("Cada clip de entrenamiento debe durar como maximo ~8 segundos.")
    ensure_dir(CLIPS_DIR)
    name = f"{uuid.uuid4().hex}.wav"
    with open(os.path.join(CLIPS_DIR, name), "wb") as f:
        f.write(data)
    append_tsv(LABELS_PATH, name, text)


def clear():
    shutil.rmtree(FOLDER, ignore_errors=True)


def build_model(n_chars):
    inp = layers.Input(shape=(MAX_FRAMES, N_MELS, 1))
    x = layers.Conv2D(32, (11, 41), strides=(2, 2), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (11, 21), strides=(1, 2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)            # (400, 16, 32)
    x = layers.Reshape((MAX_FRAMES // 2, (N_MELS // 4) * 32))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_chars + 1)(x)            # +1 = simbolo blanco CTC
    return tf.keras.Model(inp, out)


def train(epochs, on_epoch):
    feats_list, texts, lens = [], [], []
    for name, text in read_tsv(LABELS_PATH):
        path = os.path.join(CLIPS_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                feats = _features(_decode_wav(f.read()))
        except ValueError:
            continue
        text = _normalize(text)
        out_len = (feats.shape[0] + 1) // 2
        if feats.shape[0] > MAX_FRAMES or not text or len(text) > out_len:
            continue
        feats_list.append(feats)
        texts.append(text)
        lens.append(out_len)
    n = len(feats_list)
    if n < MIN_SAMPLES:
        raise ValueError(f"Se necesitan al menos {MIN_SAMPLES} clips validos (hay {n}).")

    charset = sorted(set("".join(texts)))
    c2i = {c: i for i, c in enumerate(charset)}
    x = np.zeros((n, MAX_FRAMES, N_MELS, 1), dtype=np.float32)
    max_len = max(len(t) for t in texts)
    y = np.zeros((n, max_len), dtype=np.int32)
    for i, (feats, t) in enumerate(zip(feats_list, texts)):
        x[i, :feats.shape[0], :, 0] = feats
        y[i, :len(t)] = [c2i[c] for c in t]
    y_len = np.array([len(t) for t in texts], dtype=np.int32)
    x_len = np.array(lens, dtype=np.int32)

    model = build_model(len(charset))
    fit_ctc(model, x, y, y_len, x_len, blank=len(charset),
            epochs=epochs, on_epoch=on_epoch, batch_size=8)

    ensure_dir(MODEL_DIR)
    model.save(MODEL_PATH)
    with open(CHARSET_PATH, "w", encoding="utf-8") as f:
        json.dump(charset, f, ensure_ascii=False)
    _cache.clear()


def _load():
    if "model" not in _cache:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(CHARSET_PATH)):
            raise FileNotFoundError("Modelo de audio no entrenado.")
        _cache["model"] = tf.keras.models.load_model(MODEL_PATH)
        with open(CHARSET_PATH, "r", encoding="utf-8") as f:
            _cache["charset"] = json.load(f)
    return _cache["model"], _cache["charset"]


def predict(data):
    model, charset = _load()
    feats = _features(_decode_wav(data))
    chunks = [feats[s:s + MAX_FRAMES] for s in range(0, len(feats), MAX_FRAMES)]
    chunks = [c for c in chunks if len(c) >= MIN_FRAMES]
    if not chunks:
        raise ValueError("El audio es demasiado corto.")

    x = np.zeros((len(chunks), MAX_FRAMES, N_MELS, 1), dtype=np.float32)
    for i, c in enumerate(chunks):
        x[i, :len(c), :, 0] = c
    logits = model(x, training=False).numpy()

    blank = len(charset)
    parts, confs = [], []
    for i, c in enumerate(chunks):
        out_len = (len(c) + 1) // 2
        ids = greedy_decode(logits[i], out_len, blank)
        parts.append("".join(charset[j] for j in ids).strip())
        confs.append(confidence(logits[i], out_len, blank))
    return {
        "text": " ".join(p for p in parts if p),
        "confidence": float(np.mean(confs)),
    }