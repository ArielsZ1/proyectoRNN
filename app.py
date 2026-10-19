import os
import threading
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, request, send_from_directory

import audio_asr
import handwriting
import translation
from common import BASE_DIR

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

_lock = threading.Lock()
_state = {"status": "idle", "task": None, "epoch": 0, "epochs": 0,
          "progress": 0, "history": [], "error": None}


def err(message, code=400):
    return jsonify(error=message), code


def _run_training(fn, epochs):
    def on_epoch(epoch, metrics):
        with _lock:
            _state["epoch"] = epoch
            _state["progress"] = round(100 * epoch / epochs)
            _state["history"].append({"epoch": epoch, **metrics})

    try:
        fn(epochs, on_epoch)
        with _lock:
            _state.update(status="done", progress=100)
    except Exception as e:  # el error se muestra en la interfaz
        traceback.print_exc()
        with _lock:
            _state.update(status="error", error=str(e))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/overview")
def overview():
    return jsonify(handwriting=handwriting.info(), audio=audio_asr.info(),
                   translation=translation.list_pairs())


@app.get("/api/status")
def status():
    with _lock:
        return jsonify(dict(_state, history=list(_state["history"])))


# ---------- Entrenamiento ----------
@app.post("/api/train/<task>")
def train(task):
    body = request.get_json(silent=True) or {}
    try:
        epochs = int(body.get("epochs", 10))
    except (TypeError, ValueError):
        return err("El numero de epocas no es valido.")
    if not 1 <= epochs <= 200:
        return err("Las epocas deben estar entre 1 y 200.")

    if task == "handwriting":
        fn = handwriting.train
    elif task == "audio":
        fn = audio_asr.train
    elif task == "translation":
        try:
            pair = translation.clean_pair(body.get("pair"))
        except ValueError as e:
            return err(str(e))
        fn = lambda e, cb: translation.train(pair, e, cb)
    else:
        return err("Tarea desconocida.", 404)

    with _lock:
        if _state["status"] == "training":
            return err("Ya hay un entrenamiento en curso.", 409)
        _state.update(status="training", task=task, epoch=0, epochs=epochs,
                      progress=0, history=[], error=None)
    threading.Thread(target=_run_training, args=(fn, epochs), daemon=True).start()
    return jsonify(ok=True)


# ---------- Manuscrito ----------
@app.post("/api/handwriting/samples")
def hw_add():
    f = request.files.get("image")
    if not f:
        return err("Falta el archivo 'image'.")
    try:
        handwriting.add_sample(f.read(), request.form.get("text", ""))
    except ValueError as e:
        return err(str(e))
    return jsonify(ok=True, **handwriting.info())


@app.delete("/api/handwriting/samples")
def hw_clear():
    handwriting.clear()
    return jsonify(ok=True)


@app.post("/api/handwriting/predict")
def hw_predict():
    f = request.files.get("image")
    if not f:
        return err("Falta el archivo 'image'.")
    try:
        return jsonify(handwriting.predict(f.read()))
    except FileNotFoundError:
        return err("Todavia no hay un modelo de manuscrito entrenado.", 409)
    except ValueError as e:
        return err(str(e))


# ---------- Audio ----------
@app.post("/api/audio/samples")
def audio_add():
    f = request.files.get("audio")
    if not f:
        return err("Falta el archivo 'audio'.")
    try:
        audio_asr.add_sample(f.read(), request.form.get("text", ""))
    except ValueError as e:
        return err(str(e))
    return jsonify(ok=True, **audio_asr.info())


@app.delete("/api/audio/samples")
def audio_clear():
    audio_asr.clear()
    return jsonify(ok=True)


@app.post("/api/audio/predict")
def audio_predict():
    f = request.files.get("audio")
    if not f:
        return err("Falta el archivo 'audio'.")
    try:
        return jsonify(audio_asr.predict(f.read()))
    except FileNotFoundError:
        return err("Todavia no hay un modelo de audio entrenado.", 409)
    except ValueError as e:
        return err(str(e))


# ---------- Traduccion ----------
@app.post("/api/translation/corpus")
def tr_corpus():
    f = request.files.get("file")
    if not f:
        return err("Falta el archivo 'file'.")
    try:
        added = translation.add_corpus(request.form.get("pair"), f.read())
    except ValueError as e:
        return err(str(e))
    return jsonify(ok=True, added=added)


@app.delete("/api/translation/<pair>")
def tr_delete(pair):
    try:
        translation.delete_pair(pair)
    except ValueError as e:
        return err(str(e))
    return jsonify(ok=True)


@app.post("/api/translate")
def tr_translate():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(translation.translate(body.get("pair"), body.get("text", "")))
    except FileNotFoundError:
        return err("Ese par de idiomas todavia no esta entrenado.", 409)
    except ValueError as e:
        return err(str(e))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)