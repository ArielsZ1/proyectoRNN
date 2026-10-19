# Prompt para la herramienta de IA (generador de interfaz)

Actua como disenador y desarrollador frontend. Genera UN SOLO archivo `index.html`
(HTML + CSS + JavaScript vanilla, sin librerias ni CDN, en espanol) para administrar
una aplicacion local de redes neuronales RNN que:
1) transcribe texto manuscrito desde imagenes,
2) transcribe audio (WAV) a texto,
3) traduce texto entre idiomas.

## Diseno
- Estilo moderno y limpio, tarjetas con bordes redondeados, color primario indigo,
  soporte de modo claro/oscuro con `prefers-color-scheme`, responsive (movil y escritorio).
- Cabecera con el titulo "Transcriptor y traductor RNN" y 3 pestanas:
  "Manuscrito", "Audio" y "Traduccion".

## Pestanas "Manuscrito" y "Audio" (misma estructura, tres tarjetas)
1. Dataset: input de archivo (imagen o .wav), campo de texto con la transcripcion
   exacta, boton "Agregar muestra", contador "Muestras: N (minimo 10)" y boton
   "Vaciar dataset".
2. Entrenamiento: input numerico de epocas (1-200, por defecto 10), boton "Entrenar",
   barra de progreso, texto de estado y tabla con columnas Epoca, Perdida,
   Perdida val. y Metrica val. (CER, menor es mejor).
3. Prediccion: input de archivo con vista previa (<img> o <audio controls>), boton
   "Transcribir", y un cuadro con el texto resultante, la confianza en % y un
   boton "Traducir este texto" que lleva el texto a la pestana Traduccion.

## Pestana "Traduccion"
1. Pares de idiomas: lista desplegable con los pares (nombre, cantidad de pares y
   si esta entrenado), campo para el nombre de un par nuevo (ej. "en-es"),
   input de archivo TSV (origen<TAB>destino) y botones "Cargar corpus" y "Eliminar par".
2. Entrenamiento: igual que arriba (tabla con Epoca, Perdida, Perdida val. y Precision val.).
3. Traduccion: textarea de entrada, boton "Traducir" y cuadro de salida.

## Comportamiento y API (fetch al mismo origen)
- GET  /api/overview -> {handwriting:{samples,min_samples,trained}, audio:{...}, translation:[{name,pairs,trained}]}
- POST /api/handwriting/samples (FormData: image, text) | DELETE /api/handwriting/samples
- POST /api/handwriting/predict (FormData: image) -> {text, confidence}
- POST /api/audio/samples (FormData: audio, text) | DELETE /api/audio/samples
- POST /api/audio/predict (FormData: audio) -> {text, confidence}
- POST /api/translation/corpus (FormData: file, pair) -> {added} | DELETE /api/translation/<par>
- POST /api/train/<handwriting|audio|translation> (JSON {epochs, pair}) -> {ok}
- GET  /api/status -> {status: idle|training|done|error, task, epoch, epochs, progress, history:[{epoch,loss,val_loss,val_cer|val_accuracy}], error}
- POST /api/translate (JSON {pair, text}) -> {text}
- Los errores llegan como {error: "mensaje"} con codigo HTTP != 200: mostrarlos en rojo.
- Consultar /api/status cada segundo; mientras status sea "training", deshabilitar todos
  los botones "Entrenar". Al pasar a "done", refrescar /api/overview.
- Mostrar texto con textContent (nunca innerHTML con datos del servidor).