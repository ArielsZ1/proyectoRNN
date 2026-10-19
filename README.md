# Transcriptor y traductor con redes recurrentes

Aplicacion web local para experimentar con tres tareas de procesamiento de secuencias:

- **Reconocimiento de escritura manuscrita:** convierte una imagen con una palabra o linea corta en texto.
- **Reconocimiento de voz:** transcribe clips de audio WAV a texto.
- **Traduccion:** aprende a traducir frases entre dos idiomas a partir de un corpus paralelo.

La interfaz permite cargar ejemplos etiquetados, entrenar los modelos, consultar el progreso y probar predicciones. Los datos y modelos entrenados se guardan localmente en `data/` y `models/`.

## Que es una RNN

Una red neuronal recurrente (RNN) procesa una secuencia paso a paso y mantiene un estado que resume el contexto anterior. Esto resulta util cuando el orden importa, por ejemplo, al leer caracteres de izquierda a derecha, interpretar una señal de audio en el tiempo o generar una traduccion palabra por palabra.

En este proyecto, las arquitecturas recurrentes se combinan con otras capas segun la tarea:

- **Manuscrito:** capas convolucionales extraen patrones visuales; dos capas LSTM bidireccionales procesan la imagen como una secuencia horizontal. La salida usa CTC (Connectionist Temporal Classification), que permite entrenar sin indicar la posicion exacta de cada caracter.
- **Audio:** capas convolucionales procesan espectrogramas Mel y dos capas GRU bidireccionales modelan la secuencia temporal. CTC convierte las salidas en caracteres.
- **Traduccion:** un GRU codificador resume la frase de origen y un GRU decodificador genera la frase destino token por token. Es un modelo encoder-decoder sin mecanismo de atencion.

Por lo tanto, el proyecto no es una unica RNN: combina CNN, LSTM, GRU y CTC en modelos distintos. Son modelos didacticos que se entrenan con los datos suministrados; su calidad depende mucho del tamano y la variedad del dataset.

## Requisitos

- Python 3.9 a 3.12.
- `pip` y un entorno virtual recomendado.
- Para entrenamiento, memoria suficiente para TensorFlow. El uso de GPU es opcional.

## Instalacion y ejecucion

Desde una terminal, entra en la carpeta `proyectoRNN`:

```powershell
cd proyectoRNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

En Windows, si PowerShell impide activar el entorno virtual, puedes ejecutarlo sin activacion explicita:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Abre [http://127.0.0.1:5001](http://127.0.0.1:5001) en el navegador. El servidor escucha solo en la maquina local.

## Uso

### Manuscrito

1. En la pestana **Manuscrito**, agrega imagenes y escribe la transcripcion exacta.
2. Reune al menos 10 muestras validas y pulsa **Entrenar**. Puedes elegir entre 1 y 200 epocas.
3. Cuando termine el entrenamiento, carga una imagen para transcribirla.

Se admiten imagenes JPEG, PNG, BMP, GIF y WebP. El modelo normaliza la imagen a escala de grises y admite etiquetas de hasta 32 caracteres. Para que generalice, usa ejemplos variados y representativos de la escritura que quieras reconocer.

Hay un generador de muestras sinteticas basado en fuentes instaladas en Windows:

```powershell
python generar_dataset.py manuscrito --n 1500
```

Para indicar fuentes adicionales `.ttf` o `.otf`:

```powershell
python generar_dataset.py manuscrito --n 1500 --fuente C:\ruta\a\fuentes
```

Las muestras generadas se agregan a `data/handwriting/`. La escritura sintetica puede ser distinta de la escritura real y no garantiza buenas predicciones sobre manuscritos nuevos.

### Audio

1. Agrega clips WAV junto con su transcripcion exacta.
2. Usa al menos 10 clips validos y pulsa **Entrenar**.
3. Carga un WAV para obtener la transcripcion y una confianza aproximada.

El formato requerido es WAV PCM de 16 bits, 16 kHz y mono o multicanal (los canales se mezclan a mono). Cada clip de entrenamiento debe durar como maximo aproximadamente 8 segundos. El proyecto no incluye un dataset inicial de audio: debes agregar muestras propias desde la interfaz.

### Traduccion

La interfaz trabaja con un nombre por par de idiomas, por ejemplo `en-es`. Carga un archivo TSV codificado en UTF-8 con una frase de origen y su traduccion por linea:

```text
Hello.\tHola.
How are you?\t¿Como estas?
```

Puedes generar un corpus de ejemplo ingles-espanol y su inversion:

```powershell
python generar_dataset.py traduccion
python generar_dataset.py traduccion --invertir
```

El generador crea `datasets/corpus_en-es.tsv` o `datasets/corpus_es-en.tsv`. En la pestana **Traduccion**, selecciona el nombre del par (`en-es` o `es-en`) y carga el archivo correspondiente. Se requieren al menos 50 pares validos para entrenar; el modelo usa hasta 30.000 pares y secuencias de hasta 20 tokens. Despues de entrenar, escribe un texto y pulsa **Traducir**.

El corpus de ejemplo es limitado y sirve para probar el flujo, no para obtener traducciones de calidad general. La carpeta `models/translation/es-en/` puede contener un modelo y vocabulario ya entrenados; el resto de modelos se crea al entrenar desde la interfaz.

## Datos y modelos

- `data/handwriting/`: imagenes y etiquetas de manuscrito.
- `data/audio/`: clips WAV y transcripciones, creados al agregar muestras.
- `data/translation/<par>/pairs.tsv`: corpus por par de idiomas.
- `datasets/`: corpus de ejemplo generado para traduccion.
- `models/handwriting/`, `models/audio/` y `models/translation/<par>/`: modelos y vocabularios guardados tras entrenar.

La interfaz incluye acciones para borrar los datos de manuscrito/audio y para eliminar un par de traduccion junto con su modelo. Esas acciones son destructivas y no tienen deshacer.

## Metricas

Durante el entrenamiento se muestra la perdida de entrenamiento y validacion. Para manuscrito y audio tambien se calcula CER (Character Error Rate): menor es mejor. En traduccion se muestra la precision por token. La confianza que se muestra al transcribir es una estimacion aproximada, no una probabilidad calibrada.

## API local

La interfaz se comunica con Flask en el mismo origen. Las rutas principales son:

- `GET /api/overview` y `GET /api/status`: estado de datasets, modelos y entrenamiento.
- `POST /api/train/<handwriting|audio|translation>`: iniciar entrenamiento.
- `POST /api/handwriting/samples` y `POST /api/audio/samples`: agregar ejemplos etiquetados.
- `POST /api/handwriting/predict` y `POST /api/audio/predict`: transcribir una imagen o un WAV.
- `POST /api/translation/corpus` y `POST /api/translate`: cargar corpus y traducir texto.

La aplicacion limita las cargas HTTP a 200 MB. Para el formato de cada solicitud, consulta las rutas en `app.py`.
