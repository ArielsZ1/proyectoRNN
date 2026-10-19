"""Generador de datasets de ejemplo para proyectoRNN.

Uso (desde la carpeta IA):
    python proyectoRNN/generar_datasets.py traduccion
    python proyectoRNN/generar_datasets.py traduccion --invertir
    python proyectoRNN/generar_datasets.py manuscrito --n 1500
    python proyectoRNN/generar_datasets.py manuscrito --n 1500 --fuente C:\\ruta\\fuente.ttf
"""
import argparse
import os
import random
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "datasets")
HW_DIR = os.path.join(BASE_DIR, "data", "handwriting")  # misma ruta que handwriting.py
HW_IMAGES = os.path.join(HW_DIR, "images")
HW_LABELS = os.path.join(HW_DIR, "labels.tsv")

# =====================================================================
# TRADUCCION (ingles -> espanol)
# =====================================================================
FRASES = [
    ("Hello.", "Hola."), ("Good morning.", "Buenos días."),
    ("Good afternoon.", "Buenas tardes."), ("Good night.", "Buenas noches."),
    ("Thank you.", "Gracias."), ("Thank you very much.", "Muchas gracias."),
    ("You are welcome.", "De nada."), ("Please.", "Por favor."),
    ("Excuse me.", "Disculpe."), ("I am sorry.", "Lo siento."),
    ("See you later.", "Hasta luego."), ("See you tomorrow.", "Hasta mañana."),
    ("Goodbye.", "Adiós."), ("How are you?", "¿Cómo estás?"),
    ("I am fine.", "Estoy bien."), ("What is your name?", "¿Cómo te llamas?"),
    ("My name is Ana.", "Mi nombre es Ana."), ("Nice to meet you.", "Mucho gusto."),
    ("Where is the bathroom?", "¿Dónde está el baño?"),
    ("How much does it cost?", "¿Cuánto cuesta?"),
    ("I do not understand.", "No entiendo."),
    ("Do you speak English?", "¿Hablas inglés?"),
    ("I speak a little Spanish.", "Hablo un poco de español."),
    ("Can you help me?", "¿Puedes ayudarme?"), ("I need help.", "Necesito ayuda."),
    ("I am hungry.", "Tengo hambre."), ("I am thirsty.", "Tengo sed."),
    ("I am tired.", "Estoy cansado."), ("I am happy.", "Estoy feliz."),
    ("I am from Argentina.", "Soy de Argentina."),
    ("I live in a small city.", "Vivo en una ciudad pequeña."),
    ("What time is it?", "¿Qué hora es?"), ("It is late.", "Es tarde."),
    ("It is early.", "Es temprano."), ("Today is Monday.", "Hoy es lunes."),
    ("Tomorrow is Tuesday.", "Mañana es martes."),
    ("The weather is nice today.", "Hace buen tiempo hoy."),
    ("It is cold.", "Hace frío."), ("It is hot.", "Hace calor."),
    ("It is raining.", "Está lloviendo."), ("I like music.", "Me gusta la música."),
    ("I like to read.", "Me gusta leer."),
    ("She likes to dance.", "A ella le gusta bailar."),
    ("We like this city.", "Nos gusta esta ciudad."),
    ("Where do you live?", "¿Dónde vives?"), ("Where are you from?", "¿De dónde eres?"),
    ("What do you do?", "¿A qué te dedicas?"),
    ("I work in an office.", "Trabajo en una oficina."),
    ("I study at school.", "Estudio en la escuela."),
    ("The book is on the table.", "El libro está sobre la mesa."),
    ("The cat is under the chair.", "El gato está debajo de la silla."),
    ("The dog is in the garden.", "El perro está en el jardín."),
    ("The coffee is hot.", "El café está caliente."),
    ("The water is cold.", "El agua está fría."),
    ("The food is delicious.", "La comida es deliciosa."),
    ("The sea is blue.", "El mar es azul."), ("The sky is clear.", "El cielo está despejado."),
    ("The moon is beautiful.", "La luna es hermosa."),
    ("My mother is a doctor.", "Mi madre es médica."),
    ("My father works a lot.", "Mi padre trabaja mucho."),
    ("My brother is tall.", "Mi hermano es alto."),
    ("My sister is very kind.", "Mi hermana es muy amable."),
    ("We are friends.", "Somos amigos."), ("They are students.", "Ellos son estudiantes."),
    ("He is a teacher.", "Él es profesor."), ("She is a nurse.", "Ella es enfermera."),
    ("Where is the station?", "¿Dónde está la estación?"),
    ("I want to go home.", "Quiero ir a casa."),
    ("I want to learn Spanish.", "Quiero aprender español."),
    ("We are going to the beach.", "Vamos a la playa."), ("Let us go.", "Vamos."),
    ("Wait a moment.", "Espera un momento."), ("Open the door.", "Abre la puerta."),
    ("Close the window.", "Cierra la ventana."), ("Come here.", "Ven aquí."),
    ("Sit down, please.", "Siéntate, por favor."),
    ("I am learning to program.", "Estoy aprendiendo a programar."),
    ("I like programming.", "Me gusta programar."),
    ("The computer is on the table.", "La computadora está sobre la mesa."),
    ("My phone is new.", "Mi teléfono es nuevo."),
    ("Happy birthday!", "¡Feliz cumpleaños!"), ("Happy new year!", "¡Feliz año nuevo!"),
    ("Good luck!", "¡Buena suerte!"), ("Welcome!", "¡Bienvenido!"),
    ("Have a nice trip.", "Buen viaje."), ("I love you.", "Te quiero."),
    ("What is this?", "¿Qué es esto?"), ("Who is he?", "¿Quién es él?"),
    ("Why are you sad?", "¿Por qué estás triste?"),
    ("When does the train leave?", "¿Cuándo sale el tren?"),
    ("How old are you?", "¿Cuántos años tienes?"),
    ("I am twenty years old.", "Tengo veinte años."),
    ("The bus is late.", "El autobús llega tarde."),
    ("I am going to work.", "Voy al trabajo."), ("We eat at noon.", "Comemos al mediodía."),
    ("I drink water every day.", "Bebo agua todos los días."),
    ("I read a book every night.", "Leo un libro cada noche."),
    ("The children play in the park.", "Los niños juegan en el parque."),
    ("The birds sing in the morning.", "Los pájaros cantan por la mañana."),
    ("The store is closed.", "La tienda está cerrada."),
    ("The store is open.", "La tienda está abierta."),
    ("She is very intelligent.", "Ella es muy inteligente."),
    ("He is a good friend.", "Él es un buen amigo."),
    ("Here is my phone number.", "Aquí está mi número de teléfono."),
    ("I am ready.", "Estoy listo."), ("It is important.", "Es importante."),
    ("It is possible.", "Es posible."), ("I do not know.", "No sé."),
    ("Yes, of course.", "Sí, claro."), ("No, thank you.", "No, gracias."),
]

# (pronombre en, pronombre es, indice de conjugacion)
SUJETOS = [("I", "Yo", 0), ("You", "Tú", 1), ("He", "Él", 2),
           ("She", "Ella", 2), ("We", "Nosotros", 3), ("They", "Ellos", 4)]

# verbo en (base), verbo en (3a persona), [conjugaciones es], [(objeto en, objeto es)]
VERBOS = [
    ("eat", "eats", ["como", "comes", "come", "comemos", "comen"], [
        ("an apple", "una manzana"), ("bread", "pan"), ("a sandwich", "un sándwich"),
        ("fish", "pescado"), ("rice", "arroz"), ("a pizza", "una pizza"),
        ("fruit", "fruta"), ("soup", "sopa")]),
    ("drink", "drinks", ["bebo", "bebes", "bebe", "bebemos", "beben"], [
        ("water", "agua"), ("coffee", "café"), ("milk", "leche"), ("tea", "té"),
        ("juice", "jugo")]),
    ("read", "reads", ["leo", "lees", "lee", "leemos", "leen"], [
        ("a book", "un libro"), ("the newspaper", "el periódico"),
        ("a letter", "una carta"), ("a magazine", "una revista"),
        ("the message", "el mensaje")]),
    ("buy", "buys", ["compro", "compras", "compra", "compramos", "compran"], [
        ("a car", "un coche"), ("a house", "una casa"), ("a ticket", "un billete"),
        ("a gift", "un regalo"), ("new shoes", "zapatos nuevos"), ("bread", "pan"),
        ("a computer", "una computadora")]),
    ("see", "sees", ["veo", "ves", "ve", "vemos", "ven"], [
        ("a dog", "un perro"), ("a cat", "un gato"), ("the sea", "el mar"),
        ("a bird", "un pájaro"), ("the mountain", "la montaña"), ("the moon", "la luna")]),
    ("have", "has", ["tengo", "tienes", "tiene", "tenemos", "tienen"], [
        ("a car", "un coche"), ("a dog", "un perro"), ("a house", "una casa"),
        ("two brothers", "dos hermanos"), ("a good friend", "un buen amigo"),
        ("a question", "una pregunta")]),
    ("want", "wants", ["quiero", "quieres", "quiere", "queremos", "quieren"], [
        ("a coffee", "un café"), ("some water", "un poco de agua"),
        ("a new phone", "un teléfono nuevo"), ("more time", "más tiempo"),
        ("a glass of milk", "un vaso de leche")]),
    ("love", "loves", ["amo", "amas", "ama", "amamos", "aman"], [
        ("music", "la música"), ("this city", "esta ciudad"),
        ("the summer", "el verano"), ("chocolate", "el chocolate")]),
    ("need", "needs", ["necesito", "necesitas", "necesita", "necesitamos", "necesitan"], [
        ("help", "ayuda"), ("a doctor", "un médico"), ("more time", "más tiempo"),
        ("a new phone", "un teléfono nuevo"), ("water", "agua")]),
    ("find", "finds", ["encuentro", "encuentras", "encuentra", "encontramos", "encuentran"], [
        ("the keys", "las llaves"), ("a job", "un trabajo"), ("the book", "el libro"),
        ("my phone", "mi teléfono")]),
    ("write", "writes", ["escribo", "escribes", "escribe", "escribimos", "escriben"], [
        ("a letter", "una carta"), ("a message", "un mensaje"), ("a book", "un libro"),
        ("an email", "un correo")]),
]

SUSTANTIVOS = [("house", "casa", "f"), ("car", "coche", "m"), ("dog", "perro", "m"),
               ("cat", "gato", "m"), ("book", "libro", "m"), ("table", "mesa", "f"),
               ("city", "ciudad", "f"), ("garden", "jardín", "m"),
               ("door", "puerta", "f"), ("window", "ventana", "f")]
ADJETIVOS = [("big", "grande", "grande"), ("small", "pequeño", "pequeña"),
             ("new", "nuevo", "nueva"), ("old", "viejo", "vieja"),
             ("beautiful", "hermoso", "hermosa"), ("red", "rojo", "roja"),
             ("white", "blanco", "blanca"), ("clean", "limpio", "limpia")]


def generar_traduccion(invertir):
    pares = list(FRASES)
    for en_base, en_3, es_conj, objetos in VERBOS:
        for en_p, es_p, k in SUJETOS:
            en_v = en_3 if k == 2 else en_base
            aux = "does" if k == 2 else "do"
            for en_o, es_o in objetos:
                pares.append((f"{en_p} {en_v} {en_o}.", f"{es_p} {es_conj[k]} {es_o}."))
                pares.append((f"{en_p} {aux} not {en_base} {en_o}.",
                              f"{es_p} no {es_conj[k]} {es_o}."))
    for en_n, es_n, g in SUSTANTIVOS:
        for en_a, es_am, es_af in ADJETIVOS:
            adj = es_am if g == "m" else es_af
            pares.append((f"The {en_n} is {en_a}.",
                          f"{'El' if g == 'm' else 'La'} {es_n} es {adj}."))
            pares.append((f"This {en_n} is {en_a}.",
                          f"{'Este' if g == 'm' else 'Esta'} {es_n} es {adj}."))
    pares = list(dict.fromkeys(pares))  # sin duplicados
    random.shuffle(pares)
    nombre = "es-en" if invertir else "en-es"
    os.makedirs(OUT_DIR, exist_ok=True)
    ruta = os.path.join(OUT_DIR, f"corpus_{nombre}.tsv")
    with open(ruta, "w", encoding="utf-8") as f:
        for en, es in pares:
            a, b = (es, en) if invertir else (en, es)
            f.write(f"{a}\t{b}\n")
    print(f"{len(pares)} pares guardados en {ruta}")
    print(f"En la interfaz: pestaña Traducción -> nombre del par '{nombre}' -> cargar este archivo.")


# =====================================================================
# MANUSCRITO (sintetico con fuentes tipo escritura a mano)
# =====================================================================
PALABRAS = """casa perro gato sol luna mar río árbol flor libro mesa silla agua fuego tierra
cielo nube lluvia viento nieve montaña ciudad pueblo calle escuela amigo familia madre padre
hermano niño niña mujer hombre trabajo tiempo día noche mañana tarde hoy ayer siempre nunca
rápido lento grande pequeño bueno malo nuevo viejo feliz triste fuerte claro oscuro verde rojo
azul amarillo blanco negro naranja uno dos tres cuatro cinco seis siete ocho nueve diez comer
beber correr saltar leer escribir hablar escuchar mirar pensar soñar vivir amar jugar cantar
bailar dormir estudiar aprender enseñar abrir cerrar venir volver comida pan leche café queso
fruta manzana pera uva limón tomate papa arroz carne pollo pescado ensalada teléfono
computadora ventana puerta camino bosque jardín playa isla país mundo idea historia música
canción juego regalo sorpresa esperanza libertad amistad paz pingüino cigüeña español
cumpleaños niñez montañas otoño invierno verano primavera
buenos_días buenas_noches muchas_gracias por_favor hasta_luego mucho_gusto buen_viaje
feliz_cumpleaños año_nuevo""".split()
PALABRAS = list(dict.fromkeys(p.replace("_", " ") for p in PALABRAS))

FUENTES_WINDOWS = ["segoesc.ttf", "Inkfree.ttf", "BRADHITC.TTF", "LHANDW.TTF", "comic.ttf",
                   "ITCKRIST.TTF", "FREESCPT.TTF", "SCRIPTBL.TTF", "MTCORSVA.TTF"]


def buscar_fuentes(extra):
    rutas = []
    for e in extra or []:
        if os.path.isdir(e):
            rutas += [os.path.join(e, n) for n in os.listdir(e)
                      if n.lower().endswith((".ttf", ".otf"))]
        elif os.path.isfile(e):
            rutas.append(e)
    carpeta = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for n in FUENTES_WINDOWS:
        p = os.path.join(carpeta, n)
        if os.path.isfile(p):
            rutas.append(p)
    return rutas


def dibujar(texto, fuente_ruta):
    size = random.randint(40, 56)
    font = ImageFont.truetype(fuente_ruta, size)
    anchos = [font.getlength(c) for c in texto]
    sep = [random.uniform(-2, 3) for _ in texto]
    dy = [random.randint(-3, 3) for _ in texto]
    pad = 30
    w = max(int(sum(anchos) + sum(sep)) + 2 * pad, 80)
    h = int(size * 2.4)
    fondo = random.randint(200, 255)
    tinta = random.randint(0, 80)
    img = Image.new("L", (w, h), fondo)
    d = ImageDraw.Draw(img)
    x = float(pad)
    for c, a, s, off in zip(texto, anchos, sep, dy):
        d.text((x, size * 0.45 + off), c, font=font, fill=tinta)
        x += a + s
    img = img.rotate(random.uniform(-4, 4), resample=Image.Resampling.BICUBIC,
                     expand=True, fillcolor=fondo)
    s = random.uniform(-0.25, 0.25)
    img = img.transform(img.size, Image.Transform.AFFINE,
                        (1, s, -s * img.size[1] / 2, 0, 1, 0),
                        resample=Image.Resampling.BICUBIC, fillcolor=fondo)
    arr = np.asarray(img)
    ys, xs = np.where(arr < fondo - 60)
    if len(xs) == 0:
        return None
    m = random.randint(4, 14)
    img = img.crop((max(xs.min() - m, 0), max(ys.min() - m, 0),
                    min(xs.max() + m + 1, img.size[0]), min(ys.max() + m + 1, img.size[1])))
    if img.size[0] / img.size[1] > 4.2:  # el modelo trabaja con relacion de aspecto <= ~4:1
        return None
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0, random.uniform(0, 8), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generar_manuscrito(n, fuentes_extra):
    fuentes = buscar_fuentes(fuentes_extra)
    if not fuentes:
        sys.exit("No se encontraron fuentes. Usa --fuente ruta.ttf (o una carpeta con .ttf).")
    print(f"Fuentes encontradas: {len(fuentes)}")
    os.makedirs(HW_IMAGES, exist_ok=True)
    hechos = 0
    with open(HW_LABELS, "a", encoding="utf-8") as lab:
        while hechos < n:
            texto = random.choice(PALABRAS)
            if random.random() < 0.3:
                texto = texto.capitalize()
            try:
                img = dibujar(texto, random.choice(fuentes))
            except OSError:
                continue
            if img is None:
                continue
            nombre = f"{uuid.uuid4().hex}.png"
            img.save(os.path.join(HW_IMAGES, nombre))
            lab.write(f"{nombre}\t{texto}\n")
            hechos += 1
    print(f"{hechos} imágenes agregadas en {HW_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("traduccion")
    t.add_argument("--invertir", action="store_true", help="genera es-en en lugar de en-es")
    m = sub.add_parser("manuscrito")
    m.add_argument("--n", type=int, default=1500)
    m.add_argument("--fuente", nargs="+", help="archivos .ttf/.otf o carpetas con fuentes")
    args = ap.parse_args()
    random.seed(123)
    np.random.seed(123)
    if args.cmd == "traduccion":
        generar_traduccion(args.invertir)
    else:
        generar_manuscrito(args.n, args.fuente)