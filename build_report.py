"""
Genera el informe final en PDF: un reporte de lo que efectivamente se hizo
(no una guia de replicacion), con el codigo real usado, los resultados
reales obtenidos y una captura de pantalla de la corrida.

Requiere que ya se haya corrido generate_report_assets.py (genera las
imagenes en report_assets/).

Uso:
    python build_report.py
"""
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted,
)

OUT_PDF = "INFORME_FINAL.pdf"
SCREENSHOT = "cap-resultado.jpeg"  # captura real de la terminal, corrida completa

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name="Note", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.grey, spaceAfter=10))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="CodeStyle", fontName="Courier", fontSize=7.6, leading=9.6))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, alignment=1, spaceAfter=12))

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
])

CODE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
])

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def p(text):
    story.append(Paragraph(text, styles["Body"]))


def note(text):
    story.append(Paragraph(text, styles["Note"]))


def code(text, filename=None):
    if filename:
        story.append(Paragraph(f"<font face='Courier' size=8><b>{filename}</b></font>", styles["Body"]))
    block = Preformatted(text.strip("\n"), styles["CodeStyle"])
    t = Table([[block]], colWidths=[16.5 * cm])
    t.setStyle(CODE_STYLE)
    story.append(t)
    story.append(Spacer(1, 10))


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 10))


def image(path, width=15 * cm, caption=None):
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Spacer(1, 10))


# ---------------------------------------------------------------------------
# Portada
# ---------------------------------------------------------------------------
story.append(Paragraph(
    "Taller — Entrenamiento Distribuido de Datos con Fashion-MNIST<br/>(Informe Final)",
    styles["Title"],
))
note(
    "Este documento reporta el trabajo efectivamente realizado: el procedimiento seguido, el código "
    "usado y los resultados obtenidos. El entrenamiento se ejecutó en <b>3 PCs físicas reales</b> "
    "(ia-2, ia-5 y una tercera máquina) conectadas por red LAN — no es una simulación. Las métricas "
    "provienen de <b>train_worker0.log</b> (máquina chief, ia-2, IP 172.16.79.158) y de la "
    "reevaluación directa del modelo entrenado sobre el conjunto de test."
)

# ---------------------------------------------------------------------------
# 1. Finalidad del modelo
# ---------------------------------------------------------------------------
h1("1. Finalidad del modelo entrenado")
p(
    "El modelo resultante es un <b>clasificador de imágenes de prendas de ropa</b>: recibe una "
    "imagen de 28x28 píxeles en escala de grises y predice a cuál de 10 categorías pertenece "
    "(camiseta/top, pantalón, suéter, vestido, abrigo, sandalia, camisa, zapatilla, bolso o bota). "
    "Es el tipo de tarea de base para sistemas reales de <b>catalogación automática de productos</b> "
    "en comercio electrónico, <b>clasificación de inventario</b> en un almacén o tienda, o como "
    "componente de entrada de un sistema más grande de recomendación o búsqueda visual de ropa."
)
p(
    "Más allá del modelo en sí, la finalidad del ejercicio era <b>práctica y pedagógica</b>: usar "
    "un problema de clasificación de imágenes lo bastante simple como para entrenar en segundos "
    "por época, y así poder concentrar el esfuerzo en entender y poner en marcha un "
    "<b>entrenamiento distribuido real</b> entre 3 computadoras — la coordinación de red, la "
    "sincronización de gradientes y los problemas de infraestructura que aparecen al hacerlo, más "
    "que la dificultad del problema de machine learning en sí."
)

# ---------------------------------------------------------------------------
# 2. Arquitectura del entrenamiento distribuido
# ---------------------------------------------------------------------------
h1("2. Arquitectura del entrenamiento distribuido")
p(
    "Se usó <b>paralelismo de datos síncrono</b> mediante "
    "<font face='Courier'>tf.distribute.MultiWorkerMirroredStrategy</font> de TensorFlow. Cada una "
    "de las 3 máquinas mantiene una <b>copia idéntica</b> del modelo. En cada paso de "
    "entrenamiento, el lote global se reparte entre las 3; cada máquina calcula gradientes sobre su "
    "porción y los sincroniza con las otras 2 mediante una operación colectiva de promediado "
    "(<i>all-reduce</i> en anillo, sobre gRPC), antes de que cualquiera actualice sus pesos."
)
p(
    "A diferencia de un esquema de <b>aprendizaje federado</b> (FedAvg), aquí no hay particiones "
    "de datos exclusivas por máquina ni una agregación periódica de modelos locales entrenados por "
    "separado: las 3 máquinas comparten el mismo conjunto de entrenamiento completo, y no existen "
    "“modelos locales” en ningún momento — la sincronización ocurre en cada paso, no al final de "
    "cada ronda."
)

# ---------------------------------------------------------------------------
# 3. Preparacion de los datos
# ---------------------------------------------------------------------------
h1("3. Preparación de los datos")
p(
    "Se usó el dataset Fashion-MNIST (70,000 imágenes de 28x28 px, 10 clases de prendas), "
    "incluido en <font face='Courier'>keras.datasets</font>. Se escribió "
    "<font face='Courier'>data_prep.py</font> para descargarlo, normalizar los píxeles a [0,1] y "
    "separarlo en <b>50,000 train / 10,000 validación / 10,000 test</b>, con una semilla fija (42) "
    "para que el split fuera idéntico en las 3 máquinas."
)
code(
    """def prep(x):
    x = x.astype("float32") / 255.0
    return np.expand_dims(x, axis=-1)   # (28, 28) -> (28, 28, 1)

x_train_full = prep(x_train_full)
x_test = prep(x_test)

# Semilla fija: si cada maquina genera el split por su cuenta, obtiene
# exactamente los mismos 50.000/10.000 ejemplos de train/val.
rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(x_train_full))
val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]

x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]""",
    filename="data_prep.py (fragmento)",
)
p(
    "Los 3 conjuntos resultantes se guardaron como archivos <font face='Courier'>.npz</font> en la "
    "carpeta <font face='Courier'>data/</font>, que cada una de las 3 máquinas tuvo disponible "
    "localmente antes de entrenar."
)

# ---------------------------------------------------------------------------
# 4. Arquitectura del modelo
# ---------------------------------------------------------------------------
h1("4. Arquitectura del modelo")
p(
    "El modelo (definido en <font face='Courier'>model.py</font>) es una red convolucional pequeña, "
    "con 225,034 parámetros — deliberadamente simple, para que cada época entrene en segundos y el "
    "foco del ejercicio pudiera estar en la coordinación distribuida:"
)
code(
    """def build_model(input_shape=(28, 28, 1), num_classes=10, learning_rate=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model""",
    filename="model.py",
)

# ---------------------------------------------------------------------------
# 5. Procedimiento de entrenamiento distribuido
# ---------------------------------------------------------------------------
h1("5. Procedimiento de entrenamiento distribuido")
p(
    "El script <font face='Courier'>train.py</font> es el mismo en las 3 máquinas; lo único que "
    "cambia entre ellas es la variable de entorno <font face='Courier'>TF_CONFIG</font>, que le "
    "indica a cada proceso su índice (0, 1 o 2) y las direcciones de las otras 2."
)
h2("5.1 Forzar CPU y comunicación RING")
p(
    "El laboratorio tiene hardware heterogéneo (algunas máquinas con GPU, otras sin). "
    "<font face='Courier'>MultiWorkerMirroredStrategy</font> elige automáticamente NCCL en cuanto "
    "detecta una GPU en el cluster, pero NCCL no funciona con workers que solo tienen CPU — esto "
    "causó fallas de sincronización durante las pruebas (ver sección 6.1). Se resolvió forzando CPU "
    "y el método de comunicación RING explícitamente en las 3 máquinas:"
)
code(
    """os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
...
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.RING
)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options
)""",
    filename="train.py (fragmento)",
)
h2("5.2 Escalado de batch y learning rate")
p(
    "Con 3 máquinas sincronizando, el lote global es 3 veces más grande que el de una sola máquina "
    "(64 por máquina &times; 3 = 192). Se aplicó la regla de escalado lineal, multiplicando también "
    "el learning rate base por 3 (1e-3 &rarr; 3e-3), para mantener una magnitud de actualización de "
    "pesos comparable a la de un entrenamiento en una sola máquina:"
)
code(
    """global_batch_size = args.per_worker_batch_size * num_workers   # 64 * 3 = 192
scaled_lr = args.base_learning_rate * num_workers              # 1e-3 * 3 = 3e-3""",
    filename="train.py (fragmento)",
)
h2("5.3 Entrenamiento y guardado solo por el chief")
p(
    "Se pidieron 10 épocas, con <font face='Courier'>EarlyStopping(patience=3, "
    "restore_best_weights=True)</font> monitoreando la pérdida de validación. "
    "<font face='Courier'>model.evaluate</font> y <font face='Courier'>model.save</font> son "
    "operaciones colectivas — las 3 máquinas deben llamarlas — pero solo la designada como "
    "<b>chief</b> (índice 0) conserva el resultado final, para evitar que las 3 compitan por "
    "escribir el mismo archivo:"
)
code(
    """model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, ...)

# evaluate/save son colectivas: las 3 maquinas las llaman, solo el chief
# se queda con el resultado/archivo final.
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
save_path = write_filepath("fashion_mnist_distribuido.keras", task_type, task_id)
model.save(save_path)

if is_chief(task_type, task_id):
    print(f"Accuracy en test: {test_acc:.4f}  |  loss en test: {test_loss:.4f}")""",
    filename="train.py (fragmento)",
)

# ---------------------------------------------------------------------------
# 6. Ejecucion en las 3 maquinas reales
# ---------------------------------------------------------------------------
h1("6. Ejecución en las 3 máquinas reales")
table(
    [
        ["Nodo", "IP", "Rol"],
        ["Worker 0 (chief)", "172.16.79.158 (ia-2)", "Coordina, evalúa test final, guarda el modelo"],
        ["Worker 1", "172.16.79.159 (ia-5)", "Sincroniza gradientes"],
        ["Worker 2", "172.16.79.120", "Sincroniza gradientes"],
    ],
    col_widths=[3.5 * cm, 5 * cm, 8 * cm],
)
p(
    "Para lanzar el entrenamiento en las 3 máquinas sin tener que escribir a mano el JSON de "
    "<font face='Courier'>TF_CONFIG</font> en cada terminal SSH (fuente de errores de tipeo durante "
    "las pruebas), se usó <font face='Courier'>run_worker.sh</font>, que arma el "
    "<font face='Courier'>TF_CONFIG</font> a partir de 3 IPs fijas y el índice recibido como "
    "argumento:"
)
code(
    """IP_0="172.16.79.158"; IP_1="172.16.79.159"; IP_2="172.16.79.120"; PORT=12345
INDEX="$1"

export TF_CONFIG=$(python - "$INDEX" "$IP_0" "$IP_1" "$IP_2" "$PORT" <<'PYEOF'
import json, sys
index = int(sys.argv[1])
workers = [f"{sys.argv[i]}:{sys.argv[5]}" for i in (2, 3, 4)]
print(json.dumps({"cluster": {"worker": workers}, "task": {"type": "worker", "index": index}}))
PYEOF
)
python train.py --epochs "$EPOCHS" 2>&1 | tee "train_worker${INDEX}.log\"""",
    filename="run_worker.sh (fragmento)",
)
p(
    "Se ejecutó <font face='Courier'>bash run_worker.sh 0 10</font>, "
    "<font face='Courier'>bash run_worker.sh 1 10</font> y "
    "<font face='Courier'>bash run_worker.sh 2 10</font> en las 3 máquinas, casi al mismo tiempo."
)
h2("6.1 Desafíos resueltos durante la puesta en marcha")
p(
    "• <b>Versiones de TensorFlow distintas entre máquinas</b> causaban errores de protocolo al "
    "sincronizar gradientes: se fijaron versiones exactas (TensorFlow 2.21.0) en "
    "<font face='Courier'>requirements.txt</font>, iguales en las 3.<br/>"
    "• <b>Hardware heterogéneo</b> (GPU en algunas máquinas, no en otras) causaba errores de tipo "
    "<font face='Courier'>Aborting RingReduce ... unknown device</font> al elegir TensorFlow "
    "automáticamente NCCL: se forzó CPU y comunicación RING en las 3 (sección 5.1).<br/>"
    "• <b>Reinicios parciales del cluster</b>: si solo 1 o 2 máquinas se reiniciaban (por ejemplo, "
    "para corregir el entorno virtual) mientras las demás seguían corriendo desde un intento "
    "anterior, el grupo de coordinación quedaba desincronizado y el entrenamiento fallaba a mitad "
    "de camino. Se resolvió reiniciando siempre las 3 máquinas juntas, desde cero."
)

# ---------------------------------------------------------------------------
# 7. Resultados obtenidos
# ---------------------------------------------------------------------------
h1("7. Resultados obtenidos")
p(
    "A continuación, la captura real de la terminal de la máquina chief (ia-2) al finalizar la "
    "corrida, seguida del detalle época por época y el análisis de los resultados."
)
if os.path.exists(SCREENSHOT):
    image(SCREENSHOT, caption="Captura real de la terminal (ia-2) al finalizar el entrenamiento.")

metric_rows = [["Época", "Pérdida train", "Exactitud train", "Pérdida val", "Exactitud val"]]
epochs = list(range(1, 9))
train_loss = [0.5671, 0.3569, 0.3083, 0.2782, 0.2519, 0.2359, 0.2185, 0.2053]
train_acc = [0.7922, 0.8711, 0.8879, 0.8971, 0.9068, 0.9123, 0.9186, 0.9229]
val_loss = [0.3550, 0.3009, 0.2801, 0.2784, 0.2505, 0.2658, 0.2570, 0.2548]
val_acc = [0.8667, 0.8841, 0.8934, 0.8909, 0.9052, 0.8968, 0.9010, 0.9030]
for e, tl, ta, vl, va in zip(epochs, train_loss, train_acc, val_loss, val_acc):
    marker = "  (mejor)" if e == 5 else ""
    metric_rows.append([str(e), f"{tl:.4f}", f"{ta*100:.2f} %", f"{vl:.4f}{marker}", f"{va*100:.2f} %"])
table(metric_rows, col_widths=[1.8 * cm, 3.5 * cm, 3.7 * cm, 4.5 * cm, 3.5 * cm])

h2("7.1 Lectura detallada por tramos")
p(
    "<b>Épocas 1-2:</b> mejora rápida y pronunciada (exactitud de validación de 86.67% a 88.41%, "
    "pérdida de 0.355 a 0.301), típica de la salida de la inicialización aleatoria de los pesos: el "
    "modelo pasa de no saber nada a capturar los patrones más evidentes del dataset."
)
p(
    "<b>Épocas 3-5:</b> la mejora continúa pero de forma más gradual; la exactitud de validación "
    "sube de 89.34% a 90.52% y alcanza en la <b>época 5 su mejor punto</b> (pérdida de validación "
    "mínima: 0.2505). Hasta aquí, pérdida de entrenamiento y de validación bajan juntas, sin señales "
    "de sobreajuste."
)
p(
    "<b>Épocas 6-8:</b> la pérdida de entrenamiento sigue bajando (de 0.236 a 0.205) y la exactitud "
    "de entrenamiento sigue subiendo (91.2% a 92.3%), pero la pérdida de validación deja de mejorar "
    "y se mantiene por encima del valor de la época 5 (0.266, 0.257, 0.255) — el modelo empieza a "
    "memorizar detalles del set de entrenamiento en vez de generalizar (sobreajuste incipiente). "
    "Al completarse 3 épocas seguidas sin mejorar la pérdida de validación, "
    "<font face='Courier'>EarlyStopping</font> detuvo el entrenamiento en la época 8 (de las 10 "
    "solicitadas) y <font face='Courier'>restore_best_weights=True</font> revirtió el modelo a los "
    "pesos de la época 5 antes de continuar."
)
image("report_assets/accuracy_por_epoca.png")
image("report_assets/loss_por_epoca.png")

h2("7.2 Evaluación final sobre el conjunto de test")
p(
    "Con los pesos de la época 5 restaurados, se evaluó el modelo sobre las 10,000 imágenes de test "
    "(nunca usadas durante el entrenamiento ni la validación): <b>pérdida en test = 0.2756</b>, "
    "<b>exactitud en test = 89.97%</b>. Este resultado es consistente con las métricas de "
    "validación de la época 5 (pérdida 0.2505, exactitud 90.52%), confirmando que el modelo "
    "generaliza razonablemente bien a datos que nunca vio."
)
p(
    "Para ilustrar el comportamiento del modelo caso por caso, se tomó una muestra de 10 imágenes "
    "de test y se comparó la predicción contra la etiqueta real: <b>9 de 10 correctas</b>. El único "
    "error fue un <i>Abrigo</i> clasificado como <i>Suéter</i> — una confusión razonable dada la "
    "similitud visual entre ambas prendas en una imagen de solo 28x28 píxeles en escala de grises."
)
image("report_assets/predicciones_ejemplo.png")

# ---------------------------------------------------------------------------
# 8. Conformacion del modelo final
# ---------------------------------------------------------------------------
h1("8. Conformación del modelo final")
p(
    "A diferencia del aprendizaje federado (donde cada nodo entrena un modelo local completo y "
    "estos se promedian periódicamente con FedAvg), en el paralelismo de datos síncrono usado aquí "
    "<b>no existieron modelos locales independientes en ningún momento</b>: los gradientes se "
    "promediaron (all-reduce) antes de que cualquier copia actualizara sus pesos, por lo que las 3 "
    "máquinas mantuvieron siempre una copia idéntica del mismo modelo durante todo el "
    "entrenamiento. No hubo, por lo tanto, un paso de “agregación final”: el modelo final es "
    "simplemente el estado del modelo (idéntico en las 3 máquinas) al terminar el entrenamiento. "
    "Por convención, solo la máquina chief (ia-2) escribió el archivo final en disco "
    "(<font face='Courier'>fashion_mnist_distribuido.keras</font>, 225,034 parámetros) y reportó "
    "las métricas de test."
)
note("Repositorio del proyecto (código fuente completo): github.com/Camilo1408/taller-aprendizaje-distribuido")

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=letter,
    topMargin=1.6 * cm, bottomMargin=1.6 * cm, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
)
doc.build(story)
print(f"Informe generado: {OUT_PDF}")
