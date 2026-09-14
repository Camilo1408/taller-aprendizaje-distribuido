"""
Genera el informe final en PDF a partir de los resultados reales del
entrenamiento distribuido (train_worker0.log de la maquina chief y el
modelo fashion_mnist_distribuido_fixed.keras). Requiere que ya se haya
corrido generate_report_assets.py (genera las imagenes en report_assets/).

Uso:
    python build_report.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
)

OUT_PDF = "INFORME_FINAL.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name="Note", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.grey, spaceAfter=10))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6))

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

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def p(text):
    story.append(Paragraph(text, styles["Body"]))


def note(text):
    story.append(Paragraph(text, styles["Note"]))


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TABLE_STYLE)
    story.append(t)
    story.append(Spacer(1, 10))


def image(path, width=15 * cm):
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    story.append(Spacer(1, 10))


# ---------------------------------------------------------------------------
# Portada / titulo
# ---------------------------------------------------------------------------
story.append(Paragraph(
    "Taller — Entrenamiento Distribuido de Datos con Fashion-MNIST<br/>(Informe Final)",
    styles["Title"],
))
note(
    "Todos los números provienen de <b>train_worker0.log</b> (máquina chief, "
    "ia-2, IP 172.16.79.158) y de la reevaluación directa del modelo entrenado "
    "(<b>fashion_mnist_distribuido.keras</b>) sobre el conjunto de test. El entrenamiento "
    "se ejecutó en <b>3 PCs físicas reales</b> (ia-2, ia-5 y una tercera máquina) "
    "conectadas por red LAN — no es una simulación."
)

# ---------------------------------------------------------------------------
# 1. Objetivo y arquitectura de 3 nodos
# ---------------------------------------------------------------------------
h1("1. Objetivo y arquitectura de 3 nodos")
p(
    "Entrenar una red convolucional "
    "(<font face='Courier'>Conv2D(32) &rarr; MaxPool &rarr; Conv2D(64) &rarr; MaxPool &rarr; "
    "Flatten &rarr; Dense(128, relu) &rarr; Dropout(0.3) &rarr; Dense(10, softmax)</font>, "
    "225,034 parámetros) que clasifica imágenes de Fashion-MNIST (10 categorías de "
    "prendas de ropa, 28x28 px en escala de grises) repartiendo el <b>cómputo del "
    "entrenamiento</b> entre 3 computadoras, mediante <b>paralelismo de datos síncrono</b> "
    "(<font face='Courier'>tf.distribute.MultiWorkerMirroredStrategy</font> de TensorFlow), "
    "en vez de entrenar en una sola máquina."
)
p(
    "Cada una de las 3 máquinas mantiene una <b>copia idéntica</b> del modelo. En cada "
    "paso de entrenamiento, el lote global se reparte entre las 3; cada máquina calcula "
    "gradientes sobre su porción y los sincroniza con las otras 2 mediante una operación "
    "colectiva de promediado (<i>all-reduce</i> en anillo, sobre gRPC). Se forzó "
    "explícitamente el método de comunicación <font face='Courier'>RING</font> y el uso "
    "exclusivo de CPU (<font face='Courier'>CUDA_VISIBLE_DEVICES=-1</font>) en las 3 "
    "máquinas, ya que el hardware de laboratorio es heterogéneo (algunas máquinas con "
    "GPU y otras sin) y el modo automático de TensorFlow intenta usar NCCL en cuanto "
    "detecta una GPU en el cluster — NCCL no funciona con workers que solo tienen CPU, "
    "lo cual rompe la sincronización de gradientes."
)
h2("Diferencia con aprendizaje federado")
p(
    "A diferencia de un esquema federado (FedAvg), aquí <b>no hay particiones de datos "
    "exclusivas por máquina</b> ni una agregación periódica de modelos locales. Las 3 "
    "máquinas comparten el mismo conjunto de entrenamiento completo, y TensorFlow reparte "
    "automáticamente cada lote global entre ellas en cada paso. No existen “modelos "
    "locales” en ningún momento: la sincronización de gradientes ocurre antes de cada "
    "actualización de pesos, por lo que las 3 copias del modelo permanecen matemáticamente "
    "idénticas durante todo el entrenamiento."
)

# ---------------------------------------------------------------------------
# 2. Como se entreno en 3 nodos
# ---------------------------------------------------------------------------
h1("2. Cómo se entrenó en 3 nodos")
table(
    [
        ["Nodo", "IP", "Rol", "Comando"],
        ["Worker 0 (chief)", "172.16.79.158 (ia-2)", "Coordina, evalúa test\nfinal, guarda el modelo", "bash run_worker.sh 0 10"],
        ["Worker 1", "172.16.79.159 (ia-5)", "Sincroniza gradientes", "bash run_worker.sh 1 10"],
        ["Worker 2", "172.16.79.120", "Sincroniza gradientes", "bash run_worker.sh 2 10"],
    ],
    col_widths=[3.2 * cm, 3.8 * cm, 4.5 * cm, 4.5 * cm],
)
p(
    "Dataset preparado con <font face='Courier'>data_prep.py</font>: Fashion-MNIST "
    "normalizado a [0,1] y separado en <b>50,000 train / 10,000 validación / 10,000 "
    "test</b>, con semilla fija (42) para que el split sea idéntico si cada máquina lo "
    "genera por su cuenta. Batch por máquina: 64 &rarr; batch global = 64 &times; 3 = 192. "
    "Learning rate base 1e-3, escalado &times;3 (regla de escalado lineal, "
    "3e-3) para compensar el batch global mayor. Se pidieron 10 épocas, con "
    "<font face='Courier'>EarlyStopping(patience=3, restore_best_weights=True)</font> "
    "monitoreando <font face='Courier'>val_loss</font>."
)
p(
    "<b>Por paso de entrenamiento:</b> (1) TensorFlow reparte el lote global entre las 3 "
    "máquinas; (2) cada una calcula gradientes localmente sobre su porción; (3) las 3 "
    "sincronizan y promedian esos gradientes por red (all-reduce); (4) cada máquina "
    "aplica la misma actualización, manteniendo sus copias del modelo idénticas. "
    "<b>Qué viaja entre máquinas:</b> gradientes de cada paso. <b>Qué nunca viaja:</b> "
    "las imágenes ni las etiquetas del dataset (cada máquina tiene su propia copia local "
    "generada por <font face='Courier'>data_prep.py</font>)."
)
h2("Desafíos resueltos durante la puesta en marcha")
p(
    "• <b>Versiones de TensorFlow distintas entre máquinas</b> causaban errores de "
    "protocolo al sincronizar gradientes: se fijaron versiones exactas "
    "(TensorFlow 2.21.0) en <font face='Courier'>requirements.txt</font>.<br/>"
    "• <b>Hardware heterogéneo</b> (GPU en algunas máquinas, no en otras) causaba "
    "errores de tipo <font face='Courier'>Aborting RingReduce ... unknown device</font> al "
    "elegir TensorFlow automáticamente NCCL: se forzó CPU y comunicación RING en las 3.<br/>"
    "• <b>Errores de tipeo del <font face='Courier'>TF_CONFIG</font></b> al armarlo a mano "
    "en 3 terminales SSH distintas: se resolvió con "
    "<font face='Courier'>run_worker.sh</font>, que arma el <font face='Courier'>TF_CONFIG</font> "
    "automáticamente a partir de 3 IPs fijas y activa el entorno virtual si hace falta."
)

# ---------------------------------------------------------------------------
# 3. Resumen de resultados
# ---------------------------------------------------------------------------
h1("3. Resumen de resultados (train_worker0.log)")
metric_rows = [["Época", "Pérdida train", "Exactitud train", "Pérdida val", "Exactitud val"]]
epochs = list(range(1, 9))
train_loss = [0.5671, 0.3569, 0.3083, 0.2782, 0.2519, 0.2359, 0.2185, 0.2053]
train_acc = [0.7922, 0.8711, 0.8879, 0.8971, 0.9068, 0.9123, 0.9186, 0.9229]
val_loss = [0.3550, 0.3009, 0.2801, 0.2784, 0.2505, 0.2658, 0.2570, 0.2548]
val_acc = [0.8667, 0.8841, 0.8934, 0.8909, 0.9052, 0.8968, 0.9010, 0.9030]
for e, tl, ta, vl, va in zip(epochs, train_loss, train_acc, val_loss, val_acc):
    marker = "  (mejor val_loss)" if e == 5 else ""
    metric_rows.append([str(e), f"{tl:.4f}", f"{ta*100:.2f} %", f"{vl:.4f}{marker}", f"{va*100:.2f} %"])
table(metric_rows, col_widths=[1.8 * cm, 3.5 * cm, 3.7 * cm, 4.5 * cm, 3.5 * cm])
p(
    "El entrenamiento se detuvo en la <b>época 8 de 10</b> por "
    "<font face='Courier'>EarlyStopping</font>: la pérdida de validación no mejoró durante "
    "3 épocas consecutivas después de su mejor valor (época 5, val_loss = 0.2505). "
    "Al activarse, <font face='Courier'>restore_best_weights=True</font> revirtió el modelo "
    "a los pesos de la época 5 antes de evaluar sobre el conjunto de test — por eso las "
    "métricas de test que siguen corresponden a ese punto, no a la época 8."
)
image("report_assets/accuracy_por_epoca.png")
image("report_assets/loss_por_epoca.png")
p(
    "La curva muestra una mejora rápida en las primeras 2-3 épocas (saliendo de la "
    "inicialización aleatoria de los pesos) y una estabilización alrededor del 90% de "
    "exactitud de validación desde la época 5, con un leve repunte de "
    "<font face='Courier'>val_loss</font> después (señal temprana de sobreajuste) que es "
    "justamente lo que <font face='Courier'>EarlyStopping</font> detectó y corrigió."
)
h2("Test final")
p(
    f"Evaluado sobre <font face='Courier'>fashion_mnist_distribuido.keras</font>: "
    f"<b>pérdida en test = 0.2756</b>, <b>exactitud en test = 89.97%</b>. "
    "Predicciones sobre una muestra de 10 imágenes de test: <b>9/10 correctas</b> "
    "(figura siguiente); el único error observado fue un <i>Abrigo</i> clasificado como "
    "<i>Suéter</i>, una confusión esperable dada la similitud visual entre ambas prendas "
    "en 28x28 px."
)
image("report_assets/predicciones_ejemplo.png")

# ---------------------------------------------------------------------------
# 4. Conformacion del modelo final
# ---------------------------------------------------------------------------
h1("4. Conformación del modelo final")
p(
    "A diferencia del aprendizaje federado (donde cada nodo entrena un modelo local "
    "completo y estos se promedian periódicamente con FedAvg), en el paralelismo de "
    "datos síncrono usado aquí <b>no existen modelos locales independientes en ningún "
    "momento</b>: los gradientes se promedian (all-reduce) <i>antes</i> de que cualquier "
    "copia actualice sus pesos, por lo que las 3 máquinas mantienen siempre una copia "
    "idéntica del mismo modelo. No hay, por lo tanto, un paso de “agregación final”: "
    "el modelo final es simplemente el estado del modelo (idéntico en las 3 máquinas) al "
    "terminar el entrenamiento. Por convención, solo la máquina designada como "
    "<b>chief</b> (worker índice 0, ia-2) escribe el archivo en disco "
    "(<font face='Courier'>fashion_mnist_distribuido.keras</font>) y reporta las métricas "
    "de test, para evitar que las 3 máquinas compitan por escribir el mismo archivo."
)

# ---------------------------------------------------------------------------
# 5. Como reproducirlo
# ---------------------------------------------------------------------------
h1("5. Cómo reproducirlo")
p(
    "Repositorio: "
    "<font face='Courier'>github.com/Camilo1408/taller-aprendizaje-distribuido</font>"
)
p(
    "En <b>cada una de las 3 máquinas</b>:<br/>"
    "<font face='Courier'>git clone https://github.com/Camilo1408/taller-aprendizaje-distribuido.git<br/>"
    "cd taller-aprendizaje-distribuido<br/>"
    "python3.11 -m venv .venv &amp;&amp; source .venv/bin/activate<br/>"
    "pip install -r requirements.txt<br/>"
    "python data_prep.py</font>"
)
p(
    "Editar una sola vez las 3 IPs en <font face='Courier'>run_worker.sh</font> y lanzar, "
    "casi al mismo tiempo, en cada máquina:<br/>"
    "<font face='Courier'>bash run_worker.sh 0 10</font>  (en la máquina IP_0)<br/>"
    "<font face='Courier'>bash run_worker.sh 1 10</font>  (en la máquina IP_1)<br/>"
    "<font face='Courier'>bash run_worker.sh 2 10</font>  (en la máquina IP_2)"
)
p(
    "Al terminar, el modelo final queda en <font face='Courier'>fashion_mnist_distribuido.keras</font> "
    "en la máquina chief, con el log completo en "
    "<font face='Courier'>train_worker0.log</font>. Para regenerar las gráficas y la figura "
    "de predicciones de este informe a partir de ese modelo: "
    "<font face='Courier'>python generate_report_assets.py</font>."
)

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=letter,
    topMargin=1.8 * cm, bottomMargin=1.8 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
)
doc.build(story)
print(f"Informe generado: {OUT_PDF}")
