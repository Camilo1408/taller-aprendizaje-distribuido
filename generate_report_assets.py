"""
Genera las graficas y la imagen de predicciones de ejemplo para el informe
final, a partir del modelo realmente entrenado en las 3 maquinas
(fashion_mnist_distribuido_fixed.keras) y de las metricas reales
registradas durante ese entrenamiento (train_worker0.log de la maquina
chief, ia-2).

Uso:
    python generate_report_assets.py
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

OUT_DIR = "report_assets"
CLASS_NAMES = [
    "Camiseta/top", "Pantalon", "Sueter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota",
]

# Metricas reales impresas por la maquina chief (ia-2) durante las 8 epocas
# que efectivamente corrieron antes de que EarlyStopping detuviera el
# entrenamiento (ver train_worker0.log).
EPOCHS = list(range(1, 9))
TRAIN_LOSS = [0.5671, 0.3569, 0.3083, 0.2782, 0.2519, 0.2359, 0.2185, 0.2053]
TRAIN_ACC = [0.7922, 0.8711, 0.8879, 0.8971, 0.9068, 0.9123, 0.9186, 0.9229]
VAL_LOSS = [0.3550, 0.3009, 0.2801, 0.2784, 0.2505, 0.2658, 0.2570, 0.2548]
VAL_ACC = [0.8667, 0.8841, 0.8934, 0.8909, 0.9052, 0.8968, 0.9010, 0.9030]
TEST_LOSS = 0.2756
TEST_ACC = 0.8997


def plot_accuracy():
    plt.figure(figsize=(6, 4))
    plt.plot(EPOCHS, TRAIN_ACC, marker="o", label="Exactitud entrenamiento")
    plt.plot(EPOCHS, VAL_ACC, marker="o", label="Exactitud validacion")
    plt.title("Exactitud por epoca (3 maquinas sincronizadas)")
    plt.xlabel("Epoca")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/accuracy_por_epoca.png", dpi=150)
    plt.close()


def plot_loss():
    plt.figure(figsize=(6, 4))
    plt.plot(EPOCHS, TRAIN_LOSS, marker="o", label="Perdida entrenamiento")
    plt.plot(EPOCHS, VAL_LOSS, marker="o", label="Perdida validacion")
    plt.title("Perdida por epoca (3 maquinas sincronizadas)")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/loss_por_epoca.png", dpi=150)
    plt.close()


def plot_predictions():
    model = tf.keras.models.load_model("fashion_mnist_distribuido.keras")
    with np.load("data/fashion_mnist_test.npz") as d:
        x_test, y_test = d["x"], d["y"]

    rng = np.random.default_rng(seed=7)
    idx = rng.choice(len(x_test), size=10, replace=False)
    images = x_test[idx]
    true_labels = y_test[idx]
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    correct = int((pred_labels == true_labels).sum())

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    for ax, img, true_l, pred_l in zip(axes.flat, images, true_labels, pred_labels):
        ax.imshow(img.squeeze(), cmap="gray")
        ok = true_l == pred_l
        color = "green" if ok else "red"
        ax.set_title(
            f"Real: {CLASS_NAMES[true_l]}\nPred: {CLASS_NAMES[pred_l]}",
            fontsize=8, color=color,
        )
        ax.axis("off")
    fig.suptitle(f"Predicciones de ejemplo ({correct}/10 correctas)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/predicciones_ejemplo.png", dpi=150)
    plt.close()

    return correct, len(images)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_accuracy()
    plot_loss()
    correct, total = plot_predictions()
    print(f"Graficas generadas en {OUT_DIR}/. Predicciones de ejemplo: {correct}/{total} correctas.")
