"""
Descarga Fashion-MNIST (incluido en keras.datasets) y lo deja listo en disco,
particionado en train / val / test, normalizado y en formato .npz.

Ejecutar UNA VEZ en cada una de las 3 maquinas (o ejecutarlo en una sola
maquina y copiar la carpeta 'data/' generada a las otras dos):

    python data_prep.py

Genera:
    data/fashion_mnist_train.npz   (50 000 ejemplos)
    data/fashion_mnist_val.npz     (10 000 ejemplos)
    data/fashion_mnist_test.npz    (10 000 ejemplos, el test set original)
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

VAL_SIZE = 10_000
OUT_DIR = "data"

CLASS_NAMES = [
    "Camiseta/top", "Pantalon", "Sueter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Descargando Fashion-MNIST (se guarda en cache ~/.keras/datasets)...")
    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # Normalizamos a [0, 1] y agregamos el canal (28, 28) -> (28, 28, 1)
    def prep(x):
        x = x.astype("float32") / 255.0
        return np.expand_dims(x, axis=-1)

    x_train_full = prep(x_train_full)
    x_test = prep(x_test)
    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    # Separamos un conjunto de validacion a partir del train original.
    # Se mezcla con una semilla fija para que las 3 maquinas, si generan
    # el dataset por separado, obtengan exactamente el mismo split.
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(x_train_full))
    val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    np.savez_compressed(f"{OUT_DIR}/fashion_mnist_train.npz", x=x_train, y=y_train)
    np.savez_compressed(f"{OUT_DIR}/fashion_mnist_val.npz", x=x_val, y=y_val)
    np.savez_compressed(f"{OUT_DIR}/fashion_mnist_test.npz", x=x_test, y=y_test)

    print("Listo. Tamanos de cada particion:")
    print(f"  train: {x_train.shape} imagenes, {y_train.shape} etiquetas")
    print(f"  val:   {x_val.shape} imagenes, {y_val.shape} etiquetas")
    print(f"  test:  {x_test.shape} imagenes, {y_test.shape} etiquetas")
    print(f"Clases: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
