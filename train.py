"""
Entrenamiento distribuido de datos (data-parallel, sincrono) sobre 3 maquinas
usando tf.distribute.MultiWorkerMirroredStrategy.

Este MISMO script se ejecuta sin cambios en las 3 maquinas. Lo unico que
cambia entre ellas es la variable de entorno TF_CONFIG, que le dice a cada
proceso cual es su indice (0, 1 o 2) dentro del cluster y cuales son las
IP:puerto de las otras dos.

Ejemplo de TF_CONFIG para la maquina con indice 0 (ver README.md para el
detalle de las 3 maquinas):

    TF_CONFIG={"cluster": {"worker": ["192.168.1.10:12345",
                                       "192.168.1.11:12345",
                                       "192.168.1.12:12345"]},
               "task": {"type": "worker", "index": 0}}

Uso:
    python train.py --epochs 10 --per_worker_batch_size 64
"""
import argparse
import json
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Fuerza a que las 3 maquinas usen solo CPU, aunque alguna tenga GPU.
# MultiWorkerMirroredStrategy elige automaticamente NCCL cuando detecta GPU,
# pero NCCL no soporta workers sin GPU: si el cluster es heterogeneo (unas
# maquinas con GPU y otras sin), la sincronizacion de gradientes falla con
# errores tipo "Aborting RingReduce ... unknown device". Como este modelo es
# chico y no necesita GPU, lo mas simple y robusto es desactivarla en todas.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from model import build_model

DATA_DIR = "data"


def load_split(name):
    with np.load(f"{DATA_DIR}/fashion_mnist_{name}.npz") as data:
        return data["x"], data["y"]


def make_dataset(x, y, batch_size, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def is_chief(task_type, task_id):
    # Sin TF_CONFIG (una sola maquina) task_type es None -> tambien es chief.
    return task_type is None or task_type == "chief" or (task_type == "worker" and task_id == 0)


def write_filepath(filepath, task_type, task_id):
    """Evita que los workers no-chief pisen el archivo real: cada uno escribe
    en un directorio temporal propio y solo el chief usa la ruta final."""
    dirpath = os.path.dirname(filepath) or "."
    base = os.path.basename(filepath)
    if not is_chief(task_type, task_id):
        dirpath = os.path.join(dirpath, f"workertemp_{task_id}")
        os.makedirs(dirpath, exist_ok=True)
    return os.path.join(dirpath, base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_worker_batch_size", type=int, default=64)
    parser.add_argument("--base_learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    print(f"TF_CONFIG detectado: {tf_config or '(vacio -> modo 1 maquina)'}")

    # Se fuerza RING (en vez de dejar el AUTO por defecto) porque RING
    # funciona por gRPC y soporta CPU; AUTO puede elegir NCCL si alguna
    # maquina del cluster tiene GPU, y NCCL no funciona con workers en CPU.
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.RING
    )
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options
    )
    num_workers = strategy.num_replicas_in_sync
    print(f"Numero de workers sincronizando gradientes: {num_workers}")

    # Regla de escalado lineal: si repartimos el batch entre N maquinas,
    # el batch global crece x N, así que el learning rate tambien se escala
    # x N para mantener una magnitud de actualizacion comparable a 1 sola
    # maquina con per_worker_batch_size.
    global_batch_size = args.per_worker_batch_size * num_workers
    scaled_lr = args.base_learning_rate * num_workers

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    train_ds = make_dataset(x_train, y_train, global_batch_size, shuffle=True)
    val_ds = make_dataset(x_val, y_val, global_batch_size, shuffle=False)

    with strategy.scope():
        model = build_model(learning_rate=scaled_lr)

    task_type = strategy.cluster_resolver.task_type if strategy.cluster_resolver else None
    task_id = strategy.cluster_resolver.task_id if strategy.cluster_resolver else 0

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]
    if is_chief(task_type, task_id):
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="./logs"))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2 if is_chief(task_type, task_id) else 0,
    )

    # model.evaluate/save son operaciones colectivas: TODOS los workers deben
    # llamarlas, aunque solo el chief se quede con el resultado/archivo final.
    x_test, y_test = load_split("test")
    test_ds = make_dataset(x_test, y_test, global_batch_size, shuffle=False)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    save_path = write_filepath("fashion_mnist_distribuido.keras", task_type, task_id)
    model.save(save_path)

    if is_chief(task_type, task_id):
        print(f"\nAccuracy en test: {test_acc:.4f}  |  loss en test: {test_loss:.4f}")
        print(f"Modelo final guardado en: {save_path}")


if __name__ == "__main__":
    main()
