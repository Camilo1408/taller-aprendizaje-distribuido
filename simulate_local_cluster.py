"""
Simula un cluster de 3 workers EN UNA SOLA MAQUINA (3 procesos locales
hablando entre si por localhost). Sirve para probar que train.py funciona
de punta a punta ANTES de salir a configurar las 3 maquinas reales.

Uso:
    python simulate_local_cluster.py --epochs 2
"""
import json
import os
import subprocess
import sys

NUM_WORKERS = 3
BASE_PORT = 12345


def main():
    workers = [f"localhost:{BASE_PORT + i}" for i in range(NUM_WORKERS)]
    procs = []

    for i in range(NUM_WORKERS):
        tf_config = {
            "cluster": {"worker": workers},
            "task": {"type": "worker", "index": i},
        }
        env = os.environ.copy()
        env["TF_CONFIG"] = json.dumps(tf_config)
        print(f"Lanzando worker {i} en {workers[i]}...")
        procs.append(
            subprocess.Popen([sys.executable, "train.py", *sys.argv[1:]], env=env)
        )

    exit_codes = [p.wait() for p in procs]
    if any(exit_codes):
        print(f"Algun worker termino con error. Codigos: {exit_codes}")
        sys.exit(1)
    print("Simulacion local completada sin errores.")


if __name__ == "__main__":
    main()
