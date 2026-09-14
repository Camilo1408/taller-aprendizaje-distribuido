#!/usr/bin/env bash
# Arma el TF_CONFIG automaticamente y lanza train.py, para no tener que
# escribir el JSON a mano en cada maquina (facil de tipear mal el index o
# la lista de IPs, como ya nos paso).
#
# Editar UNA VEZ las 3 IPs de mas abajo (deben quedar iguales en las 3
# maquinas). Despues, en cada maquina correr solo:
#
#   ./run_worker.sh 0 10   # en la maquina que es el worker 0
#   ./run_worker.sh 1 10   # en la maquina que es el worker 1
#   ./run_worker.sh 2 10   # en la maquina que es el worker 2
#
# (el segundo argumento son las epocas, opcional, default 10)

set -euo pipefail
cd "$(dirname "$0")"

IP_0="172.16.79.158"
IP_1="172.16.79.159"
IP_2="172.16.79.120"
PORT=12345

if [ $# -lt 1 ]; then
  echo "Uso: $0 <index 0|1|2> [epochs]"
  exit 1
fi

INDEX="$1"
EPOCHS="${2:-10}"

# Activa el entorno virtual automaticamente si no esta activo, para evitar
# el error de "ModuleNotFoundError: tensorflow" por olvidar activarlo.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  source .venv/bin/activate
fi

export TF_CONFIG=$(python - "$INDEX" "$IP_0" "$IP_1" "$IP_2" "$PORT" <<'PYEOF'
import json, sys
index = int(sys.argv[1])
ip0, ip1, ip2, port = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
workers = [f"{ip0}:{port}", f"{ip1}:{port}", f"{ip2}:{port}"]
print(json.dumps({"cluster": {"worker": workers}, "task": {"type": "worker", "index": index}}))
PYEOF
)

echo "Worker index: $INDEX"
echo "TF_CONFIG=$TF_CONFIG"
python train.py --epochs "$EPOCHS" 2>&1 | tee "train_worker${INDEX}.log"
