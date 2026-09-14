# Entrenamiento distribuido de un clasificador Fashion-MNIST en 3 maquinas

## 1. Finalidad del proyecto

Entrenar un mismo modelo de clasificacion de imagenes (una CNN pequena)
repartiendo el trabajo entre **3 computadoras** que colaboran para producir
**un unico modelo final**, en lugar de que una sola maquina cargue con todo
el entrenamiento. El objetivo pedagogico es entender y demostrar:

- Como se reparte el computo de un entrenamiento entre varias maquinas
  (paralelismo de datos, sincrono).
- Como se sincronizan los gradientes entre maquinas (all-reduce).
- Los ajustes que hay que hacer (learning rate, tamano de batch) cuando se
  agregan mas maquinas.
- Las particularidades de organizar un pipeline completo: descarga de datos,
  particion train/val/test, entrenamiento, evaluacion y mejora del modelo.

## 2. Como funciona (arquitectura)

Se usa **paralelismo de datos sincrono** con
`tf.distribute.MultiWorkerMirroredStrategy` (nativo de TensorFlow/Keras):

```
Maquina A (worker 0) ---\
Maquina B (worker 1) ----+---> all-reduce (promedia gradientes) ---> misma actualizacion
Maquina C (worker 2) ---/                                            aplicada en las 3
```

1. Las 3 maquinas ejecutan **exactamente el mismo script** (`train.py`) con
   una **copia identica del modelo**.
2. En cada paso de entrenamiento, el lote global (`global_batch_size`) se
   reparte en 3 partes; cada maquina calcula el forward/backward pass sobre
   su parte usando su propio CPU/GPU.
3. Las 3 maquinas intercambian y promedian sus gradientes por red mediante
   una operacion colectiva (all-reduce, en anillo).
4. Cada maquina aplica la misma actualizacion promediada a su copia del
   modelo, por lo que las 3 copias se mantienen **identicas** paso a paso.
5. Al terminar, cualquiera de las 3 copias es el modelo final; por
   convencion solo la maquina "chief" (worker 0) guarda el archivo y
   reporta metricas, para no pisarse entre maquinas.

Quien coordina esto es la variable de entorno `TF_CONFIG`: un JSON que le
dice a cada proceso (a) la lista completa de IP:puerto del cluster y (b) su
propio indice dentro de esa lista. TensorFlow usa eso para abrir conexiones
gRPC entre las 3 maquinas antes de empezar a entrenar.

## 3. Dataset: Fashion-MNIST

- Incluido directamente en `keras.datasets` (no hay que descargarlo a mano).
- 70,000 imagenes en escala de grises de 28x28 px, 10 clases de prendas de
  ropa (camiseta, pantalon, sueter, vestido, abrigo, sandalia, camisa,
  zapatilla, bolso, bota).
- Viene pre-separado en 60,000 (train) + 10,000 (test). Este proyecto separa
  10,000 del train original para usarlos como **validacion**, quedando:
  - **train:** 50,000 imagenes
  - **val:** 10,000 imagenes
  - **test:** 10,000 imagenes (el test set original, no se toca hasta el final)

Se eligio por ser liviano (~30 MB), rapido de entrenar y suficientemente
dificil como para que se note la diferencia entre un modelo bien y mal
entrenado (a diferencia de MNIST de digitos, que es casi trivial).

## 4. El modelo

Una CNN pequena (`model.py`):

```
Input(28,28,1) -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool
  -> Flatten -> Dense(128) -> Dropout(0.3) -> Dense(10, softmax)
```

Con ~225k parametros entrena en segundos por epoca incluso en CPU, lo cual
es ideal para un proyecto de aprendizaje distribuido: el foco esta en la
coordinacion entre maquinas, no en esperar horas de entrenamiento.

**Referencia esperada:** con esta arquitectura, sin distribuir, se suele
llegar a ~91-92% de accuracy en test tras 10-15 epocas.

## 5. Requisitos previos

- 3 computadoras **Linux** en la **misma red local** (LAN o VPN), que puedan
  hacerse ping entre si (o accesibles por SSH si estan en la nube).
- La **misma version de Python y de TensorFlow** instalada en las 3 (usar
  `requirements.txt`).
- Puertos libres para el trafico entre workers (en los ejemplos se usa el
  `12345`, hay que permitirlo en el firewall de las 3 maquinas).
- `git` instalado en las 3 maquinas.

### 5.1 Obtener el proyecto en las 3 maquinas

Este proyecto vive en:
**https://github.com/Camilo1408/taller-aprendizaje-distribuido**

En **cada una de las 3 maquinas**:

```bash
git clone https://github.com/Camilo1408/taller-aprendizaje-distribuido.git
cd taller-aprendizaje-distribuido
```

### 5.2 Version de Python en Linux

> **Importante:** TensorFlow todavia no soporta Python 3.14 (la version mas
> nueva). Se necesita una version **3.10, 3.11 o 3.12**. Este proyecto se
> probo y quedo funcionando con **Python 3.11**.

Primero revisar que version trae el sistema:

```bash
python3 --version
```

Si ya marca 3.10, 3.11 o 3.12, se puede usar directamente esa (saltar a
"Crear el entorno virtual" mas abajo, usando `python3` en vez de `python3.11`).
Si no, instalar una version compatible segun la distribucion:

**Ubuntu / Debian** (via el PPA deadsnakes, la forma mas simple de tener una
version exacta sin afectar el Python del sistema):

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

**Fedora / RHEL / CentOS:**

```bash
sudo dnf install -y python3.11
```

**Cualquier distribucion, sin permisos de administrador** (via `pyenv`,
compila Python desde codigo fuente en el home del usuario):

```bash
curl -fsSL https://pyenv.run | bash
# agregar pyenv al PATH segun lo que indique el instalador (~/.bashrc), luego:
exec $SHELL
pyenv install 3.11.9
pyenv local 3.11.9
```

### 5.3 Crear el entorno virtual e instalar dependencias

En **cada una de las 3 maquinas**, dentro de la carpeta del proyecto:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Si usaste `pyenv`, reemplaza `python3.11` por `python`, ya que `pyenv
local` deja esa version activa por defecto en la carpeta.)

A partir de aqui, todos los comandos de esta guia asumen que el entorno
virtual esta activado (`source .venv/bin/activate`); si abres una terminal
nueva o te conectas de nuevo por SSH, hay que activarlo otra vez antes de
correr cualquier script.

> Se fija `TF_USE_LEGACY_KERAS=1` dentro de los scripts porque, a partir de
> TensorFlow 2.16, `tf.keras` usa Keras 3 por defecto y el soporte de
> `MultiWorkerMirroredStrategy` ahi es mas nuevo/menos probado. Con esa
> variable, `tf.keras` usa la implementacion clasica (via el paquete
> `tf_keras`), que es la ruta oficialmente documentada y estable para
> entrenamiento multi-maquina.

## 6. Preparar el dataset (train / val / test)

En **cada una de las 3 maquinas** (con el entorno virtual activado; o en una
sola maquina y despues copiar la carpeta `data/` generada a las otras dos
con `scp -r`, para garantizar que las 3 usan exactamente los mismos datos):

```bash
python data_prep.py
```

Esto descarga Fashion-MNIST, normaliza los pixeles a `[0, 1]`, hace el split
train/val con una semilla fija (para que el split sea igual si cada maquina
lo genera por separado) y guarda 3 archivos `.npz` en `data/`.

## 7. Probar antes de usar las 3 maquinas reales

**7.1 Prueba de humo en 1 maquina** (sin `TF_CONFIG`, la estrategia se
comporta como si hubiera 1 solo worker):

```bash
python train.py --epochs 2
```

Si esto corre sin errores y termina imprimiendo un accuracy de test,
el modelo y el pipeline de datos estan bien.

**7.2 Simular 3 workers en 1 sola maquina** (3 procesos locales que se
comunican por `localhost`, sin necesitar todavia las 3 computadoras):

```bash
python simulate_local_cluster.py --epochs 2
```

Este paso es el mas importante para depurar: valida que la logica de
`TF_CONFIG`, el reparto del batch y el guardado "solo el chief" funcionan,
sin la complejidad extra de la red fisica.

> **Verificado:** este proyecto se probo end-to-end con Python 3.11 y
> TensorFlow 2.21. La corrida 7.1 (1 maquina) y la 7.2 (3 workers simulados)
> terminaron ambas sin errores; solo el worker 0 imprimio el accuracy de
> test y genero `fashion_mnist_distribuido.keras`, mientras que los workers
> 1 y 2 escribieron su copia en `workertemp_1/` y `workertemp_2/` como se
> esperaba (para no pisar el archivo del chief).

## 8. Configurar las 3 maquinas reales

1. Obtener la IP local de cada maquina:
   ```bash
   hostname -I
   # o, mas detallado:
   ip addr show
   ```
   Anotar la IP de la interfaz de la red que comparten las 3 (ej. `192.168.1.10`).

2. Elegir un puerto libre (ej. `12345`) y abrirlo en el firewall de las 3
   maquinas:

   **Ubuntu/Debian (ufw):**
   ```bash
   sudo ufw allow 12345/tcp
   ```
   **Fedora/RHEL (firewalld):**
   ```bash
   sudo firewall-cmd --add-port=12345/tcp --permanent
   sudo firewall-cmd --reload
   ```
   Verificar desde otra maquina que el puerto responde (una vez que
   `train.py` este corriendo) con `nc -zv <ip> 12345`.

3. Definir el `TF_CONFIG` de cada maquina **antes** de ejecutar `train.py`.
   Ejemplo con IPs de muestra (reemplazar por las reales):

   | Maquina | IP              | index |
   |---------|-----------------|-------|
   | A       | 192.168.1.10    | 0     |
   | B       | 192.168.1.11    | 1     |
   | C       | 192.168.1.12    | 2     |

   En la maquina A (con el entorno virtual activado):
   ```bash
   export TF_CONFIG='{"cluster": {"worker": ["192.168.1.10:12345","192.168.1.11:12345","192.168.1.12:12345"]}, "task": {"type": "worker", "index": 0}}'
   python train.py --epochs 10
   ```
   En la maquina B (mismo comando, cambia solo `"index": 1`):
   ```bash
   export TF_CONFIG='{"cluster": {"worker": ["192.168.1.10:12345","192.168.1.11:12345","192.168.1.12:12345"]}, "task": {"type": "worker", "index": 1}}'
   python train.py --epochs 10
   ```
   En la maquina C (`"index": 2`):
   ```bash
   export TF_CONFIG='{"cluster": {"worker": ["192.168.1.10:12345","192.168.1.11:12345","192.168.1.12:12345"]}, "task": {"type": "worker", "index": 2}}'
   python train.py --epochs 10
   ```

4. **Lanzar los 3 comandos casi al mismo tiempo.** Cada proceso se queda
   esperando a que las otras 2 maquinas respondan en su IP:puerto antes de
   arrancar el entrenamiento; si una maquina tarda mucho en lanzarse, las
   otras dos simplemente esperan (no fallan). Si te conectas por SSH a las 3
   maquinas, conviene:
   - Abrir 3 terminales (una por maquina, ej. 3 pestanas de SSH) y lanzar los
     3 comandos en rapida sucesion, o
   - Usar `tmux`/`screen` en cada maquina para dejar el proceso corriendo
     aunque se corte la conexion SSH, ej.:
     ```bash
     tmux new -s training
     # dentro de tmux: export TF_CONFIG=... && python train.py --epochs 10
     # Ctrl+b luego d para salir sin matar el proceso
     ```

## 9. Evaluar resultados

- Solo la maquina "chief" (index 0) imprime el accuracy/loss final de test y
  guarda el modelo (`fashion_mnist_distribuido.keras`).
- Comparar el accuracy de test contra la corrida de un solo worker (paso
  7.1) para confirmar que el resultado es equivalente (el paralelismo de
  datos sincrono no deberia perjudicar la calidad del modelo).
- Para ver curvas de entrenamiento: `tensorboard --logdir logs` en la
  maquina chief.
- **Nota realista:** con un dataset tan chico, es normal que 3 maquinas no
  sean mas rapidas que 1 sola (el overhead de comunicacion por red puede
  superar el tiempo de computo). El paralelismo de datos rinde cuando el
  computo por batch es costoso (modelos mas grandes, imagenes mas grandes);
  por eso en la seccion 11 se sugiere probar con CIFAR-10.

## 10. Mejorar el modelo

Ideas para iterar, de mas simple a mas avanzado:

1. **Mas epocas** y dejar que `EarlyStopping` decida cuando parar.
2. **Data augmentation** (flip horizontal, pequenas rotaciones/zoom) con
   `tf.keras.layers.RandomFlip` / `RandomRotation` antes de la primera capa
   `Conv2D`, para reducir overfitting.
3. **Ajustar el learning rate**: `train.py` ya aplica la "regla de escalado
   lineal" (multiplica el LR base por el numero de workers); si el
   entrenamiento se ve inestable, probar con un warm-up de LR en las
   primeras epocas en vez de escalar desde el primer paso.
4. **Aumentar `per_worker_batch_size`** y ver el efecto en velocidad y
   accuracy final.
5. **Arquitectura**: agregar `BatchNormalization`, otra capa `Conv2D`, o
   probar `Dropout` mas alto/bajo.
6. **Regularizacion**: `kernel_regularizer=tf.keras.regularizers.l2(...)` en
   las capas densas si se ve overfitting en las curvas de val vs train.

## 11. Otros datasets / modelos posibles (todos en `keras.datasets`)

| Dataset | Tamano | Tarea | Modelo sugerido | Cuando usarlo |
|---|---|---|---|---|
| `mnist` | 70k imagenes 28x28 grises | Clasificacion de digitos 0-9 | MLP o CNN muy simple | Baseline mas facil todavia; util para una primera prueba rapida del pipeline distribuido. |
| `fashion_mnist` (este proyecto) | 70k imagenes 28x28 grises | 10 clases de ropa | CNN pequena | Buen balance dificultad/velocidad para practicar distribucion. |
| `cifar10` | 60k imagenes 32x32 color | 10 clases de objetos/animales | CNN mas profunda (varias Conv2D + BatchNorm) | Cuando se quiera que el paralelismo de datos realmente acelere el entrenamiento (mas computo por imagen). |
| `imdb` | 50k resenas de peliculas | Sentimiento positivo/negativo (texto) | Embedding + LSTM/GRU o Dense | Si se prefiere practicar con texto en vez de imagenes. |
| `california_housing` | ~20k filas tabulares | Regresion (precio de vivienda) | MLP pequena (Dense + Dense + salida lineal) | Si se prefiere un problema tabular/regresion en vez de clasificacion de imagenes. |

Cambiar de dataset solo implica editar `data_prep.py` (que dataset carga y
como normaliza) y `model.py` (la arquitectura de entrada/salida); `train.py`
y la logica de distribucion no cambian.

## 12. Problemas comunes

- **Un worker se queda "colgado" esperando**: revisar que las 3 maquinas
  puedan verse por red (`ping <ip>`), que el puerto no este bloqueado por el
  firewall (`sudo ufw status` / `sudo firewall-cmd --list-ports`), y que el
  puerto realmente este escuchando una vez lanzado `train.py`
  (`ss -tulnp | grep 12345` en la maquina que deberia estar escuchando).
- **`python3.11: command not found`**: la version instalada quedo como
  `python3.11` pero el binario no esta en el PATH, o el paquete
  `python3.11-venv` no se instalo (en Ubuntu, `venv` es un paquete aparte).
- **Version distinta de TensorFlow entre maquinas**: puede causar errores de
  protocolo al conectar los workers; usar la misma version en las 3 (fijada
  en `requirements.txt`, instalada dentro del `.venv` de cada una).
- **Rutas relativas distintas**: `train.py` busca `data/` en la carpeta
  actual; ejecutar siempre desde la carpeta del proyecto (`cd
  taller-aprendizaje-distribuido`) en las 3 maquinas, con el `.venv`
  activado.
- **La conexion SSH se corta y mata el entrenamiento**: usar `tmux` o
  `screen` (ver seccion 8) en vez de dejar el proceso atado directamente a
  la sesion SSH.
- **Uno de los workers falla a mitad de entrenamiento**: con
  `MultiWorkerMirroredStrategy` todos los workers deben seguir vivos durante
  todo el entrenamiento (es una operacion colectiva); si una maquina se cae
  hay que reiniciar el entrenamiento en las 3.
