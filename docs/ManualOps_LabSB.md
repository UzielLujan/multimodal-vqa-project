
# Manual de Operaciones: Pipeline de ML en el Clúster Lab-SB (CIMAT)

Esta guía resume el flujo de trabajo y las mejores prácticas para ejecutar proyectos de Machine Learning en el clúster de supercómputo Lab-SB, basándose en lecciones aprendidas.

## 1. Estructura del Proyecto

Una estructura de directorios organizada es fundamental para evitar problemas de rutas.

```text
/tu-proyecto/
├── data/               # Datasets de entrada
├── src/                # Código fuente (.py)
│   ├── train.py
│   └── ...
├── models/             # Checkpoints de modelos guardados
├── results/            # Métricas y gráficos de evaluación
├── logs/               # Logs de salida de los trabajos de SLURM
└── run_entrenamiento.sh  # Scripts de lanzamiento en la raíz
```

> **Clave:** Mantén los scripts de lanzamiento (.sh) en la raíz del proyecto. Esto simplifica enormemente el manejo de rutas.

---

## 2. El Script de Lanzamiento (.sh)

Este archivo es el "control remoto" que le da las instrucciones al clúster. Usa los siguientes componentes como plantilla.

### 2.1. Configuración de SLURM (#SBATCH)

Estas son las directivas que le dicen a SLURM qué recursos necesitas.

```bash
#!/bin/bash

# --- Configuración de SLURM para Lab-SB ---
#SBATCH --job-name=mi-nuevo-proyecto   # Nombre del trabajo (aparece en squeue)
#SBATCH --partition=GPU              # Pide un nodo con GPU (la forma correcta en este clúster)
#SBATCH --nodes=1                    # Número de nodos (casi siempre 1)
#SBATCH --ntasks=1                   # Tareas por nodo (casi siempre 1)
#SBATCH --cpus-per-task=16           # Número de núcleos de CPU
#SBATCH --mem=0                      # 0 significa que pides toda la memoria del nodo
#SBATCH --time=08:00:00              # Tiempo máximo de ejecución (HH:MM:SS)
#SBATCH --chdir=/home/tu_usuario/tu-proyecto # ¡CRÍTICO! Directorio de trabajo
#SBATCH --output=logs/%x-%j.log      # Dónde guardar el log. %x=nombre, %j=ID
```

> **Lección Aprendida:** La directiva `--chdir` es esencial. Fija el punto de partida de tu script y evita errores de "archivo no encontrado". La directiva `--gpus-per-task` no es compatible con este clúster; pedir la partición GPU es suficiente.

### 2.2. Argumentos y Preparación

Define los parámetros que pasarás a tu script de Python.

```bash
# --- Buenas Prácticas ---
set -e # El script fallará si un comando falla

# --- Argumentos del Script ---
# Define valores por defecto para que el script pueda correr sin argumentos
MODELO_BASE=${1:-"ruta/a/modelo_default"}
NOMBRE_RUN=${2:-"run_default"}
EPOCAS=${3:-5}

# --- Preparación del Entorno ---
mkdir -p logs # Crea la carpeta de logs si no existe

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Modelo Base: $MODELO_BASE | Nombre Run: $NOMBRE_RUN"
echo "========================================================"
```

### 2.3. Ejecución del Código Python

Esta es la sección donde activas tu entorno y lanzas el script.

```bash
# --- Activación de Conda y Ejecución ---
# Añade anaconda al PATH del sistema
export PATH="/opt/anaconda_python311/bin:$PATH"

# Lanza el script de Python usando el entorno de Conda
# Para entrenamiento distribuido en 2 GPUs (DDP):
echo "Iniciando entrenamiento distribuido (DDP)..."
conda run -n tu-env-conda torchrun --nproc_per_node=2 src/train.py \
    --model_name "$MODELO_BASE" \
    --run_name "$NOMBRE_RUN" \
    --epochs "$EPOCAS"

# Para un script normal (entrenamiento en 1 GPU o inferencia):
# echo "Iniciando script de Python..."
# conda run -n tu-env-conda python src/tu_script.py \
#     --arg1 "$ARG1" \
#     --arg2 "$ARG2"
```

> **Lección Aprendida:** Usa `torchrun --nproc_per_node=N` para el entrenamiento distribuido. La librería transformers lo detectará automáticamente y activará DDP.

---

## 3. El Script de Python

Tu script de Python debe estar preparado para recibir los argumentos desde la línea de comandos usando la librería argparse.

```python
# En tu src/train.py
import argparse

def main(args):
    print(f"Entrenando el modelo: {args.model_name}")
    # ... tu lógica aquí ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo de ML.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    # ... otros argumentos ...
    
    args = parser.parse_args()
    main(args)
```

---

## 4. Flujo de Trabajo en la Terminal

### Conectarse al Clúster:
- Usa Bitvise SSH Client o `ssh tu_usuario@el-insurgente.cimat.mx`.

### Lanzar el Trabajo:
- Navega a la raíz de tu proyecto y ejecuta sbatch.
```bash
cd /home/tu_usuario/tu-proyecto
sbatch run_entrenamiento.sh "ruta/a/mi_modelo" "mi_primer_run" 10
```

### Monitorear el Trabajo:
- Ver la cola: `squeue -u tu_usuario` (Busca el estado R para Running o PD para Pending).
- Ver detalles post-ejecución: `sacct --jobs=JOB_ID -o JobID,State,ExitCode`. Un ExitCode de 0:0 significa éxito. 1:0 significa que el script falló.

### Depurar (Si Falla):
- El ExitCode es 1:0. ¿Qué haces?
- Revisa el log: La respuesta siempre está en el archivo de log que definiste.
```bash
cd logs
cat mi-nuevo-proyecto-JOB_ID.log
```
- Lee el Traceback de Python al final del archivo. Te dirá exactamente en qué línea de tu código ocurrió el error.

> **Lección Final:** El 99% de los errores se resuelven leyendo el archivo de log. Es tu mejor amigo en el clúster.
    
## Comandos Esenciales para el Clúster Lab-SB

### 1. Entorno y Activación

- Carga la configuración inicial de la terminal (a veces necesario en nuevas sesiones):
```bash
source ~/.bashrc
```

- Activar entorno de Conda específico para el proyecto
```bash
conda activate llms-mx-env
```

### 2. Lanzar Trabajos (sbatch) 
Lanzar un entrenamiento MTL con argumentos posicionales:
- Uso: 
```bash
sbatch <script> "<ruta_modelo>" "<nombre_run>" <epocas> <batch_size> <max_length>
```
- Ejemplo:

```bash
sbatch run_mtl_2gpu.sh "models/BETO_local" "BETO_MTL_SO" 6 32 256
```
Lanzar una predicción con argumentos posicionales
- Uso: 
```bash
sbatch <script> "<ruta_modelo>" "<nombre_run>" "<ruta_tokenizer>"
```
- Ejemplo: 

```bash
sbatch run_prediction.sh "models/BETO_MTL_SO" "BETO_MTL_SO_submission" "dccuchile/bert-base-spanish-wwm-cased"
```
### 3. Monitorear Trabajos 

- Ver todos tus trabajos en la cola (R=Running, PD=Pending)
```bash
squeue -u est_posgrado_uziel.lujan
```
- Ver el estado final y código de salida de un trabajo específico (0:0 = Éxito, 1:0 = Error)
```bash
sacct --jobs=152147 -o JobID,State,ExitCode
```
- Cancelar un trabajo que está en la cola o corriendo
```bash
scancel 152155
```
- Revisar el estado de todos los trabajos en la partición GPU
```bash 
squeue -p GPU
```

### 4. Revisar Logs y Archivos 

- Ver el contenido de un log en tiempo real (muy útil para monitorear)
```bash
tail -f logs/sentiment-analysis-mx-152147.log
```
- Ver el contenido completo de un log de entrenamiento
```bash
cat logs/sentiment-analysis-mx-152128.log
```
- Ver el contenido completo de un log de inferencia
```bash
cat logs/sentiment-inference-mx-152535.log
```
- Listar archivos en el directorio actual, ordenados por fecha de modificación (los más nuevos al principio)
```bash
ls -lt
```
### 5. Obtener un shell interactivo en un nodo de cómputo (pendiente a probar): 
A veces necesitas depurar algo directamente en un nodo con GPU. Este comando es invaluable.

```bash
# Pide un trabajo interactivo en un nodo GPU por 30 minutos
srun --partition=GPU --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=0 --time=00:30:00 --pty bash -i
```
Una vez dentro, puedes activar tu entorno de conda y ejecutar nvidia-smi para ver las GPUs o correr un script de python interactivamente.

### 6. Ver información detallada de un trabajo: 
`sacct` es bueno para el post-mortem, pero `scontrol` te da todos los detalles de un trabajo en ejecución o en cola.

```bash
scontrol show job <JOB_ID>
```

### 7. Manejo de Modelos Pre-entrenados (Sin Acceso a Internet)
Un detalle fundamental del clúster Lab-SB es que los nodos de cómputo no tienen conexión a internet. Esto es una medida de seguridad y estabilidad común en entornos de alto rendimiento.

Como consecuencia, no puedes descargar modelos, tokenizers o datasets directamente desde hubs como Hugging Face o PyTorch Hub dentro de un script que se ejecuta en el clúster.

El flujo de trabajo correcto es un proceso de dos pasos: Descargar localmente, transferir y usar en el clúster.

### 7.1 Paso 1: Descargar el Modelo en tu Máquina Local
En tu computadora personal (o cualquier máquina con internet), necesitas un script para descargar todos los artefactos necesarios (el modelo, el tokenizer, archivos de configuración, etc.) y guardarlos en una carpeta.

Aquí tienes un script de ejemplo (download_model.py) que hace precisamente eso:

```Python

# src/download_model.py
import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model(model_name, output_dir):
    """
    Descarga un modelo y su tokenizer desde Hugging Face Hub 
    y los guarda en un directorio local.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio creado: {output_dir}")

    print(f"Descargando tokenizer para '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer guardado en {output_dir}")

    print(f"Descargando modelo '{model_name}'...")
    # Ajusta num_labels según tu tarea específica
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.save_pretrained(output_dir)
    print(f"Modelo guardado en {output_dir}")
    print("\n✅ Descarga completa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargar un modelo desde Hugging Face Hub.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo en el Hub (ej: 'dccuchile/bert-base-spanish-wwm-cased').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio para guardar el modelo (ej: 'models/BETO_local').")
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir)
```    
Lo ejecutarías en la terminal de tu máquina local así:

```Bash
# Activa tu entorno de conda local
conda activate tu-env-local

# Ejecuta el script
python src/download_model.py \
    --model_name "dccuchile/bert-base-spanish-wwm-cased" \
    --output_dir "models/BETO_local"
```
Esto creará una carpeta models/BETO_local con todos los archivos necesarios.

### 7.2. Paso 2: Transferir y Usar en el Clúster
Transferir: Usa un cliente SFTP como Bitvise (o scp desde la línea de comandos) para subir la carpeta completa (ej: models/BETO_local) a tu directorio de proyecto en el clúster.

Usar: Ahora, en tus scripts de lanzamiento (.sh), en lugar de pasar el nombre del hub de Hugging Face, pasas la ruta local a la carpeta que acabas de subir.

```Bash
# Ejemplo de cómo lanzar el trabajo en el clúster
# Nota cómo el primer argumento ahora es una ruta local
sbatch run_training.sh "models/BETO_local" "BETO_fine_tuned" 3
```
Lección Aprendida: Preparar los "ingredientes" (modelos, datasets) localmente antes de empezar a "cocinar" (entrenar) en el clúster es una práctica esencial que evita muchos dolores de cabeza.

