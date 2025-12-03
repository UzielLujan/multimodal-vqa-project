#!/bin/bash
#SBATCH --job-name=vqa_fire_test      # Nombre del trabajo
#SBATCH --output=logs/slurm_%j.out    # Donde se guarda el log de salida (stdout)
#SBATCH --error=logs/slurm_%j.err     # Donde se guardan los errores (stderr)
#SBATCH --nodes=1                     # Numero de nodos
#SBATCH --ntasks=1                    # Numero de tareas
#SBATCH --cpus-per-task=8             # CPUs para procesar datos
#SBATCH --gres=gpu:1                  # Pide 1 GPU (Ajustar según clúster: a100:1, v100:1, etc.)
#SBATCH --time=02:00:00               # Tiempo límite (2 horas para la prueba)
#SBATCH --mem=32G                     # Memoria RAM del sistema

# 1. Cargar módulos del sistema (Esto depende del clúster, tu amigo sabrá cuáles)
# module load cuda/12.1
# module load anaconda3

# 2. Activar el entorno virtual
source activate vqa-env  # O la ruta a tu env

# 3. Debugging: Imprimir información de la GPU asignada
echo "Starting Fire Test : $(hostname)"
nvidia-smi

# 4. Ejecutar el script de entrenamiento
# Nota: python -u hace que los prints salgan en tiempo real en el log
python -u train.py \
    --output_dir ./results/llava-test-run \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2

echo "✅ Trabajo terminado."