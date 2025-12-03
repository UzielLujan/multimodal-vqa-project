#!/bin/bash
#SBATCH --job-name=vqa_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/multimodal-vqa-project  <-- ¡ACTUALIZA ESTO!

set -e  # El script se detiene si hay error, igual que en tu referencia

# 1. Preparar Logs
mkdir -p logs

# 2. Cargar Entorno (Igual que tu manual)
source ~/.bashrc
conda activate vqa-env

# 3. Variables y Configuración
# Puedes pasar el config como argumento, o usa el default
CONFIG_FILE=${1:-"configs/train_config.yaml"}

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "Config: $CONFIG_FILE"
echo "========================================================"

# 4. Ejecutar (Desde la raíz, llamamos a src/...)
# Usamos python -u para ver el output en tiempo real en el log
python -u train.py --config "$CONFIG_FILE"

echo "Entrenamiento completado."