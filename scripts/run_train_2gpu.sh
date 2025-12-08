#!/bin/bash
#SBATCH --job-name=vqa_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/multimodal-vqa-project

set -e

mkdir -p logs

#. Variables y Configuración
# Puedes pasar el config como argumento, o usa el default
CONFIG_FILE=${1:-"configs/train_config.yaml"}

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"
echo "Config: $CONFIG_FILE"
echo "========================================================"

echo "Iniciando entrenamiento con torchrun..."
torchrun --nproc_per_node=2 train.py --config "$CONFIG_FILE"

echo "Entrenamiento completado."
