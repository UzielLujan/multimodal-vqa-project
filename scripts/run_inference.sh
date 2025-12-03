#!/bin/bash
#SBATCH --job-name=vqa_infer
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/multimodal-vqa-project  <-- ¡ACTUALIZA ESTO!

set -e

mkdir -p logs
source ~/.bashrc
conda activate vqa-env

CONFIG_FILE=${1:-"configs/train_config.yaml"}

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Inferencia VQA"
echo "========================================================"

python -u inference.py --config "$CONFIG_FILE"

echo "Inferencia completada."