#!/bin/bash
#SBATCH --job-name=vqa_data_debug_diego
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

# 4. Ejecutar (Desde la raíz, llamamos a src/...)
# Usamos python -u para ver el output en tiempo real en el log
python -u debug_real_data.py 

echo "Script completed."