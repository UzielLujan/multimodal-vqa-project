#!/bin/bash
#SBATCH --job-name=vqa_infer_sample
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/multimodal-vqa-project 

set -e

mkdir -p logs

CONFIG_FILE=${1:-"configs/inference_config.yaml"}

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Inferencia SAMPLE VQA"
echo "Config: $CONFIG_FILE"
echo "========================================================"

#python -u src/evaluation/inference_sample.py --config "$CONFIG_FILE"

python -m src.evaluation.inference_sample --config "$CONFIG_FILE"

echo "========================================================"
echo "Inferencia SAMPLE completada."
echo "Archivo generado en: results/samples/"
echo "========================================================"
