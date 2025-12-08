# src/evaluation/analyze_single.py

"""
analyze_single.py

Script simple y directo para analizar UN modelo VQA usando las herramientas
del módulo evaluation_tools.py.

En vez de pasar argumentos por CLI, configura aquí:

    CSV_PATH   → archivo de predicciones a analizar
    CONFIG_PATH → archivo YAML con rutas y parámetros globales

Luego ejecuta simplemente:

    python -m src.evaluation.analyze_single

Y se generará automáticamente un archivo JSON con:
    - métricas avanzadas 
    - top-k errores
    - información agregada

El archivo final queda guardado en:
    results/summary/analysis_<modelo>_<split>.json
"""

from pathlib import Path
from src.evaluation.evaluation_tools import analyze_predictions

# -------------------------------------------------------------------
# 1. CONFIGURA AQUÍ LOS ARCHIVOS A ANALIZAR
# -------------------------------------------------------------------
# Ejemplo:
# CSV_PATH = "results/predictions/predictions_tinyllama-clip-768_validation.csv"

CSV_PATH = "results/predictions/predictions_tinyllama-clip-1024_validation.csv"
CONFIG_PATH = "configs/inference_config.yaml"


# -------------------------------------------------------------------
# 2. EJECUTAR ANÁLISIS
# -------------------------------------------------------------------
def main():
    print("===================================================")
    print(" ANALIZANDO MODELO INDIVIDUAL (analyze_single.py)")
    print("===================================================\n")

    csv_path = Path(CSV_PATH)
    config_path = Path(CONFIG_PATH)

    print(f" Usando CSV     : {csv_path}")
    print(f" Usando Config  : {config_path}\n")

    analyze_predictions(csv_path, config_path)

if __name__ == "__main__":
    main()
