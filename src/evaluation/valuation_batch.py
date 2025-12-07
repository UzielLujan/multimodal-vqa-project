# src/evaluation/evaluation_batch.py

"""
evaluation_batch.py
--------------------

Este módulo busca automáticamente todos los archivos:
    results/predictions/predictions_*.csv

Calcula métricas globales para cada uno usando evaluation_tools.py
y genera una tabla resumen en:

    results/summary/summary_models.csv

Además genera un JSON:

    results/summary/summary_models.json

Uso:
    python -m src.evaluation.evaluation_batch --config configs/inference_config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import pandas as pd

from src.utils.paths import get_path, check_path
from src.evaluation.evaluation_tools import (
    compute_global_metrics,
    load_predictions,
)


# ---------------------------------------------------------
#  Cargar config YAML
# ---------------------------------------------------------
def load_config(config_path: str) -> dict:
    config_path = get_path(config_path)
    print(f"📖 Cargando configuración desde: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "paths" not in cfg or "results_dir" not in cfg["paths"]:
        raise KeyError("❌ El archivo YAML debe contener paths.results_dir")

    return cfg


# ---------------------------------------------------------
#  Extraer nombre del modelo y split desde el nombre del CSV
#  Ejemplo:
#   predictions_tinyllama-clip-768_test.csv
#   → model_name = tinyllama-clip-768
#   → split = test
# ---------------------------------------------------------
def parse_filename(csv_path: Path):
    stem = csv_path.stem  # predictions_tinyllama-clip-768_test
    parts = stem.split("_", maxsplit=2)

    if len(parts) < 3:
        raise ValueError(f"No se puede interpretar el nombre del archivo: {stem}")

    _, model_name, split = parts
    return model_name, split


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comparación automática de múltiples modelos VQA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Ruta al archivo YAML de configuración",
    )
    args = parser.parse_args()

    # 1. Cargar configuración
    cfg = load_config(args.config)

    results_dir = get_path(cfg["paths"]["results_dir"])
    predictions_dir = results_dir / "predictions"
    summary_dir = results_dir / "summary"

    check_path(results_dir, is_dir=True)
    check_path(predictions_dir, is_dir=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 2. Buscar predicciones existentes
    csv_files = sorted(predictions_dir.glob("predictions_*.csv"))

    if not csv_files:
        print("❌ No se encontraron archivos en results/predictions/predictions_*.csv")
        return

    print("============================================================")
    print(f"🔎 Encontrados {len(csv_files)} archivos de predicciones:")
    for f in csv_files:
        print(f"   • {f.name}")
    print("============================================================")

    # 3. Procesar cada archivo y calcular métricas
    summary_rows = []

    for csv_path in csv_files:
        print(f"\n📂 Procesando: {csv_path.name}")

        # Extraer modelo y split desde el nombre del archivo
        model_name, split = parse_filename(csv_path)

        # Cargar predicciones
        df = load_predictions(csv_path)

        # Calcular métricas globales
        metrics = compute_global_metrics(df)

        print(f"   ✔ Samples : {metrics['samples']}")
        print(f"   ✔ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✔ BLEU    : {metrics['bleu']:.4f}")

        summary_rows.append({
            "model": model_name,
            "split": split,
            "samples": metrics["samples"],
            "accuracy": metrics["accuracy"],
            "bleu": metrics["bleu"],
            "csv_path": str(csv_path),
        })

    # 4. Crear DataFrame resumen
    summary_df = pd.DataFrame(summary_rows)

    # Orden sugerido: split, accuracy desc, BLEU desc
    summary_df = summary_df.sort_values(by=["split", "accuracy", "bleu"], ascending=[True, False, False])

    # 5. Guardar CSV
    summary_csv = summary_dir / "summary_models.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n============================================================")
    print("📊 TABLA RESUMEN DE MODELOS")
    print(summary_df)
    print("============================================================")
    print(f"💾 Archivo guardado en: {summary_csv}")

    # 6. Guardar JSON equivalente
    summary_json = summary_dir / "summary_models.json"
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=4)

    print(f"💾 JSON guardado en: {summary_json}")
    print("🎉 Comparación de modelos completada.")


if __name__ == "__main__":
    main()
