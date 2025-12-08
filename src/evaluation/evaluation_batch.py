# src/evaluation/evaluation_batch.py

"""
evaluation_batch.py

Evalúa múltiples modelos automáticamente leyendo todos los archivos:
    results/predictions/predictions_*.csv

Calcula métricas avanzadas:
    - accuracy_yesno
    - accuracy_general_flexible
    - keyword_accuracy
    - bleu_short
    - bleu

Produce:
    results/summary/summary_models.csv
    results/summary/summary_models.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

from src.utils.paths import get_path, check_path
from src.evaluation.evaluation_tools import (
    load_predictions,
    compute_all_metrics,
)


# ---------------------------------------------------------
#  Cargar configuración YAML
# ---------------------------------------------------------
def load_config(config_path: str) -> dict:
    config_path = get_path(config_path)
    print(f" Cargando configuración desde: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "paths" not in cfg or "results_dir" not in cfg["paths"]:
        raise KeyError("El archivo YAML debe incluir paths.results_dir")

    return cfg


# ---------------------------------------------------------
#  Extraer modelo y split desde nombre del archivo
# ---------------------------------------------------------
def parse_filename(csv_path: Path):
    """
    predictions_tinyllama-clip-768_test.csv
        → model_name = tinyllama-clip-768
        → split = test
    """
    stem = csv_path.stem  # predictions_model_split
    parts = stem.split("_", maxsplit=2)

    if len(parts) < 3:
        raise ValueError(f"No se puede interpretar nombre de archivo: {stem}")

    _, model_name, split = parts
    return model_name, split


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comparación automática de múltiples modelos VQA")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # 1. Cargar configuración
    cfg = load_config(args.config)
    results_dir = get_path(cfg["paths"]["results_dir"])

    predictions_dir = results_dir / "predictions"
    summary_dir = results_dir / "summary"

    check_path(predictions_dir, is_dir=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 2. Buscar archivos CSV de predicciones
    csv_files = sorted(predictions_dir.glob("predictions_*.csv"))

    if not csv_files:
        print("No se encontraron predicciones en results/predictions/*.csv")
        return

    print("============================================================")
    print(" Encontrados archivos de predicciones:")
    for f in csv_files:
        print(f"   • {f.name}")
    print("============================================================")

    # 3. Procesar CSV uno por uno
    summary_rows = []

    for csv_path in csv_files:
        print(f"\n Procesando archivo: {csv_path.name}")

        # Nombres desde el archivo
        model_name, split = parse_filename(csv_path)

        # Cargar predicciones
        df = load_predictions(csv_path)

        # Calcular TODAS las métricas nuevas
        metrics = compute_all_metrics(df)

        print(f"   ✔ Samples                  : {metrics['samples']}")
        print(f"   ✔ Accuracy Yes/No          : {metrics['accuracy_yesno']:.4f}")
        print(f"   ✔ Accuracy Flexible        : {metrics['accuracy_general_flexible']:.4f}")
        print(f"   ✔ Keyword Accuracy         : {metrics['keyword_accuracy']:.4f}")
        print(f"   ✔ BLEU-short               : {metrics['bleu_short']:.4f}")
        print(f"   ✔ BLEU                     : {metrics['bleu']:.4f}")

        # Registrar fila del resumen
        summary_rows.append({
            "model": model_name,
            "split": split,
            "samples": metrics["samples"],
            "accuracy_yesno": metrics["accuracy_yesno"],
            "accuracy_general_flexible": metrics["accuracy_general_flexible"],
            "keyword_accuracy": metrics["keyword_accuracy"],
            "bleu_short": metrics["bleu_short"],
            "bleu": metrics["bleu"],
            "csv_path": str(csv_path),
        })

    # 4. Crear tabla resumen
    summary_df = pd.DataFrame(summary_rows)

    # Orden: primero por split, luego por accuracy flexible, luego keyword acc.
    summary_df = summary_df.sort_values(
        by=["split", "accuracy_general_flexible", "keyword_accuracy"],
        ascending=[True, False, False]
    )

    # 5. Guardar CSV
    summary_csv = summary_dir / "summary_models.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n Tabla resumen guardada en: {summary_csv}")

    # 6. Guardar JSON
    summary_json = summary_dir / "summary_models.json"
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=4)

    print(f" Resumen JSON guardado en: {summary_json}")
    print(" Comparación entre modelos completada.")


if __name__ == "__main__":
    main()
