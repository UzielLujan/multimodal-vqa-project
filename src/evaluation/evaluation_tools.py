# src/evaluation/evaluation_tools.py

"""
evaluation_tools.py

Herramientas avanzadas para analizar predicciones VQA generadas
por inference_eval.py.

Incluye:
✓ Carga de CSV de predicciones
✓ Cálculo de métricas avanzadas:
     - accuracy_yesno
     - accuracy_general_flexible
     - keyword_accuracy
     - bleu_short
     - bleu
✓ Extracción de top-k errores
✓ Generación de reportes JSON
✓ Soporte para evaluation_batch.py y plot_metrics.py
"""

import json
from pathlib import Path
import pandas as pd
import yaml

from src.utils.paths import get_path, check_path
from src.evaluation.metrics import (
    compute_bleu,
    compute_bleu_short,
    compute_keyword_accuracy,
    compute_yesno_accuracy,
    compute_general_accuracy_flexible,
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
        raise KeyError("❌ El YAML debe incluir paths.results_dir")

    return cfg


# ---------------------------------------------------------
#  Cargar CSV
# ---------------------------------------------------------
def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    csv_path = get_path(csv_path)
    check_path(csv_path, is_file=True)

    print(f" Cargando predicciones desde: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"id", "question", "reference", "prediction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ El CSV debe contener las columnas: {required_cols}")

    return df


# ---------------------------------------------------------
#  Calcular métricas avanzadas
# ---------------------------------------------------------
def compute_all_metrics(df: pd.DataFrame) -> dict:
    preds = df["prediction"].astype(str).tolist()
    refs = df["reference"].astype(str).tolist()

    acc_yesno = compute_yesno_accuracy(preds, refs)
    acc_general = compute_general_accuracy_flexible(preds, refs)
    keyword_acc = compute_keyword_accuracy(preds, refs)
    bleu_short = compute_bleu_short(preds, refs, max_words=5)
    bleu_full = compute_bleu(preds, refs)

    return {
        "samples": len(df),
        "accuracy_yesno": acc_yesno,
        "accuracy_general_flexible": acc_general,
        "keyword_accuracy": keyword_acc,
        "bleu_short": bleu_short,
        "bleu": bleu_full,
    }


# ---------------------------------------------------------
#  Top-K errores
# ---------------------------------------------------------
def find_top_errors(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Identifica errores basados en keyword mismatch.
    Es robusta contra NaN, floats, None y cualquier tipo inesperado.
    """

    def is_error(row):
        ref = str(row["reference"]).lower().strip()
        pred = str(row["prediction"]).lower().strip()

        if ref in pred:
            return False
        return True

    df_errors = df[df.apply(is_error, axis=1)]
    return df_errors.head(k).copy()

# ---------------------------------------------------------
#  Guardar JSON
# ---------------------------------------------------------
def save_json(data: dict, output_path: str | Path):
    output_path = get_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f" JSON guardado en: {output_path}")


# ---------------------------------------------------------
#  Análisis completo para un CSV
# ---------------------------------------------------------
def analyze_predictions(csv_path: str | Path, config_path: str | Path):
    print("============================================================")
    print(" Análisis detallado de predicciones")
    print("============================================================")

    cfg = load_config(config_path)
    results_dir = get_path(cfg["paths"]["results_dir"])
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(csv_path)

    # ---- Métricas avanzadas ----
    metrics = compute_all_metrics(df)
    print("\n MÉTRICAS AVANZADAS:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # ---- Top-K errores ----
    k = cfg.get("evaluation", {}).get("top_k_errors", 20)
    df_errors = find_top_errors(df, k=k)

    # ---- Empaquetar reporte ----
    report = {
        "metrics": metrics,
        "n_samples": len(df),
        "top_k": k,
        "top_k_errors": df_errors.to_dict(orient="records"),
    }

    # ---- Guardar JSON ----
    model_name = Path(csv_path).stem.replace("predictions_", "")
    out_path = summary_dir / f"analysis_{model_name}.json"
    save_json(report, out_path)

    print(" Análisis completo.")
    return report
