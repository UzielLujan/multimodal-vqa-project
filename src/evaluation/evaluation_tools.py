# src/evaluation/evaluation_tools.py

"""
Herramientas auxiliares para analizar predicciones generadas en inference_eval.py.

Incluye:
- carga de CSV de predicciones
- cálculo de métricas globales
- análisis por tipo de pregunta
- extracción de casos fallidos (top-k)
- guardado de resúmenes en JSON
"""

import json
from pathlib import Path

import yaml
import pandas as pd

from src.utils.paths import get_path, check_path
from src.evaluation.metrics import compute_vqa_accuracy, compute_bleu


# ---------------------------------------------------------
#  Carga de configuración YAML
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
#  Cargar CSV de predicciones
# ---------------------------------------------------------
def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    csv_path = get_path(csv_path)
    check_path(csv_path, is_file=True)

    print(f"📂 Cargando predicciones desde: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"id", "question", "reference", "prediction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ El CSV debe contener columnas: {required_cols}")

    return df


# ---------------------------------------------------------
#  Métricas globales (Accuracy + BLEU)
# ---------------------------------------------------------
def compute_global_metrics(df: pd.DataFrame) -> dict:
    preds = df["prediction"].astype(str).tolist()
    refs = df["reference"].astype(str).tolist()

    acc = compute_vqa_accuracy(preds, refs)
    bleu = compute_bleu(preds, refs)

    return {
        "samples": len(df),
        "accuracy": acc,
        "bleu": bleu,
    }


# ---------------------------------------------------------
#  Separación de preguntas Yes/No vs preguntas abiertas
# ---------------------------------------------------------
def split_by_question_type(df: pd.DataFrame) -> dict:
    """
    Divide el dataset en:
    - yes/no (cuando reference es 'yes' o 'no')
    - open (preguntas abiertas)
    """
    df_yesno = df[df["reference"].str.lower().isin(["yes", "no"])]
    df_open = df[~df["reference"].str.lower().isin(["yes", "no"])]

    return {
        "yesno": df_yesno,
        "open": df_open,
    }


# ---------------------------------------------------------
#  Análisis de errores (top-k discrepancias)
# ---------------------------------------------------------
def find_top_errors(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Encuentra los casos donde prediction != reference.
    Devuelve los primeros k errores.
    """
    df_errors = df[df["prediction"].str.strip().str.lower() != df["reference"].str.strip().str.lower()]
    return df_errors.head(k).copy()


# ---------------------------------------------------------
#  Guardar resumen en JSON
# ---------------------------------------------------------
def save_metrics_json(metrics: dict, output_path: str | Path):
    output_path = get_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"💾 Resumen JSON guardado en: {output_path}")


# ---------------------------------------------------------
#  Flujo completo para análisis de un CSV
# ---------------------------------------------------------
def analyze_predictions(csv_path: str | Path, config_path: str | Path):
    """
    Pipeline de análisis:
    - cargar config
    - cargar CSV
    - calcular métricas globales
    - dividir por tipo de pregunta
    - extraer top-k errores
    - guardar resumen
    """
    print("============================================================")
    print("🔎 ANALIZANDO PREDICCIONES")
    print("============================================================")

    cfg = load_config(config_path)
    df = load_predictions(csv_path)

    results_dir = get_path(cfg["paths"]["results_dir"])
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Métricas globales
    global_metrics = compute_global_metrics(df)
    print("\n📊 MÉTRICAS GLOBALES:")
    print(global_metrics)

    # Yes/No vs Open
    split_types = split_by_question_type(df)
    print(f"\n• Casos yes/no: {len(split_types['yesno'])}")
    print(f"• Casos open  : {len(split_types['open'])}")

    # Errores principales
    top_k = cfg.get("evaluation", {}).get("top_k_errors", 20)
    df_errors = find_top_errors(df, k=top_k)

    # Crear objeto del reporte final
    report = {
        "global_metrics": global_metrics,
        "n_samples": len(df),
        "n_yesno": len(split_types["yesno"]),
        "n_open": len(split_types["open"]),
        "top_k_errors": df_errors.to_dict(orient="records"),
    }

    # Guardar JSON
    model_name = Path(csv_path).stem.replace("predictions_", "")
    output_json = summary_dir / f"analysis_{model_name}.json"
    save_metrics_json(report, output_json)

    print("\n🎉 Análisis completado correctamente.")
    return report


'''
Cómo usar:
Supongamos que ya corriste:

python -m src.evaluation.inference_eval --config configs/inference_config.yaml

Y generaste un CSV de predicciones en:

results/predictions/predictions_tinyllama-clip-768_test.csv

Entonces puedes analizarlo así:

python -c "from src.evaluation.evaluation_tools import analyze_predictions; analyze_predictions('results/predictions/predictions_tinyllama-clip-768_test.csv', 'configs/inference_config.yaml')"

python -m src.evaluation.evaluation_tools --config configs/inference_config.yaml --csv results/predictions/predictions_model_test.csv



Esto generará un JSON resumen en:
results/summary/analysis_tinyllama-clip-768_test.json

'''