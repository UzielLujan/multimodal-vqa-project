# src/evaluation/plot_metrics.py

"""
plot_metrics.py
--------------------

Genera gráficas a partir de:

    results/summary/summary_models.csv

Produce:

    results/plots/accuracy_by_model.png
    results/plots/bleu_by_model.png
    results/plots/accuracy_vs_bleu.png

Uso:
    python -m src.evaluation.plot_metrics --config configs/inference_config.yaml
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import get_path, check_path


# ---------------------------------------------------------
#  Loader de configuración YAML
# ---------------------------------------------------------
def load_config(config_path: str) -> dict:
    config_path = get_path(config_path)
    print(f"📖 Cargando configuración desde: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "paths" not in cfg or "results_dir" not in cfg["paths"]:
        raise KeyError("❌ El YAML debe contener paths.results_dir")

    return cfg


# ---------------------------------------------------------
#  Funciones de graficación
# ---------------------------------------------------------
def plot_accuracy_by_model(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["accuracy"], color="slateblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy por modelo (Yes/No)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"📊 Accuracy plot guardado en: {out_path}")


def plot_bleu_by_model(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["bleu"], color="darkgreen")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("BLEU")
    plt.title("BLEU por modelo")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"📊 BLEU plot guardado en: {out_path}")


def plot_accuracy_vs_bleu(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6, 6))

    plt.scatter(df["accuracy"], df["bleu"], s=80, color="firebrick")

    for _, row in df.iterrows():
        plt.text(row["accuracy"] + 0.002, row["bleu"] + 0.2, row["model"], fontsize=9)

    plt.xlabel("Accuracy")
    plt.ylabel("BLEU")
    plt.title("Comparación entre modelos: Accuracy vs BLEU")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"📊 Scatter Accuracy vs BLEU guardado en: {out_path}")


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generar gráficas a partir de summary_models.csv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Ruta al archivo YAML de configuración",
    )
    args = parser.parse_args()

    # 1. Cargar config
    cfg = load_config(args.config)

    results_dir = get_path(cfg["paths"]["results_dir"])
    summary_dir = results_dir / "summary"
    plots_dir = results_dir / "plots"

    check_path(summary_dir, is_dir=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summary_dir / "summary_models.csv"
    check_path(summary_csv, is_file=True)

    print(f"📂 Leyendo tabla resumen: {summary_csv}")
    df = pd.read_csv(summary_csv)

    # 2. Generar gráficas
    acc_path = plots_dir / "accuracy_by_model.png"
    bleu_path = plots_dir / "bleu_by_model.png"
    scatter_path = plots_dir / "accuracy_vs_bleu.png"

    plot_accuracy_by_model(df, acc_path)
    plot_bleu_by_model(df, bleu_path)
    plot_accuracy_vs_bleu(df, scatter_path)

    print("🎉 Todas las gráficas han sido generadas correctamente.")


if __name__ == "__main__":
    main()
