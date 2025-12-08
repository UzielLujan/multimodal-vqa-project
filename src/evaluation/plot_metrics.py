# src/evaluation/plot_metrics.py

"""
plot_metrics.py

Genera gráficas avanzadas a partir de:
    results/summary/summary_models.csv

Incluye:
    ✓ Accuracy flexible por modelo
    ✓ Keyword accuracy por modelo
    ✓ Accuracy yes/no por modelo
    ✓ BLEU-short por modelo
    ✓ BLEU clásico por modelo
    ✓ Scatter plots comparativos

Uso:
    python -m src.evaluation.plot_metrics --config configs/inference_config.yaml
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import get_path, check_path


# ---------------------------------------------------------
#  Leer config YAML
# ---------------------------------------------------------
def load_config(config_path: str):
    config_path = get_path(config_path)
    print(f" Cargando configuración desde: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "paths" not in cfg or "results_dir" not in cfg["paths"]:
        raise KeyError("❌ Falta paths.results_dir en el YAML")

    return cfg


# ---------------------------------------------------------
#  Funciones de graficación
# ---------------------------------------------------------
def plot_bar(df, column, ylabel, title, out_path):
    plt.figure(figsize=(10, 4))
    plt.bar(df["model"], df[column], color="slateblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f" Guardado: {out_path}")


def plot_scatter(df, x_col, y_col, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(df[x_col], df[y_col], s=90, color="darkgreen")

    # Etiquetas de cada punto
    for _, row in df.iterrows():
        plt.text(
            row[x_col] + 0.002,
            row[y_col] + 0.002,
            row["model"],
            fontsize=9
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f" Guardado: {out_path}")


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Graficación de métricas VQA")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config) 

    results_dir = get_path(cfg["paths"]["results_dir"])
    summary_dir = results_dir / "summary"
    plots_dir = results_dir / "plots"

    check_path(summary_dir, is_dir=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summary_dir / "summary_models.csv"
    check_path(summary_csv, is_file=True)

    print(f" Leyendo resumen: {summary_csv}")
    df = pd.read_csv(summary_csv)

    # -----------------------------------------------------
    # 2. Gráficas de barras individuales por métrica
    # -----------------------------------------------------
    plot_bar(
        df=df,
        column="accuracy_general_flexible",
        ylabel="Accuracy Flexible",
        title="Accuracy Flexible por Modelo (PathVQA)",
        out_path=plots_dir / "accuracy_flexible.png"
    )

    plot_bar(
        df=df,
        column="keyword_accuracy",
        ylabel="Keyword Accuracy",
        title="Keyword Accuracy por Modelo",
        out_path=plots_dir / "keyword_accuracy.png"
    )

    plot_bar(
        df=df,
        column="accuracy_yesno",
        ylabel="Accuracy Yes/No",
        title="Accuracy Yes/No por Modelo",
        out_path=plots_dir / "accuracy_yesno.png"
    )

    plot_bar(
        df=df,
        column="bleu_short",
        ylabel="BLEU-short (5 palabras)",
        title="BLEU-short por Modelo",
        out_path=plots_dir / "bleu_short.png"
    )

    plot_bar(
        df=df,
        column="bleu",
        ylabel="BLEU",
        title="BLEU clásico por Modelo",
        out_path=plots_dir / "bleu.png"
    )

    # -----------------------------------------------------
    # 3. Scatter plots comparativos
    # -----------------------------------------------------
    plot_scatter(
        df=df,
        x_col="accuracy_general_flexible",
        y_col="bleu_short",
        xlabel="Accuracy Flexible",
        ylabel="BLEU-short",
        title="Accuracy Flexible vs BLEU-short",
        out_path=plots_dir / "scatter_flexible_vs_bleu_short.png"
    )

    plot_scatter(
        df=df,
        x_col="accuracy_general_flexible",
        y_col="keyword_accuracy",
        xlabel="Accuracy Flexible",
        ylabel="Keyword Accuracy",
        title="Accuracy Flexible vs Keyword Accuracy",
        out_path=plots_dir / "scatter_flexible_vs_keyword.png"
    )

    print(" Todas las gráficas generadas correctamente.")


if __name__ == "__main__":
    main()
