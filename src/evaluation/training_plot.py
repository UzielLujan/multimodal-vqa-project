import json
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def plot_training_progress(log_file, output_img):
    if not os.path.exists(log_file):
        print(f"No se encontró el archivo de logs en: {log_file}")
        return

    steps = []
    train_loss = []
    val_loss = []
    val_steps = []

    print(f"Leyendo logs desde: {log_file}...")

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)

                # Extraer Training Loss
                if 'loss' in entry and 'step' in entry:
                    steps.append(entry['step'])
                    train_loss.append(entry['loss'])

                # Extraer Validation Loss (si existe)
                if 'eval_loss' in entry and 'step' in entry:
                    val_steps.append(entry['step'])
                    val_loss.append(entry['eval_loss'])

            except json.JSONDecodeError:
                continue

    if not steps:
        print("El archivo de logs está vacío o no tiene entradas válidas aún.")
        return

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # Curva de Entrenamiento (Suavizada visualmente si hay muchos puntos)
    plt.plot(steps, train_loss, label='Training Loss', color="#175fd3", alpha=0.6, linewidth=2)

    # Curva de Validación (Puntos grandes rojos)
    if val_loss:
        plt.plot(val_steps, val_loss, label='Validation Loss', color='#ef4444', marker='o', linestyle='-', linewidth=2)
        # Anotar el último valor
        plt.annotate(f"{val_loss[-1]:.4f}",
                     (val_steps[-1], val_loss[-1]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     color='#ef4444',
                     fontweight='bold')
        # Marcar los finales de época con etiquetas mejoradas
        ymin, ymax = plt.ylim()
        y_text = ymin + (ymax - ymin) * 0.2  # 20% arriba del mínimo
        for i, step in enumerate(val_steps):
            plt.axvline(x=step, color='#888888', linestyle='--', alpha=0.5, linewidth=1)
            plt.text(step, y_text, f"Epoch {i+1}", color='#888888', fontsize=10, ha='right', va='bottom', rotation=0, alpha=0.9, fontweight='bold', backgroundcolor='white')
    else:
        plt.text(0.5, 0.5, "Esperando fin de epoch para Validación...",
                 transform=plt.gca().transAxes, ha='center', alpha=0.5)

    plt.title(f"Curvas de Aprendizaje VQA \n Modelo TinyLlama-CLIP-768", fontsize=16)
    plt.xlabel("Pasos (Steps)")
    plt.ylabel("Pérdida (Loss)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Guardar
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f" Gráfica guardada en: {output_img}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="results/tinyllama-clip-768/training_logs_clean.jsonl", help="Ruta al jsonl")
    parser.add_argument("--out", type=str, default="results/tinyllama-clip-768/training_plot.png", help="Nombre de la imagen de salida")
    args = parser.parse_args()

    plot_training_progress(args.logs, args.out)
