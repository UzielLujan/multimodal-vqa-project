import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_training_loss(log_file: Path, save_path: Path):
    """
    Lee el log JSONL y genera una gráfica de la curva de aprendizaje.
    """
    if not log_file.exists():
        print(f"No se encontró archivo de log en {log_file}")
        return

    data = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not data:
        print("El archivo de log está vacío.")
        return

    df = pd.DataFrame(data)

    # Filtramos solo las filas que tengan 'loss' (a veces loggea evaluación sin loss)
    if 'loss' not in df.columns:
        print("No hay métrica 'loss' en los logs todavía.")
        return

    df_train = df.dropna(subset=['loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(df_train['step'], df_train['loss'], label='Training Loss', color='blue', linewidth=2)

    # Si hay eval_loss, graficarla también
    if 'eval_loss' in df.columns:
        df_eval = df.dropna(subset=['eval_loss'])
        if not df_eval.empty:
            plt.plot(df_eval['step'], df_eval['eval_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)

    plt.title(f"Curva de Entrenamiento - LLaVA PathVQA")
    plt.xlabel("Pasos (Steps)")
    plt.ylabel("Pérdida (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    print(f"Guardando gráfica de pérdida en: {save_path}")
    plt.savefig(save_path)
    plt.close()
