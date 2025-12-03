import json
import os
from transformers import TrainerCallback
from pathlib import Path

class FileLoggerCallback(TrainerCallback):
    """
    Callback personalizado para guardar logs de entrenamiento en un archivo JSONL.
    Útil para entornos offline donde WandB no está disponible o para respaldo.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_file = output_dir / "training_logs.jsonl"
        
        # Asegurar que el directorio existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Iniciar archivo limpio o append? Mejor append para resume.
        print(f"[Callback] Los logs se guardarán en: {self.log_file}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Se ejecuta cada vez que el Trainer registra métricas (logging_steps)."""
        if logs:
            # Añadimos el epoch actual y el step global
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs
            }
            
            # Escribir línea JSON
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")