import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# --- CONFIGURACIÓN DE RUTAS ROBUSTA ---
# Detectar la ruta de ESTE archivo y subir un nivel para encontrar la raíz del proyecto
# Asumiendo estructura: proyecto/scripts/download_llm.py
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Definir ruta de destino absoluta
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_DIR = CHECKPOINTS_DIR / "TinyLlama-1.1B-Chat"

def download_llm():
    print(f" Raíz del proyecto detectada en: {PROJECT_ROOT}")
    print(f" Iniciando descarga de TinyLlama en: {LLM_DIR}")

    # Asegurar que el directorio existe
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Descargar Pesos y Configs
    snapshot_download(
        repo_id=LLM_ID,
        local_dir=str(LLM_DIR), # Convertir Path a str para huggingface
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax*", "tf*"]
    )

    # 2. Asegurar Tokenizer
    print("   Verificando Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        tokenizer.save_pretrained(str(LLM_DIR))
        print("Tokenizer verificado y guardado.")
    except Exception as e:
        print(f"Nota sobre tokenizer: {e}")

    print("\n¡Listo! LLM descargado correctamente en la carpeta del proyecto.")

if __name__ == "__main__":
    download_llm()
