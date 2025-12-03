import os
from pathlib import Path

# Truco para encontrar la raíz: Estamos en src/utils/paths.py
# .parent -> src/utils
# .parent -> src
# .parent -> root (multimodal_vqa_project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_path(relative_path: str) -> Path:
    """Convierte una ruta relativa del config en un Path absoluto robusto."""
    return PROJECT_ROOT / relative_path

def check_path(path: Path, is_dir=True):
    """Verificación defensiva antes de crashear."""
    if not path.exists():
        raise FileNotFoundError(f"❌ La ruta crítica no existe: {path}")
    if is_dir and not path.is_dir():
        raise NotADirectoryError(f"❌ Se esperaba un directorio: {path}")
    return path