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

def check_path(path: Path, is_dir=False, is_file=False):
    """
    Verificación defensiva antes de crashear.

    Args:
        path (Path | str): Ruta a verificar
        is_dir (bool): Si se espera un directorio
        is_file (bool): Si se espera un archivo

    Nota:
        - Si no se especifica is_dir ni is_file, solo verifica existencia general.
        - Mantiene compatibilidad con llamadas anteriores donde solo existía is_dir.
    """
    path = Path(path)

    # Verificar existencia general
    if not path.exists():
        raise FileNotFoundError(f"❌ La ruta crítica no existe: {path}")

    # Si se requiere archivo
    if is_file:
        if not path.is_file():
            raise FileNotFoundError(f"❌ Se esperaba un archivo: {path}")
        return path

    # Si se requiere directorio
    if is_dir:
        if not path.is_dir():
            raise NotADirectoryError(f"❌ Se esperaba un directorio: {path}")
        return path

    # Si no se especificó nada, simplemente devolver path existente
    return path
