import os
import torch
from safetensors.torch import save_file
from transformers.utils import logging

# Configurar logging
logging.set_verbosity_info()

def convert_clip_weights():
    # Rutas
    base_path = os.path.join("checkpoints", "clip-vit-large-patch14-336")
    bin_file = os.path.join(base_path, "pytorch_model.bin")
    safe_file = os.path.join(base_path, "model.safetensors")
    
    print(f"🕵️‍♂️ Buscando archivo en: {bin_file}")
    
    if not os.path.exists(bin_file):
        print("❌ Error: No encuentro 'pytorch_model.bin'.")
        return

    if os.path.exists(safe_file):
        print("⚠️ El archivo safetensors ya existe. Lo sobrescribiremos para asegurar integridad.")

    print("⚙️  Cargando pesos originales (.bin)...")
    try:
        # Cargamos a CPU para no saturar VRAM
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)
        print(f"✅ Pesos cargados. Claves totales: {len(state_dict)}")
    except Exception as e:
        print(f"❌ Falló la carga: {e}")
        return

    print("🔧 Reparando tensores fragmentados (Forzando Contiguity)...")
    
    clean_state_dict = {}
    
    # Bucle de reparación
    for key, tensor in state_dict.items():
        # Forzamos .contiguous() en TODOS los tensores.
        # Esto crea una copia en memoria con los bits ordenados secuencialmente.
        # Es vital para que save_file no falle.
        clean_state_dict[key] = tensor.contiguous()

    print("🛡️  Guardando SAFETENSORS...")
    try:
        # Guardamos el diccionario limpio
        save_file(clean_state_dict, safe_file)
        print(f"✅ ¡ÉXITO! Archivo guardado en: {safe_file}")
        
        # Renombrar el viejo para forzar el uso del nuevo
        backup_file = bin_file + ".bak"
        if os.path.exists(bin_file):
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(bin_file, backup_file)
            print(f"📦 Archivo .bin renombrado a .bak")
            
    except Exception as e:
        print(f"❌ Error al guardar: {e}")

if __name__ == "__main__":
    convert_clip_weights()