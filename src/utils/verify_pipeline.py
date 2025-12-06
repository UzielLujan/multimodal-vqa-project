import torch
from src.models.model_factory import build_model_and_processor
from PIL import Image
import numpy as np

# Config Mock
mock_cfg = {
    'paths': {
        'llm_model_path': 'checkpoints/TinyLlama-1.1B-Chat',
        'vision_tower_path': 'checkpoints/clip-vit-large-patch14-336'
    },
    'training': {'gradient_checkpointing': False}
}

def check_gradients(model):
    print("\n   [Test] Inspeccionando gradientes...")
    # Buscamos el LLM donde sea que esté
    llm = getattr(model, "language_model", None) or getattr(model.model, "language_model", None)
    if not llm: return False
    
    # Buscamos capas
    layers = getattr(llm, "layers", None) or getattr(llm.model, "layers", None)
    if not layers: return False
        
    # Check primera capa
    grad = layers[0].self_attn.q_proj.weight.grad
    if grad is None:
        print("❌ FALLO: Gradientes vacíos.")
        return False
        
    print(f"✅ ÉXITO: Gradiente detectado (Media: {grad.abs().mean():.6f})")
    return True

def main():
    print(" Iniciando Smoke Test (Agnóstico)...")
    
    try:
        model, processor = build_model_and_processor(mock_cfg)
        model.train() 
    except Exception as e:
        print(f"❌ Error carga: {e}")
        return

    # ---  DETECCIÓN DE TIPO AUTOMÁTICA ---
    # Esto soluciona el error "RuntimeError: expected m1 and m2 to have same dtype"
    # Si estás en CPU, target_dtype será float32. Si GPU, bfloat16.
    target_dtype = model.language_model.dtype
    target_device = model.device
    print(f"\n  [Test] Adaptando inputs a: {target_device} | {target_dtype}")

    # Generar Datos
    text = "User: <image>\nDescribe.\nAssistant: Noise."
    dummy_image = Image.new('RGB', (336, 336), color='red')

    # Procesar
    inputs = processor(text=text, images=dummy_image, return_tensors="pt")
    
    # Mover y Castear
    inputs_final = {}
    for k, v in inputs.items():
        if k == "pixel_values":
            # La imagen se convierte al tipo exacto del modelo
            inputs_final[k] = v.to(target_device, dtype=target_dtype)
        else:
            inputs_final[k] = v.to(target_device)
            
    inputs_final["labels"] = inputs_final["input_ids"].clone()

    print("\n▶️ [Test] Forward Pass...")
    try:
        outputs = model(**inputs_final)
        print(f"✅ Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"❌ FALLO Forward: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n◀️ [Test] Backward Pass...")
    outputs.loss.backward()
    
    if check_gradients(model):
        print("\n   ¡PRUEBA SUPERADA! ")
        print("   Tu modelo está listo para el clúster.")

if __name__ == "__main__":
    main()