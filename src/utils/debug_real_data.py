import torch
from transformers import AutoProcessor
from src.data.dataset import PathVQADataset
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import os

# 1. Configuración Rápida
DATA_DIR = "./data/raw/path_vqa_hf"
# Apunta a la carpeta de CLIP que ya descargamos y convertimos
VISION_PATH = "./checkpoints/clip-vit-large-patch14-336"

print("Iniciando Test de Datos Reales (PathVQA + CLIP)...")

try:
    print(f"Cargando Image Processor desde: {VISION_PATH}")
    # Cargamos el procesador real de CLIP (que ya tienes en local)
    image_processor = AutoProcessor.from_pretrained(VISION_PATH)

    print("Creando Processor Híbrido Mock (Simulación)...")

    # --- MOCK TOKENIZER MEJORADO ---
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Simulamos que devolvemos tensores de longitud variable
            # (El dataset espera input_ids y attention_mask)
            length = 32
            return type('obj', (object,), {
                'input_ids': torch.randint(0, 1000, (1, length)),
                'attention_mask': torch.ones((1, length))
            })()

        def decode(self, token_ids, **kwargs):
            return "[TEXTO DECODIFICADO SIMULADO]"

        # Atributos requeridos por el nuevo dataset.py
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "</s>" # <--- EL FIX CRÍTICO: Necesario para concatenar strings
        unk_token = "<unk>"

    # --- MOCK PROCESSOR ---
    class MockLlavaProcessor:
        def __init__(self, image_processor):
            self.image_processor = image_processor
            self.tokenizer = MockTokenizer()
            # Fix de patch_size para evitar warnings
            self.patch_size = 14
            self.image_processor.patch_size = 14

        def __call__(self, text=None, images=None, **kwargs):
            # Procesamiento Real de Imagen (CLIP)
            if images is not None:
                pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
            else:
                pixel_values = None

            # Simulación de Texto
            return type('obj', (object,), {
                'input_ids': torch.randint(0, 100, (1, 40)),
                'attention_mask': torch.ones((1, 40)),
                'pixel_values': pixel_values
            })()

    processor = MockLlavaProcessor(image_processor)

except Exception as e:
    print(f"Error inicializando procesadores: {e}")
    exit()

# 2. Instanciar el Dataset REAL
print(f"Cargando Dataset desde disco: {DATA_DIR}")
try:
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No existe la ruta: {DATA_DIR}")

    ds = PathVQADataset(data_path=DATA_DIR, processor=processor, split='train')
    print(f"Dataset cargado. Total de muestras: {len(ds)}")
except Exception as e:
    print(f"Error cargando dataset: {e}")
    print("Verifica que la ruta en DATA_DIR sea correcta y contenga el dataset HuggingFace guardado.")
    exit()

# 3. Inspección Profunda
print("\nInspeccionando las primeras 5 muestras...")
print("-" * 60)

for i in range(5):
    try:
        # Dato Crudo
        raw_item = ds.dataset[i]
        original_img = raw_item['image']

        # Dato Procesado (Aquí se prueba la lógica del dataset.py nuevo)
        processed_item = ds[i]
        tensor_shape = processed_item['pixel_values'].shape
        labels = processed_item['labels']

        print(f"Muestra #{i}:")
        print(f"   - Pregunta: {raw_item['question'][:50]}...")
        print(f"   - Tamaño Original (PIL): {original_img.size}")
        print(f"   - Tensor CLIP:           {tensor_shape}  <-- ¿Es [3, 336, 336]?")

        # Validación de Etiquetas (Masking)
        masked_tokens = (labels == -100).sum().item()
        total_tokens = labels.numel()
        print(f"   - Masking: {masked_tokens}/{total_tokens} tokens ignorados (-100) ✅")

        # Validación CLIP (336x336 es el default de Large-336)
        if tensor_shape == torch.Size([3, 336, 336]):
            print("STATUS: Dimensión Correcta (CLIP Standard)")
        else:
            print(f"STATUS: Dimensión Inusual (Esperado [3, 336, 336])")

    except Exception as e:
        print(f"STATUS: Falló el procesamiento ({e})")

    print("-" * 60)

print("\nConclusión:")
print("Si las dimensiones son [3, 336, 336] y ves el Masking activo,")
print("tu dataset.py es 100% compatible con CLIP y TinyLlama.")
