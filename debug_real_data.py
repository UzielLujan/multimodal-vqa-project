import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaProcessor
from src.data.dataset import PathVQADataset
from src.utils.paths import get_path
import matplotlib.pyplot as plt

# 1. Configuración Rápida
# Apuntamos a tus carpetas locales reales
DATA_DIR = "./data/raw/path_vqa_hf" 
VISION_PATH = "./checkpoints/siglip_vision_tower"
# Usamos un modelo dummy para el tokenizer si no tienes LLaMA en local,
# o usamos la ruta real si la tienes. Si no, el procesador de SigLIP basta para la imagen.
# Para este test, intentaremos cargar el procesador de visión puro.

print("🚀 Iniciando Test de Datos Reales (PathVQA)...")

try:
    # Cargamos solo la parte de visión para no necesitar LLaMA pesado
    print(f"📂 Cargando Image Processor desde: {VISION_PATH}")
    image_processor = AutoProcessor.from_pretrained(VISION_PATH).image_processor
    
    # Creamos un tokenizer "fake" o básico solo para que la clase Dataset no falle
    # (PathVQADataset espera un 'processor' que tenga .tokenizer y .image_processor)
    print("🔧 Creando Processor Híbrido Mock para el test...")
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Simula tokenización
            return type('obj', (object,), {
                'input_ids': torch.randint(0, 1000, (1, 32)),
                'attention_mask': torch.ones((1, 32))
            })()
        def decode(self, token_ids, **kwargs):
            return "[TEXTO DECODIFICADO SIMULADO]"
            
        pad_token_id = 0
        eos_token_id = 1

    # Clase wrapper para simular el LlavaProcessor completo
    class MockLlavaProcessor:
        def __init__(self, image_processor):
            self.image_processor = image_processor
            self.tokenizer = MockTokenizer()
            
        def __call__(self, text=None, images=None, **kwargs):
            # Esta es la magia: Usamos el procesador REAL de imagen
            pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
            
            # Simulación de texto
            return type('obj', (object,), {
                'input_ids': torch.randint(0, 100, (1, 10)),
                'attention_mask': torch.ones((1, 10)),
                'pixel_values': pixel_values
            })()

    processor = MockLlavaProcessor(image_processor)

except Exception as e:
    print(f"❌ Error inicializando procesadores: {e}")
    exit()

# 2. Instanciar el Dataset REAL
print(f"📂 Cargando Dataset desde disco: {DATA_DIR}")
try:
    # Instanciamos tu clase real
    ds = PathVQADataset(data_path=DATA_DIR, processor=processor, split='train')
    print(f"✅ Dataset cargado. Total de muestras: {len(ds)}")
except Exception as e:
    print(f"❌ Error cargando dataset: {e}")
    print("⚠️  Asegúrate de haber ejecutado 'setup_data.py' o el notebook de descarga primero.")
    exit()

# 3. Inspección Profunda de Muestras
print("\n🔎 Inspeccionando las primeras 5 muestras...")
print("-" * 60)

for i in range(5):
    # Accedemos al dato crudo primero para ver tamaño original
    raw_item = ds.dataset[i]
    original_img = raw_item['image']
    original_size = original_img.size # (W, H)
    
    # Ahora pasamos por el __getitem__ del dataset que usa el procesador
    processed_item = ds[i]
    tensor_shape = processed_item['pixel_values'].shape
    
    print(f"Muestra #{i}:")
    print(f"   - Pregunta: {raw_item['question'][:50]}...")
    print(f"   - Tamaño Original (PIL): {original_size} (Ancho x Alto)")
    print(f"   - Tensor Resultante:     {tensor_shape}  <-- ¿Es [3, 384, 384]?")
    
    # Validación automática
    if tensor_shape == torch.Size([3, 384, 384]):
        print("   ✅ STATUS: Correcto (Redimensionado OK)")
    else:
        print("   ❌ STATUS: Fallo de dimensión")
    
    print("-" * 60)

print("\n🎉 Conclusión del Test:")
print("Si viste ✅ en todas las muestras, el procesador está manejando")
print("automáticamente la variabilidad de tamaños de PathVQA.")