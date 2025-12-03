import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pathlib import Path

def build_model_and_processor(cfg):
    """
    Construye el modelo LLaVA y el procesador basado en la configuración.
    Args:
        cfg (dict): Diccionario cargado del yaml (sección paths y training).
    """
    from src.utils.paths import get_path
    
    # Resolver rutas absolutas
    llm_path = get_path(cfg['paths']['llm_model_path']) # Ojo aquí en el cluster
    vision_path = get_path(cfg['paths']['vision_tower_path'])
    
    print(f"🏗️ [Model Factory] Cargando LLaMA desde: {llm_path}")
    print(f"🏗️ [Model Factory] Cargando SigLIP desde: {vision_path}")

    # 1. Configuración de Cuantización (BNB)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Cargar Procesador
    # Usamos el LLM para el tokenizador y SigLIP para la imagen
    processor = AutoProcessor.from_pretrained(str(llm_path))
    processor.image_processor = AutoProcessor.from_pretrained(str(vision_path)).image_processor

    # 3. Cargar Modelo Base
    model = LlavaForConditionalGeneration.from_pretrained(
        str(llm_path),
        vision_tower_address=str(vision_path),
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 4. Preparar LoRA
    print("🔧 [Model Factory] Aplicando adaptadores LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "mm_projector"],
        lora_dropout=cfg['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, processor