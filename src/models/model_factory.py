import torch
from transformers import (
    LlavaConfig,
    LlavaForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    LlavaProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.paths import get_path

def build_model_and_processor(cfg):
    """
    Construye un modelo LLaVA ensamblando LLaMA-3 (Texto) y SigLIP (Visión).
    """
    # 1. Resolver rutas
    llm_path = str(get_path(cfg['paths']['llm_model_path']))
    vision_path = str(get_path(cfg['paths']['vision_tower_path']))
    
    print(f"[Factory] Cargando configs de: \n   - LLM: {llm_path}\n   - Vision: {vision_path}")

    # 2. Cargar Configuraciones
    text_config = AutoConfig.from_pretrained(llm_path)
    siglip_full_config = AutoConfig.from_pretrained(vision_path)
    vision_config = siglip_full_config.vision_config 
    
    print(f"   -> Vision Config Hidden Size: {vision_config.hidden_size}")

    # 3. Crear Configuración LLaVA Híbrida
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=128257, 
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        pad_token_id=128001
    )

    # 4. Tokenizer y Processor
    print("[Factory] Ajustando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    image_processor = AutoProcessor.from_pretrained(vision_path).image_processor
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 5. Instanciar LLaVA (Esqueleto)
    print("[Factory] Instanciando esqueleto LLaVA...")
    model = LlavaForConditionalGeneration(llava_config)

    # 6. INYECCIÓN DE PESOS 💉
    
    # A) LLaMA-3 (4-bit)
    print("[Factory] Inyectando LLaMA-3 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path, 
        quantization_config=bnb_config, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.language_model = llm_model
    model.resize_token_embeddings(len(tokenizer))

    # B) SigLIP (Vision Tower)
    print("[Factory] Inyectando SigLIP...")
    siglip_full_model = AutoModel.from_pretrained(vision_path, torch_dtype=torch.float16)
    model.vision_tower = siglip_full_model.vision_model 
    
    del siglip_full_model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # 7. Configuración LoRA (FIX CRÍTICO AQUÍ) 🛠️
    print("[Factory] Configurando LoRA...")
    
    # Primero preparamos para kbit training (congela capas, prepara layernorm)
    model = prepare_model_for_kbit_training(model)

    # Definimos la config de LoRA
    # IMPORTANTE: target_modules debe apuntar a los nombres reales dentro de language_model
    # En LLaMA-3 suelen ser: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # También queremos entrenar el 'multi_modal_projector'
    
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj", 
            "multi_modal_projector" # Incluimos el proyector para que aprenda
        ],
        lora_dropout=cfg['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicamos LoRA al modelo COMPLETO (wrapper LLaVA)
    # El truco es que PEFT buscará los target_modules recursivamente.
    model = get_peft_model(model, peft_config)
    
    # Aseguramos que el proyector sea entrenable (a veces LoRA lo ignora si no es Linear)
    # Aunque al ponerlo en target_modules, LoRA le pondrá adaptadores.
    # Alternativa: No poner adaptadores al proyector y descongelarlo full.
    # Por ahora, probamos full LoRA en todo (incluido proyector) que es más estable.

    model.print_trainable_parameters()
    
    return model, processor