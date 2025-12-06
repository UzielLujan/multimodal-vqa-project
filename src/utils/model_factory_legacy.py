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
from accelerate import init_empty_weights
from src.utils.paths import get_path
import gc

def build_model_and_processor(cfg):
    """
    Construye un modelo LLaVA ensamblando LLaMA-3 (Texto) y SigLIP (Visión).
    Fusión Final: Token ID Dinámico + Carga Segura + LoRA Correcto (Diego).
    """
    # 1. Resolver rutas
    llm_path = str(get_path(cfg['paths']['llm_model_path']))
    vision_path = str(get_path(cfg['paths']['vision_tower_path']))
    
    print(f"🏗️ [Factory] Cargando rutas:\n   - LLM: {llm_path}\n   - Vision: {vision_path}")

    # -------------------------------------------------------------------------
    # PASO 1: Configurar Tokenizer y obtener ID real (Corrección Uzi)
    # -------------------------------------------------------------------------
    print("🔧 [Factory] Configurando Tokenizer y IDs...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    # Obtener ID dinámico para evitar hardcoding
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    print(f"   -> Token <image> asignado al ID: {image_token_index}")

    # 2. Cargar Configuraciones
    text_config = AutoConfig.from_pretrained(llm_path)
    siglip_full_config = AutoConfig.from_pretrained(vision_path)
    vision_config = siglip_full_config.vision_config 

    # 3. Configuración LLaVA
    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=image_token_index, # <--- ID Correcto
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        pad_token_id=tokenizer.pad_token_id
    )

    # 4. Processor
    image_processor = AutoProcessor.from_pretrained(vision_path).image_processor
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # -------------------------------------------------------------------------
    # 5. Instanciar LLaVA (Esqueleto VACÍO)
    # -------------------------------------------------------------------------
    print("💀 [Factory] Instanciando esqueleto LLaVA (init_empty_weights)...")
    with init_empty_weights():
        model = LlavaForConditionalGeneration(llava_config)
    
    model.tie_weights()

    # -------------------------------------------------------------------------
    # 6. INYECCIÓN DE PESOS
    # -------------------------------------------------------------------------
    
    # A) LLaMA-3 (4-bit)
    print("🧠 [Factory] Inyectando LLaMA-3 (4-bit)...")
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
    
    # Redimensionar embeddings (Vital por el token <image>)
    print(f"   -> Redimensionando embeddings a: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # B) SigLIP (Vision Tower)
    print("👀 [Factory] Inyectando SigLIP (Carga REAL)...")
    siglip_full_model = AutoModel.from_pretrained(
        vision_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False, # Vital para evitar Meta Tensor
        device_map=None 
    )
    
    if torch.cuda.is_available():
        siglip_full_model = siglip_full_model.to("cuda")

    model.vision_tower = siglip_full_model.vision_model
    
    del siglip_full_model
    gc.collect()
    torch.cuda.empty_cache()

    # C) Proyector Multimodal
    print("🔌 [Factory] Inicializando Proyector Multimodal...")
    model.multi_modal_projector.to_empty(device="cuda") 
    for p in model.multi_modal_projector.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    # -------------------------------------------------------------------------
    # 7. Configuración LoRA (CORRECCIÓN DIEGO)
    # -------------------------------------------------------------------------
    print("🔧 [Factory] Configurando LoRA...")
    
    # Habilitar gradientes en inputs para checkpoints
    model.enable_input_require_grads() 
    
    # Aseguramos config correcta
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        # FIX DIEGO: Usamos regex para LLaMA y modules_to_save para el proyector
        target_modules=r"language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        modules_to_save=["multi_modal_projector"], # <--- ESTO ENTRENA EL PROYECTOR SIN LORA
        lora_dropout=cfg['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, processor