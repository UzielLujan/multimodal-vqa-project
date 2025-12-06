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
    CLIPVisionModel,
    CLIPImageProcessor
)
from src.utils.paths import get_path
import gc

def build_model_and_processor(cfg):
    """
    Construye un modelo LLaVA: CLIP-Large + TinyLlama.
    FUSIÓN: Lógica de inyección original (User) + Configuración CLIP (Fix 577 tokens).
    """
    llm_path = str(get_path(cfg['paths']['llm_model_path']))
    vision_path = str(get_path(cfg['paths']['vision_tower_path']))
    
    # 1. Hardware Setup
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"⚡ [Factory] Dispositivo: CUDA ({torch_dtype})")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print(" [Factory] Dispositivo: CPU (Forzando Float32)")
        
    print(f" [Factory] Rutas:\n   - LLM: {llm_path}\n   - Vision: {vision_path}")

    # 2. Tokenizer
    print(" [Factory] Configurando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) or tokenizer.eos_token_id
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    # 3. Configs
    print(" [Factory] Fusionando configuraciones...")
    text_config = AutoConfig.from_pretrained(llm_path)
    clip_config = AutoConfig.from_pretrained(vision_path)
    
    # Extraer config visual si está anidada
    vision_config = getattr(clip_config, 'vision_config', clip_config)

    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=image_token_index,
        projector_hidden_act="gelu",
        
        # EL FIX MATEMÁTICO (577 -> 576)
        # "default" en LLaVA 1.5 significa "patch", lo que elimina el token CLS.
        # "full" incluye el CLS y rompe la matriz.
        vision_feature_select_strategy="default", 
        
        vision_feature_layer=-2,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=text_config.hidden_size,
        vocab_size=text_config.vocab_size 
    )

    # 4. Processor
    print(" [Factory] Configurando Procesador...")
    image_processor = CLIPImageProcessor.from_pretrained(vision_path)
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.patch_size = 14 # Parche de seguridad

    # 5. Instanciar Esqueleto
    print(" [Factory] Creando esqueleto LLaVA...")
    model = LlavaForConditionalGeneration(llava_config)
    
    # 6. CIRUGÍA DE PESOS (Basada en tu script original) 
    print(" [Factory] Cargando TinyLlama (Donante)...")
    llm_donor = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch_dtype)
    
    # Redimensionamos el donante ANTES de inyectarlo (como en tu script original)
    # Esto asegura que la matriz de embeddings tenga el tamaño correcto (32001)
    llm_donor.resize_token_embeddings(len(tokenizer))
    
    print(" [Factory] Realizando trasplante de órganos...")
    
    # LÓGICA ORIGINAL RECUPERADA (Soluciona los parámetros duplicados)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        print("   -> Inyección Profunda (model.model.language_model)")
        model.model.language_model = llm_donor.model
        model.lm_head = llm_donor.lm_head
    else:
        print("   -> Inyección Directa (model.language_model)")
        model.language_model = llm_donor

    # Actualizamos config padre para que no haya quejas de índices
    model.config.vocab_size = len(tokenizer)
    model.config.text_config.vocab_size = len(tokenizer)

    print(" [Factory] Cargando CLIP Vision Tower...")
    vision_tower = CLIPVisionModel.from_pretrained(vision_path, dtype=torch_dtype)
    
    # Inyección de Vision Tower
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        model.model.vision_tower = vision_tower
    else:
        model.vision_tower = vision_tower
    
    # 7. Configuración Entrenamiento (FFT)
    print(" [Factory] Configurando capas entrenables...")
    
    # Búsqueda agnóstica de módulos (Tu lógica original era muy sólida aquí)
    vision_module = model.model.vision_tower if hasattr(model, "model") else model.vision_tower
    llm_module = model.model.language_model if hasattr(model, "model") else model.language_model
    projector_module = model.model.multi_modal_projector if hasattr(model, "model") else model.multi_modal_projector

    # Congelar Vision
    for param in vision_module.parameters():
        param.requires_grad = False
    
    # Descongelar LLM
    for param in llm_module.parameters():
        param.requires_grad = True
        
    # Descongelar Proyector
    projector_module.to(dtype=torch_dtype)
    for p in projector_module.parameters():
        p.requires_grad = True
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    # Limpieza
    del llm_donor
    del vision_tower
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Mover y Castear
    print(f" [Factory] Moviendo modelo a {device}...")
    model.to(device)
    
    # Casting final para evitar error Float vs BFloat16
    model.to(dtype=torch_dtype)
    
    # Stats Finales
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f" [Factory] Params Totales: {all_params:,}")
    if all_params > 1_600_000_000:
        print("⚠️ ADVERTENCIA: Aún hay duplicados (¿Seguro que se borró el fantasma?).")
    else:
        print(" ✅ TAMAÑO CORRECTO: ~1.5B (Fantasma eliminado).")
        
    print(f" [Factory] Entrenables:   {trainable_params:,}")
    
    return model, processor