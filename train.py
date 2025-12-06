import yaml
import torch
import argparse
import os
from transformers import TrainingArguments, Trainer
from src.utils.paths import get_path, check_path
from src.data.dataset import PathVQADataset
from src.models.model_factory import build_model_and_processor
from src.training.callbacks import FileLoggerCallback

# Configurar entorno para evitar fragmentación de memoria (Ayuda con Segfaults)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CLASE CUSTOM TRAINER ---
class VqaTrainer(Trainer):
    """
    Trainer personalizado para VQA Multimodal.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs)

def data_collator(features):
    """
    Collate function para apilar tensores de imagen y texto.
    """
    if not features:
        return {}
    
    first = features[0]
    batch = {}
    
    for k in first.keys():
        if k == "label": 
            continue
        if first[k] is not None:
            batch[k] = torch.stack([f[k] for f in features])
            
    return batch

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento VQA: TinyLlama + CLIP (Full Fine-Tuning)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Ruta al archivo config")
    args = parser.parse_args()

    # 1. Cargar Configuración
    config_path = get_path(args.config)
    print(f"📖 Cargando configuración desde: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Preparar Modelo y Procesador
    print("🏗️ Construyendo arquitectura multimodal (FFT)...")
    model, processor = build_model_and_processor(cfg)
    
    # 🛡️ FIX SEGFAULT: Desactivar caché explícitamente para Gradient Checkpointing
    model.config.use_cache = False
    model.generation_config.use_cache = False # También en generation config por si acaso

    # 3. Preparar Datos
    data_dir = check_path(get_path(cfg["paths"]["data_dir"]))
    print(f"📂 Dataset root: {data_dir}")

    max_len = cfg["training"].get("max_length", 1024)

    print("Dataset: Cargando split de entrenamiento...")
    train_dataset = PathVQADataset(
        data_dir,
        processor=processor,
        split="train",
        model_max_length=max_len,
    )
    
    try:
        print("Dataset: Cargando split de validación...")
        val_dataset = PathVQADataset(
            data_dir,
            processor=processor,
            split="validation",
            model_max_length=max_len,
        )
    except Exception:
        print("⚠️ No se encontró split de validación, usando subconjunto de train para debug.")
        val_dataset = torch.utils.data.Subset(train_dataset, range(min(len(train_dataset), 10)))

    # 4. Configurar Argumentos
    output_dir = get_path(cfg["project"]["output_dir"])
    logging_dir = get_path(cfg["project"]["logging_dir"])

    use_fp16 = cfg["training"].get("fp16", False)
    use_bf16 = cfg["training"].get("bf16", False)
    
    if not torch.cuda.is_available():
        print("⚠️ AVISO: Entrenando en CPU. Se desactiva FP16/BF16.")
        use_fp16 = False
        use_bf16 = False
        cfg["training"]["batch_size"] = 1 
        cfg["training"]["grad_accumulation"] = 1

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg["project"]["name"],
        
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accumulation"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        num_train_epochs=cfg["training"]["epochs"],
        
        fp16=use_fp16,
        bf16=use_bf16,
        
        # Optimizaciones
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True) if torch.cuda.is_available() else False,
        optim="adamw_torch", 
        
        logging_dir=str(logging_dir),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        
        # 🛡️ FIX SEGFAULT: Desactivar compilación si está activa por defecto en PyTorch 2.x
        torch_compile=False 
    )

    # 5. Inicializar Trainer
    trainer = VqaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[FileLoggerCallback(output_dir)],
    )

    # 6. Entrenar
    print("\n🚀 Comenzando entrenamiento...")
    
    # Contexto de precisión TF32 (Mejora estabilidad en Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    trainer.train()

    # 7. Guardar Modelo Final
    final_path = output_dir / "final_model"
    print(f"\n💾 Guardando modelo completo en: {final_path}")
    
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))

    print("✅ Entrenamiento finalizado exitosamente.")

if __name__ == "__main__":
    main()