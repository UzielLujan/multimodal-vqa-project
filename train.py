import yaml
import torch
import argparse
from transformers import TrainingArguments, Trainer
from src.utils.paths import get_path, check_path
from src.data.dataset import PathVQADataset
from src.models.model_factory import build_model_and_processor
from src.training.callbacks import FileLoggerCallback
from src.utils.visualization import plot_training_loss

def data_collator(features):
    """Collate function simple para multimodal."""
    first = features[0]
    batch = {}
    for k in first.keys():
        batch[k] = torch.stack([f[k] for f in features])
    return batch

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de LLaVA para VQA")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Ruta al archivo config")
    args = parser.parse_args()

    # 1. Cargar Configuración
    config_path = get_path(args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print("Iniciando Pipeline de Entrenamiento")

    # 2. Preparar Modelo y Procesador
    model, processor = build_model_and_processor(cfg)

    # 3. Preparar Datos
    data_dir = get_path(cfg['paths']['data_dir'])
    check_path(data_dir)
    
    train_dataset = PathVQADataset(
        data_dir, 
        processor=processor, 
        split='train',
        model_max_length=cfg['training']['max_length']
    )
    val_dataset = PathVQADataset(
        data_dir, 
        processor=processor, 
        split='validation',
        model_max_length=cfg['training']['max_length']
    )

    # 4. Configurar Argumentos de Entrenamiento
    output_dir = get_path(cfg['project']['output_dir'])
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['grad_accumulation'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['epochs'],
        logging_dir=str(get_path(cfg['project']['logging_dir'])),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=cfg['training']['fp16'],
        remove_unused_columns=False,
        report_to="wandb",
        run_name=cfg['project']['name']
    )

    # 5. Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[FileLoggerCallback(output_dir)] # Inyección del logger personalizado
    )

    # 6. Entrenar
    print("Comenzando entrenamiento...")
    trainer.train()

    # 7. Guardar y GRAFICAR
    final_path = output_dir / "final_adapter"
    print(f"Guardando modelo final en: {final_path}")
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    
    # ✨ Generar gráfica post-mortem
    print("Generando curva de aprendizaje...")
    log_file = output_dir / "training_logs.jsonl"
    plot_path = output_dir / "loss_curve.png"
    try:
        plot_training_loss(log_file, plot_path)
    except Exception as e:
        print(f"⚠️ No se pudo generar la gráfica: {e}")

if __name__ == "__main__":
    main()