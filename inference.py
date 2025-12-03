import yaml
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
from torch.utils.data import DataLoader

# Importaciones modulares
from src.utils.paths import get_path, check_path
from src.data.dataset import PathVQADataset
from src.evaluation.metrics import compute_vqa_accuracy, compute_bleu

def load_model_for_inference(cfg):
    """Carga el modelo Base + Adaptador LoRA."""
    llm_path = str(get_path(cfg['paths']['llm_model_path']))
    vision_path = str(get_path(cfg['paths']['vision_tower_path']))
    adapter_path = str(get_path(cfg['inference']['adapter_path']))

    print(f"Cargando Base: {llm_path}")
    print(f"Cargando Adaptador LoRA: {adapter_path}")

    # 1. Configurar Quantización (Igual que en train)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Cargar Procesador
    processor = AutoProcessor.from_pretrained(llm_path)
    processor.image_processor = AutoProcessor.from_pretrained(vision_path).image_processor

    # 3. Cargar Modelo Base
    model = LlavaForConditionalGeneration.from_pretrained(
        llm_path,
        vision_tower_address=vision_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 4. Inyectar LoRA (Paso Crítico de Inferencia)
    # Esto fusiona dinámicamente tus pesos entrenados con el modelo base
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval() # Modo evaluación (apaga dropout)
    
    return model, processor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    # Cargar Config
    config_path = get_path(args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Preparar Modelo
    model, processor = load_model_for_inference(cfg)

    # 2. Cargar Datos de TEST
    data_dir = get_path(cfg['paths']['data_dir'])
    test_dataset = PathVQADataset(
        data_dir, 
        processor=processor, 
        split='test',
        model_max_length=cfg['training']['max_length']
    )
    
    # DataLoader (Batch size 1 es más seguro para generación de texto variable)
    test_loader = DataLoader(test_dataset, batch_size=cfg['inference']['batch_size'], shuffle=False)

    print(f"Iniciando Inferencia sobre {len(test_dataset)} muestras de TEST...")

    predictions = []
    references = []
    questions = []

    # 3. Bucle de Inferencia
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Generando"):
            # Mover a GPU
            input_ids = batch['input_ids'].to(model.device)
            pixel_values = batch['pixel_values'].to(model.device).to(torch.float16)
            attention_mask = batch['attention_mask'].to(model.device)

            # Generar
            output_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=cfg['inference']['max_new_tokens'],
                temperature=cfg['inference']['temperature'],
                do_sample=False, # Determinista para evaluar
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

            # Decodificar
            generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)
            
            # Post-procesamiento: El modelo genera "Prompt + Respuesta". Queremos solo "Respuesta".
            # Como LLaMA-3 usa tags especiales, el decode suele limpiar bien, 
            # pero a veces queda basura del prompt. 
            # Estrategia robusta: Cortar por el token de "assistant" si aparece en el raw text,
            # pero aquí asumiremos que skip_special_tokens hace la mayor parte del trabajo.
            # Una limpieza extra simple:
            clean_preds = [txt.split("assistant\n")[-1].strip() for txt in generated_text]

            # Guardar resultados
            # Recuperamos el texto original de la pregunta/respuesta del dataset
            # Nota: Esto es un truco porque el DataLoader devuelve tensores. 
            # En un pipeline estricto, deberíamos pasar los IDs, pero para simplificar
            # guardamos solo las predicciones y luego comparamos alineando índices.
            predictions.extend(clean_preds)

    # 4. Recuperar Referencias (Ground Truth) para Métricas
    # Iteramos el dataset original para sacar los textos crudos
    print("Calculando métricas...")
    raw_dataset = test_dataset.dataset
    for item in raw_dataset:
        references.append(item['answer'])
        questions.append(item['question'])

    # 5. Calcular Métricas
    # A) Accuracy (Binarias)
    acc = compute_vqa_accuracy(predictions, references)
    
    # B) BLEU (Abiertas)
    bleu = compute_bleu(predictions, references)

    print("\n" + "="*30)
    print(f"RESULTADOS FINALES:")
    print(f"Accuracy (Sí/No): {acc*100:.2f}%")
    print(f"BLEU Score:      {bleu:.4f}")
    print("="*30 + "\n")

    # 6. Guardar CSV detallado
    output_dir = get_path(cfg['project']['output_dir'])
    results_file = output_dir / cfg['inference']['predictions_file']
    
    df_results = pd.DataFrame({
        "question": questions,
        "ground_truth": references,
        "prediction": predictions
    })
    
    df_results.to_csv(results_file, index=False)
    print(f"💾 Predicciones guardadas en: {results_file}")

if __name__ == "__main__":
    main()