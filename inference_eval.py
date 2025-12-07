import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.evaluation.metrics import compute_vqa_accuracy, compute_bleu
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluación Masiva del Modelo VQA")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado")
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta al dataset guardado (HuggingFace disk)")
    parser.add_argument("--split", type=str, default="test", help="Split a evaluar (test/validation)")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Archivo de salida")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de muestras (para pruebas rápidas)")
    args = parser.parse_args()

    # 1. Cargar Modelo y Procesador
    print(f"🏗️ Cargando modelo desde: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=dtype, 
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model)
    
    # 2. Cargar Dataset
    print(f"📂 Cargando datos ({args.split}) desde: {args.data_dir}")
    try:
        dataset = load_from_disk(args.data_dir)[args.split]
    except KeyError:
        print(f"⚠️ Split '{args.split}' no encontrado. Usando 'validation' o 'train' como fallback.")
        dataset = load_from_disk(args.data_dir)['train'] # Fallback

    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"⚠️ Limitando evaluación a {args.limit} muestras.")

    print(f"📊 Total de muestras a evaluar: {len(dataset)}")

    # 3. Loop de Inferencia
    predictions = []
    references = []
    results_data = []

    print("🚀 Iniciando generación...")
    model.eval()
    
    # Template del Sistema
    system_msg = "You are an expert pathologist. Answer the question based on the image provided."

    for item in tqdm(dataset):
        image = item['image'].convert('RGB')
        question = item['question']
        ground_truth = str(item['answer']) # La verdad absoluta
        
        # Prompt (Sin la respuesta)
        prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False, # Greedy decoding para evaluación determinista (mejor para métricas)
                temperature=0.0  # Apagar aleatoriedad
            )

        full_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Extraer respuesta limpia
        prediction = full_response.split("<|assistant|>")[-1].strip()
        
        # Guardar para métricas
        predictions.append(prediction)
        references.append(ground_truth)
        
        # Guardar para CSV
        results_data.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "match": prediction.lower().strip() == ground_truth.lower().strip()
        })

    # 4. Cálculo de Métricas
    print("\n📈 Calculando métricas finales...")
    
    # Accuracy (Exact Match / Yes-No)
    acc = compute_vqa_accuracy(predictions, references)
    
    # BLEU (Similitud semántica)
    try:
        bleu = compute_bleu(predictions, references)
    except Exception as e:
        print(f"⚠️ Error calculando BLEU: {e}")
        bleu = 0.0

    print("="*40)
    print(f"🎯 RESULTADOS FINALES ({len(dataset)} muestras)")
    print("="*40)
    print(f"✅ VQA Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"🔵 BLEU Score:   {bleu:.4f}")
    print("="*40)

    # 5. Guardar CSV
    df = pd.DataFrame(results_data)
    df.to_csv(args.output, index=False)
    print(f"💾 Resultados detallados guardados en: {args.output}")

if __name__ == "__main__":
    main()