import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
import os

# Desactivar advertencias de hf
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_path):
    print(f"⏳ Cargando modelo desde: {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=dtype,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        exit(1)
        
    print(f"✅ Modelo cargado en {device} ({dtype})")
    return model, processor

def predict(model, processor, image_path, question):
    # 1. Cargar Imagen
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error abriendo imagen: {e}"

    # 2. Template de Prompt (El mismo del training pero SIN la respuesta)
    system_msg = "You are an expert pathologist. Answer the question based on the image provided."
    prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"

    # 3. Procesar Inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Mover a GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    # 4. Generar
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,      # Sampling para variedad (o False para determinismo)
            temperature=0.2,     # Bajo para respuestas médicas precisas
            top_p=0.9,
            use_cache=True
        )

    # 5. Decodificar
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Limpieza: Extraer solo lo que sigue a "assistant"
    # El prompt original está en generated_text, hay que cortarlo
    response = generated_text.split("<|assistant|>")[-1].strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Demo de Inferencia VQA")
    parser.add_argument("--model", type=str, required=True, help="Ruta a la carpeta 'final_model'")
    parser.add_argument("--image", type=str, required=True, help="Ruta al archivo de imagen")
    parser.add_argument("--question", type=str, default="Is this tissue normal?", help="Pregunta médica")
    
    args = parser.parse_args()

    model, processor = load_model(args.model)
    
    print("-" * 50)
    print(f"❓ Pregunta: {args.question}")
    print("-" * 50)
    
    respuesta = predict(model, processor, args.image, args.question)
    
    print(f"💡 Respuesta del Modelo:\n{respuesta}")
    print("-" * 50)

if __name__ == "__main__":
    main()