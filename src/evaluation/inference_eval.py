# src/evaluation/inference_eval.py

import argparse
import json
from pathlib import Path

import yaml
import torch
import pandas as pd
from datasets import load_from_disk
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from src.utils.paths import get_path, check_path
from src.evaluation.metrics import compute_vqa_accuracy, compute_bleu


# ---------------------------------------------------------
#  UTILIDADES DE CONFIG Y RUTAS
# ---------------------------------------------------------
def resolve_path(path_str: str) -> Path:
    """Convierte ruta relativa del YAML en absoluta."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return get_path(path_str)


def load_config(config_path: str) -> dict:
    """Carga la configuración YAML."""
    cfg = None
    config_path = get_path(config_path)
    print(f"📖 Cargando configuración desde: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validación mínima
    if "paths" not in cfg:
        raise KeyError("❌ El archivo YAML debe contener sección 'paths'.")

    required_paths = ["model_dir", "data_dir", "results_dir"]
    for k in required_paths:
        if k not in cfg["paths"]:
            raise KeyError(f"❌ Falta 'paths.{k}' en el archivo YAML.")

    if "inference" not in cfg:
        raise KeyError("❌ El archivo YAML debe contener sección 'inference'.")

    if "evaluation" not in cfg:
        # Permitir que exista sin ella, pero avisando
        print("⚠️ Advertencia: No existe sección 'evaluation' en el config. Se usarán defaults.")
        cfg["evaluation"] = {}

    return cfg


# ---------------------------------------------------------
#  CARGA DEL MODELO (Misma lógica que inference_sample.py)
# ---------------------------------------------------------
def load_model(model_dir: Path):
    print(f"⏳ Cargando modelo desde: {model_dir}...")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
    ).to(device)

    processor = AutoProcessor.from_pretrained(str(model_dir))

    model.eval()
    print(f"✅ Modelo cargado ({device}, dtype={dtype})")
    return model, processor


# ---------------------------------------------------------
#  GENERACIÓN DE RESPUESTA (determinista)
# ---------------------------------------------------------
def predict_one(model, processor, image: Image.Image, question: str) -> str:
    """
    Genera una respuesta con el prompt EXACTO usado durante entrenamiento.
    Determinista: do_sample=False, temperatura=0.0.
    """
    system_msg = "You are an expert pathologist. Answer the question based on the image provided."
    prompt = (
        f"<|system|>\n{system_msg}</s>\n"
        f"<|user|>\n<image>\n{question}</s>\n"
        f"<|assistant|>\n"
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extraer parte del assistant
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()

    return decoded


# ---------------------------------------------------------
#  MAIN - INFERENCIA MASIVA Y MÉTRICAS
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inferencia masiva para PathVQA (BLEU, Accuracy)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Ruta al archivo YAML de configuración"
    )
    args = parser.parse_args()

    # 1. Cargar configuración
    cfg = load_config(args.config)

    # 2. Resolver rutas
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])

    # Validaciones
    check_path(data_dir, is_dir=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Preparar subdirectorios
    predictions_dir = results_dir / "predictions"
    summary_dir = results_dir / "summary"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 3. Parámetros del YAML
    split = cfg["inference"].get("split", "validation")

    print("============================================================")
    print(f"🔧 Configuración de inferencia masiva")
    print(f"  • Modelo      : {model_dir}")
    print(f"  • Dataset     : {data_dir}")
    print(f"  • Results dir : {results_dir}")
    print(f"  • Split       : {split}")
    print("============================================================")

    # 4. Cargar modelo
    model, processor = load_model(model_dir)

    # 5. Cargar dataset
    print(f"📂 Cargando dataset desde: {data_dir}")
    dataset = load_from_disk(str(data_dir))

    if split not in dataset:
        raise ValueError(f"❌ Split '{split}' no existe en dataset. Splits disponibles: {list(dataset.keys())}")

    split_ds = dataset[split]
    n_samples = len(split_ds)
    print(f"📊 Total muestras en split '{split}': {n_samples}")

    questions = []
    references = []
    predictions = []
    ids = []

    # 6. Inferencia sobre todas las muestras
    print("🚀 Ejecutando inferencia de todas las muestras...")

    for idx, item in enumerate(split_ds):
        image = item["image"].convert("RGB")
        question = item["question"]
        reference = str(item["answer"])

        try:
            pred = predict_one(model, processor, image, question)
        except Exception as e:
            pred = f"[ERROR] {str(e)}"

        questions.append(question)
        references.append(reference)
        predictions.append(pred)
        ids.append(idx)

        if idx % 100 == 0:
            print(f"  → Procesadas {idx}/{n_samples} muestras...")

    # 7. Guardar CSV
    # Nombre del modelo = nombre de la carpeta final del checkpoint
    model_name = model_dir.name
    csv_name = f"predictions_{model_name}_{split}.csv"
    csv_path = predictions_dir / csv_name

    df = pd.DataFrame({
        "id": ids,
        "question": questions,
        "reference": references,
        "prediction": predictions,
    })

    df.to_csv(csv_path, index=False)
    print(f"💾 CSV guardado en: {csv_path}")

    # 8. Calcular métricas globales
    print("📈 Calculando métricas globales...")
    acc = compute_vqa_accuracy(predictions, references)
    bleu = compute_bleu(predictions, references)

    print("============================================================")
    print(f"📊 MÉTRICAS FINALES ({model_name} @ {split})")
    print(f"   ✔ Accuracy (yes/no) = {acc:.4f}")
    print(f"   ✔ BLEU               = {bleu:.4f}")
    print("============================================================")

    # Guardar métricas en JSON
    summary_path = summary_dir / f"metrics_{model_name}_{split}.json"
    metrics_obj = {
        "model": model_name,
        "split": split,
        "samples": n_samples,
        "accuracy": acc,
        "bleu": bleu,
    }

    with open(summary_path, "w") as f:
        json.dump(metrics_obj, f, indent=4)

    print(f"💾 Reporte JSON guardado en: {summary_path}")
    print("🎉 Inferencia masiva completada correctamente.")


if __name__ == "__main__":
    main()
