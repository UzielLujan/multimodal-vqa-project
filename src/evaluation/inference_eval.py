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
from src.evaluation.metrics import (
    compute_bleu,
    compute_bleu_short,
    compute_keyword_accuracy,
    compute_yesno_accuracy,
    compute_general_accuracy_flexible,
)


# ---------------------------------------------------------
#  Helpers YAML
# ---------------------------------------------------------
def load_config(config_path: str) -> dict:
    config_path = get_path(config_path)
    print(f" Cargando configuración desde: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "paths" not in cfg or "results_dir" not in cfg["paths"]:
        raise KeyError("El config necesita paths.results_dir")

    if "inference" not in cfg:
        raise KeyError("El config necesita sección 'inference'.")

    return cfg


# ---------------------------------------------------------
#  Carga del modelo
# ---------------------------------------------------------
def load_model(model_dir: Path):
    print(f" Cargando modelo desde: {model_dir}...")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=dtype
    ).to(device)

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model.eval()

    print(f"Modelo cargado ({device}, dtype={dtype})")
    return model, processor


# ---------------------------------------------------------
#  Generar respuesta determinista
# ---------------------------------------------------------
def generate_answer(model, processor, image: Image.Image, question: str, max_new_tokens: int):
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

    eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            eos_token_id=eos_token_id,
        )

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extraer parte del assistant
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()

    return decoded


# ---------------------------------------------------------
#  MAIN — Evaluación masiva
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inferencia masiva para PathVQA con métricas avanzadas")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # 1. Cargar config
    cfg = load_config(args.config)
    inf_cfg = cfg["inference"]

    model_dir = get_path(cfg["paths"]["model_dir"])
    data_dir = get_path(cfg["paths"]["data_dir"])
    results_dir = get_path(cfg["paths"]["results_dir"])

    split = inf_cfg.get("split", "test")
    max_new_tokens = inf_cfg.get("max_new_tokens", 20)

    # 2. Preparar rutas
    predictions_dir = results_dir / "predictions"
    summary_dir = results_dir / "summary"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 3. Cargar modelo
    model, processor = load_model(model_dir)

    # 4. Cargar dataset
    print(f" Cargando dataset desde: {data_dir}")
    dataset = load_from_disk(str(data_dir))

    if split not in dataset:
        raise ValueError(f"Split {split} no encontrado. Splits: {list(dataset.keys())}")

    ds = dataset[split]
    n_samples = len(ds)
    print(f" Total muestras: {n_samples}")

    ids = []
    questions = []
    references = []
    predictions = []

    # 5. Inferencia
    print(" Ejecutando inferencia...")
    for idx, item in enumerate(ds):
        img = item["image"].convert("RGB")
        question = item["question"]
        reference = str(item["answer"])

        try:
            pred = generate_answer(model, processor, img, question, max_new_tokens=max_new_tokens)
        except Exception as e:
            pred = f"[ERROR] {e}"

        ids.append(idx)
        questions.append(question)
        references.append(reference)
        predictions.append(pred)

        if idx % 100 == 0:
            print(f"   → Procesadas {idx}/{n_samples}")

    # 6. Guardar CSV
    model_name = model_dir.parent.name
    csv_path = predictions_dir / f"predictions_{model_name}_{split}.csv"

    df = pd.DataFrame({
        "id": ids,
        "question": questions,
        "reference": references,
        "prediction": predictions,
    })

    df.to_csv(csv_path, index=False)
    print(f" CSV guardado en: {csv_path}")

    # 7. Calcular métricas
    print(" Calculando métricas avanzadas...")

    acc_yesno = compute_yesno_accuracy(predictions, references)
    acc_general = compute_general_accuracy_flexible(predictions, references)
    keyword_acc = compute_keyword_accuracy(predictions, references)
    bleu_full = compute_bleu(predictions, references)
    bleu_short = compute_bleu_short(predictions, references, max_words=5)

    print("============================================================")
    print(" MÉTRICAS FINALES")
    print(f"  • Accuracy Yes/No            : {acc_yesno:.4f}")
    print(f"  • Accuracy General Flexible  : {acc_general:.4f}")
    print(f"  • Keyword Accuracy           : {keyword_acc:.4f}")
    print(f"  • BLEU short (5 palabras)    : {bleu_short:.4f}")
    print(f"  • BLEU clásico               : {bleu_full:.4f}")
    print("============================================================")

    # 8. Guardar JSON resumen
    summary_path = summary_dir / f"metrics_{model_name}_{split}.json"
    metrics_obj = {
        "model": model_name,
        "split": split,
        "samples": n_samples,
        "accuracy_yesno": acc_yesno,
        "accuracy_general_flexible": acc_general,
        "keyword_accuracy": keyword_acc,
        "bleu_short": bleu_short,
        "bleu": bleu_full,
    }

    with open(summary_path, "w") as f:
        json.dump(metrics_obj, f, indent=4)

    print(f" Resumen guardado en: {summary_path}")
    print(" Evaluación completa.")


if __name__ == "__main__":
    main()
