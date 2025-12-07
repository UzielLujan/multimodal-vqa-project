# src/evaluation/inference_sample.py

import argparse
import random
import textwrap
from pathlib import Path

import yaml
import torch
from datasets import load_from_disk
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

import matplotlib
matplotlib.use("Agg")  # backend sin display, ideal para cluster
import matplotlib.pyplot as plt

from src.utils.paths import get_path, check_path


# ---------------------------------------------------------
#  Helpers de rutas / config
# ---------------------------------------------------------
def resolve_path(cfg_path: str) -> Path:
    """
    Convierte una ruta de config (relativa al proyecto) en absoluta.
    Si ya es absoluta, la regresa tal cual.
    """
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return get_path(cfg_path)


def load_config(config_rel_path: str) -> dict:
    """
    Carga el YAML de configuración de inferencia.
    Espera al menos:
      paths:
        model_dir
        data_dir
        results_dir
      inference:
        split
        index
        max_new_tokens
        trim_output
        trim_words
        panel_layout
        font_size
        wrap_width
        dpi
    """
    config_path = get_path(config_rel_path)
    print(f"📖 Cargando configuración desde: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validaciones mínimas
    if "paths" not in cfg:
        raise KeyError("❌ El config debe contener la sección 'paths'.")

    required_paths = ["model_dir", "data_dir", "results_dir"]
    for key in required_paths:
        if key not in cfg["paths"]:
            raise KeyError(f"❌ Falta 'paths.{key}' en el archivo de config.")

    if "inference" not in cfg:
        raise KeyError("❌ El config debe contener la sección 'inference'.")

    # Defaults razonables si faltan campos
    inf = cfg["inference"]
    inf.setdefault("split", "validation")
    inf.setdefault("index", -1)
    inf.setdefault("max_new_tokens", 20)
    inf.setdefault("trim_output", True)
    inf.setdefault("trim_words", 12)
    inf.setdefault("panel_layout", "horizontal")  # "horizontal" | "vertical"
    inf.setdefault("font_size", 16)
    inf.setdefault("wrap_width", 60)
    inf.setdefault("dpi", 200)

    return cfg


# ---------------------------------------------------------
#  Carga del modelo + processor (alineado con model_factory)
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
        dtype=dtype,
    ).to(device)

    processor = AutoProcessor.from_pretrained(str(model_dir))

    model.eval()
    print(f"✅ Modelo cargado ({device}, dtype={dtype})")
    return model, processor


# ---------------------------------------------------------
#  Generar predicción (con truncado inteligente)
# ---------------------------------------------------------
def generate_answer(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    trim_output: bool,
    trim_words: int,
) -> str:
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

    # Truncado inteligente por número de palabras
    if trim_output and trim_words > 0:
        words = decoded.split()
        if len(words) > trim_words:
            decoded = " ".join(words[:trim_words])

    return decoded


# ---------------------------------------------------------
#  Construir panel con matplotlib (horizontal / vertical)
# ---------------------------------------------------------
def build_panel_matplotlib(
    image: Image.Image,
    question: str,
    gt_answer: str,
    pred_answer: str,
    out_path: Path,
    layout: str = "horizontal",
    font_size: int = 16,
    wrap_width: int = 60,
    dpi: int = 150,
):
    """
    Crea un panel tipo paper con matplotlib:
      layout="horizontal": imagen izquierda, texto derecha.
      layout="vertical"  : imagen arriba, texto abajo.
    """
    # Asegurar RGB
    image = image.convert("RGB")

    # Preparar texto con envoltura
    wrapper = textwrap.TextWrapper(width=wrap_width)

    def wrap_block(title: str, text: str) -> str:
        lines = [title]
        lines.extend(wrapper.wrap(text))
        return "\n".join(lines)

    text_question = wrap_block("Question:", question)
    text_gt = wrap_block("Ground Truth:", gt_answer)
    text_pred = wrap_block("Model Prediction:", pred_answer)

    full_text = f"{text_question}\n\n{text_gt}\n\n{text_pred}"

    # ---------------------------------------------------------
    #  LAYOUT "paper": estilo artículo científico
    # ---------------------------------------------------------
    if layout == "paper":
        fig = plt.figure(figsize=(12, 7), dpi=dpi)

        # Gridspec: título arriba, panel principal abajo
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[0.15, 0.85],
            hspace=0.25
        )

        # -------------------------
        #  Título del panel
        # -------------------------
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.set_facecolor("white")
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5,
            f"PathVQA Sample – {out_path.stem}",
            fontsize=font_size + 4,
            fontweight="bold",
            ha="center",
            va="center"
        )

        # -------------------------
        #  Panel principal (imagen + texto)
        # -------------------------
        gs_main = gs[1].subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)

        ax_img = fig.add_subplot(gs_main[0, 0])
        ax_txt = fig.add_subplot(gs_main[0, 1])

        # Imagen
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title("Image", fontsize=font_size + 2)

        # Preparar texto

        # wrap
        wrapper = textwrap.TextWrapper(width=wrap_width)
        def wrap_block(title, txt):
            lines = [title]
            lines.extend(wrapper.wrap(txt))
            return "\n".join(lines)

        block_question = wrap_block("Question:", question)
        block_gt = wrap_block("Ground Truth:", gt_answer)
        block_pred = wrap_block("Model Prediction:", pred_answer)

        full_text = (
            f"{block_question}\n\n"
            f"{block_gt}\n\n"
            f"{block_pred}"
        )

        # Texto formateado
        ax_txt.set_facecolor("#7290A8EA") # fondo gris 
        ax_txt.axis("off")
        ax_txt.text(
            0.02, 0.98,
            full_text,
            fontsize=font_size,
            va="top",
            ha="left",
            wrap=True,
        )

        # Guardar
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"💾 Panel guardado (paper mode) en: {out_path}")
        return

    # Crear figura según layout
    if layout == "vertical":
        fig, (ax_img, ax_txt) = plt.subplots(
            2, 1,
            figsize=(8, 10),
            gridspec_kw={"height_ratios": [2, 1]},
            dpi=dpi,
        )

        # Imagen
        ax_img.imshow(image)
        ax_img.axis("off")

        # Texto
        ax_txt.set_facecolor("#F5F5F5")
        ax_txt.axis("off")
        ax_txt.text(
            0.01,
            0.99,
            full_text,
            fontsize=font_size,
            va="center",
            ha="left",
            wrap=True,
        )

    else:  # layout horizontal (por defecto)
        fig, (ax_img, ax_txt) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [1.5, 1]},
            dpi=dpi,
        )

        # Imagen a la izquierda
        ax_img.imshow(image)
        ax_img.axis("off")

        # Texto a la derecha
        ax_txt.set_facecolor("#F5F5F5")
        ax_txt.axis("off")
        ax_txt.text(
            0.02,
            0.98,
            full_text,
            fontsize=font_size,
            va="center",
            ha="left",
            wrap=True,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"💾 Panel guardado en: {out_path}")


# ---------------------------------------------------------
#  Lógica principal: tomar un sample del dataset y visualizarlo
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generar panel de inferencia (1 sample) para PathVQA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Ruta al archivo YAML de configuración de inferencia",
    )
    args = parser.parse_args()

    # 1. Cargar configuración
    cfg = load_config(args.config)

    # 2. Resolver rutas desde el YAML
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])

    # Verificaciones mínimas
    check_path(data_dir, is_dir=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Parámetros de inferencia
    inf_cfg = cfg["inference"]
    split = inf_cfg.get("split", "validation")
    index = inf_cfg.get("index", -1)
    max_new_tokens = inf_cfg.get("max_new_tokens", 20)
    trim_output = inf_cfg.get("trim_output", True)
    trim_words = inf_cfg.get("trim_words", 12)
    panel_layout = inf_cfg.get("panel_layout", "horizontal")
    font_size = inf_cfg.get("font_size", 16)
    wrap_width = inf_cfg.get("wrap_width", 60)
    dpi = inf_cfg.get("dpi", 200)

    print("============================================================")
    print(f"🔧 Configuración de inferencia SAMPLE")
    print(f"  • Modelo        : {model_dir}")
    print(f"  • Dataset       : {data_dir}")
    print(f"  • Results dir   : {results_dir}")
    print(f"  • Split         : {split}")
    print(f"  • Index         : {index}  ( -1 = aleatorio )")
    print(f"  • max_new_tokens: {max_new_tokens}")
    print(f"  • trim_output   : {trim_output}")
    print(f"  • trim_words    : {trim_words}")
    print(f"  • panel_layout  : {panel_layout}")
    print(f"  • font_size     : {font_size}")
    print(f"  • wrap_width    : {wrap_width}")
    print(f"  • dpi           : {dpi}")
    print("============================================================")

    # 4. Cargar modelo
    model, processor = load_model(model_dir)

    # 5. Cargar dataset
    print(f"📂 Cargando dataset desde: {data_dir}")
    dataset = load_from_disk(str(data_dir))

    if split not in dataset:
        raise ValueError(f"❌ El split '{split}' no existe en el dataset. Splits disponibles: {list(dataset.keys())}")

    split_ds = dataset[split]
    n_samples = len(split_ds)
    if n_samples == 0:
        raise RuntimeError(f"❌ El split '{split}' está vacío.")

    # 6. Seleccionar índice
    if index < 0 or index >= n_samples:
        idx = random.randint(0, n_samples - 1)
        print(f"🎲 Índice no válido o negativo. Usando muestra aleatoria: idx={idx} de {n_samples}")
    else:
        idx = index
        print(f"📌 Usando índice proporcionado: idx={idx} de {n_samples}")

    # 7. Extraer muestra
    item = split_ds[idx]
    image = item["image"].convert("RGB")
    question = item["question"]
    gt_answer = str(item["answer"])

    print("------------------------------------------------------------")
    print(f"🖼️ Sample idx: {idx}")
    print(f"❓ Pregunta: {question}")
    print(f"🟢 GT: {gt_answer}")
    print("------------------------------------------------------------")

    # 8. Generar predicción (con truncado)
    pred_answer = generate_answer(
        model=model,
        processor=processor,
        image=image,
        question=question,
        max_new_tokens=max_new_tokens,
        trim_output=trim_output,
        trim_words=trim_words,
    )
    print(f"🤖 Predicción del modelo (truncada): {pred_answer}")
    print("------------------------------------------------------------")

    # 9. Construir y guardar panel dentro de results/samples/
    samples_dir = results_dir / "samples"
    out_path = samples_dir / f"sample_{split}_{idx:05d}.png"

    build_panel_matplotlib(
        image=image,
        question=question,
        gt_answer=gt_answer,
        pred_answer=pred_answer,
        out_path=out_path,
        layout=panel_layout,
        font_size=font_size,
        wrap_width=wrap_width,
        dpi=dpi,
    )


if __name__ == "__main__":
    main()
