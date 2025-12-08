# src/evaluation/metrics.py

"""
metrics.py

Métricas mejoradas para PathVQA:

✓ Accuracy Yes/No basada en la primera palabra de la predicción
✓ Accuracy General Flexible (parcial, basada en contenido)
✓ Keyword Accuracy (coincidencia literal del Ground Truth en la predicción)
✓ BLEU clásico
✓ Short-BLEU (BLEU usando solo las primeras N palabras del modelo)

Este módulo está diseñado para integrarse con:
- inference_eval.py
- evaluation_tools.py
- evaluation_batch.py
"""

import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ---------------------------------------------------------
#  Normalización de textos
# ---------------------------------------------------------
def normalize(text: str) -> str:
    """
    Limpieza básica para comparar textos de forma flexible:
    - lowercase
    - remover puntuación
    - remover espacios extras
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()

    # Remover puntuación simple
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Colapsar espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
#  Accuracy Yes/No (solo 1era palabra del modelo)
# ---------------------------------------------------------
def compute_yesno_accuracy(predictions, references):
    """
    Para preguntas yes/no (GT ∈ {yes, no}), comparamos SOLO
    la primera palabra de la predicción.
    """
    correct = 0
    total = 0

    for pred, ref in zip(predictions, references):
        ref_norm = normalize(ref)

        if ref_norm not in {"yes", "no"}:
            continue  # no contamos las que no son yes/no

        total += 1

        pred_first = normalize(pred).split()[0] if pred else ""
        if pred_first == ref_norm:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total


# ---------------------------------------------------------
#  Keyword Accuracy
# ---------------------------------------------------------
def compute_keyword_accuracy(predictions, references):
    """
    Cuenta un acierto si la palabra clave exacta de GT aparece
    dentro de la predicción. Ideal para tareas de clasificación
    con una sola palabra (ej. 'oral', 'urinary', 'hepatic').
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_norm = normalize(pred)
        ref_norm = normalize(ref)

        if ref_norm in pred_norm:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total


# ---------------------------------------------------------
#  Accuracy flexible general
# ---------------------------------------------------------
def compute_general_accuracy_flexible(predictions, references):
    """
    Para GT que NO sean yes/no.
    La predicción se considera correcta si:
      1) GT está dentro de pred (keyword match)
      2) pred está dentro de GT
      3) la primera palabra coincide
      4) textos normalizados son idénticos
    """
    correct = 0
    total = 0

    for pred, ref in zip(predictions, references):
        ref_norm = normalize(ref)

        # ignorar yes/no (se manejan aparte)
        if ref_norm in {"yes", "no"}:
            continue

        total += 1
        pred_norm = normalize(pred)

        # 1) GT dentro de pred
        if ref_norm in pred_norm:
            correct += 1
            continue

        # 2) pred dentro de GT
        if pred_norm in ref_norm:
            correct += 1
            continue

        # 3) primer token coincide
        pred_first = pred_norm.split()[0] if pred_norm else ""
        ref_first = ref_norm.split()[0] if ref_norm else ""
        if pred_first == ref_first and pred_first != "":
            correct += 1
            continue

        # 4) match exacto normalizado
        if pred_norm == ref_norm:
            correct += 1
            continue

    if total == 0:
        return 0.0

    return correct / total


# ---------------------------------------------------------
#  BLEU clásico
# ---------------------------------------------------------
def compute_bleu(predictions, references):
    smooth_func = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = normalize(pred).split()
        ref_tokens = [normalize(ref).split()]

        if len(pred_tokens) == 0:
            scores.append(0)
            continue

        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth_func)
        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)


# ---------------------------------------------------------
#  Short-BLEU (default 5 palabras)
# ---------------------------------------------------------
def compute_bleu_short(predictions, references, max_words=5):
    """
    BLEU usando solo las primeras N palabras de la predicción.
    Mucho más estable para modelos generativos verbosos.
    """
    smooth_func = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        pred_norm = normalize(pred)
        pred_tokens = pred_norm.split()[:max_words]

        ref_tokens = [normalize(ref).split()]

        if len(pred_tokens) == 0:
            scores.append(0)
            continue

        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth_func)
        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)
