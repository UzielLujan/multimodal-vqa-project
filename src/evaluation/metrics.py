import evaluate
import numpy as np

# Cargar métricas de HuggingFace (se descargarán la primera vez)
try:
    bleu_metric = evaluate.load("sacrebleu")
    # cider_metric = evaluate.load("cider") # Ojo: CIDEr a veces requiere configuración extra de Java
except Exception as e:
    print(f"⚠️ Advertencia: No se pudieron cargar métricas de evaluate offline: {e}")

def compute_vqa_accuracy(predictions, references):
    """
    Calcula Accuracy simple para preguntas de Sí/No.
    Normaliza el texto (lowercase, strip) para ser justo.
    """
    correct = 0
    total = 0
    
    for pred, ref in zip(predictions, references):
        p_clean = pred.lower().strip()
        r_clean = ref.lower().strip()
        
        # Lógica binaria estricta
        if r_clean in ['yes', 'no']:
            total += 1
            if p_clean == r_clean:
                correct += 1
                
    if total == 0:
        return 0.0
        
    return correct / total

def compute_bleu(predictions, references):
    """
    Calcula BLEU score para preguntas abiertas.
    """
    # SacreBLEU espera referencias como lista de listas [[ref1], [ref2]]
    formatted_refs = [[r] for r in references]
    
    results = bleu_metric.compute(predictions=predictions, references=formatted_refs)
    return results['score']