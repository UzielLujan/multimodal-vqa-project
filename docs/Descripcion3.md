# Proyecto Final – Versión 3  
## **Modelos Multimodales para Visual Question Answering en Imágenes Histopatológicas**

Esta es la descripcion detallada del proyecto final del curso **Procesamiento de Texto e Imágenes con Deep Learning**, la tercera versión propuesta.

### Resumen General  
Esta tercera versión redefine el enfoque del proyecto hacia un sistema multimodal de **Visual Question Answering (VQA)** basado en deep learning.  
El objetivo principal es **replicar y mejorar metodológicamente** algunos resultados inspirados en el paper compartido por el profesor, utilizando únicamente:

- El dataset **PathVQA**.
- Las métricas **BLEU** y **CIDEr**.

Se elimina por completo la parte de OCR y la creación manual de datasets, reduciendo drásticamente la carga de trabajo y centrándonos en el componente central multimodal: **imagen + pregunta → generación de respuesta**.

---

# **1. Objetivo General**
Desarrollar, evaluar y comparar modelos multimodales modernos para la tarea de **Visual Question Answering** utilizando el dataset PathVQA, con el fin de reproducir parcialmente la metodología original del paper y aplicar mejoras con modelos actuales.

---

# **2. Descripción del Problema**
La tarea de **VQA** consiste en que el sistema reciba:

- Una **imagen histopatológica**.
- Una **pregunta en lenguaje natural** acerca de la imagen.

Y genere como salida:

- Una **respuesta coherente, correcta y clínicamente pertinente**.

El sistema debe aprender a integrar información visual y textual simultáneamente, logrando decisiones contextualizadas.

Ejemplo $1$ simulado:

```bash
Imagen: corte histológico de riñón
Pregunta: "¿Se observan membranas basales engrosadas?"
Respuesta: "Sí"
```

---

# **3. Dataset: PathVQA**
Fuente: Hugging Face → `flaviagiammarino/path-vqa`

Este dataset contiene:

- ~5,000 imágenes médicas (microscopía, IHC, tinciones).
- 32,799 pares **Pregunta–Respuesta**.
- Preguntas de varios tipos:
  - Sí/No
  - What / Where / How
  - Conteo
  - Hallazgos patológicos
- División original en train/validation/test.

Este dataset es **ideal para VQA**, está completamente pareado y listo para usar sin modificaciones.

Ejemplos reaeles del dataset PathVQA:

```bash
Imagen
Pregunta: "where are liver stem cells (oval cells) located?"
Respuesta: "in the canals of hering"
```

```bash
Imagen
Pregunta: "is embolus derived from a lower-extremity deep venous thrombus lodged in a pulmonary.."
Respuesta: "yes"
```
```bash
Imagen
Pregunta: "where is this from?"
Respuesta: "gastrointestinal system"
```
---

# **4. Metodología**

## 4.1. Baseline (Opcional replicación parcial del paper)
Como punto de partida se implementará un modelo baseline inspirado en el paper, pero no necesariamente idéntico:

- Encoder visual: **CLIP** o **OpenCLIP-ViT-B/32**
- Encoder textual: tokenizador estándar (pregunta)
- Fusión multimodal: concatenación + proyector lineal
- Generación de respuesta: modelo tipo decoder (p. ej., GPT-2 pequeño)

Esto establece una línea base para las métricas BLEU y CIDEr.

---

## 4.2. Mejora Metodológica (Extensión de la Replicación)
Se propondrán modelos VQA **modernos**, más potentes y fáciles de implementar:

### Opciones recomendadas:
- **BLIP-2** (Q-former + LLaMA/T5)
- **LLaVA 1.5** (visual encoder + instruction-tuned LLM)
- **MiniGPT-4**
- **InstructBLIP**

Estos modelos utilizan:
- Encoders visuales SOTA (EVA-CLIP, SigLIP)
- LLMs modernos (LLaMA, Vicuna, T5)
- Arquitecturas de fusión avanzadas

Y permiten evaluar mejoras reales vs baseline.

---

# **5. Métricas de Evaluación**

Siguiendo estrictamente las indicaciones del profesor:

### BLEU  
Mide solapamiento de n-gramas entre la respuesta generada y la respuesta correcta.

### CIDEr  
Mide similitud semántica ponderada entre respuestas, enfocándose en términos relevantes.

Ambas métricas ya están implementadas en:
- `torchmetrics`
- `evaluate` de HuggingFace
- Paquetes NLP estándar

---

# **6. Diseño Experimental**

### Experimento 1 — Baseline  
Evaluar el modelo básico tipo CLIP + GPT-2 o CLIP + MLP + decoder.

### Experimento 2 — Modelo Moderno  
Implementación de un modelo avanzado como BLIP-2 o LLaVA.

### Experimento 3 — Comparativa Final  
Comparar métricas BLEU y CIDEr entre:

- Baseline reproducido
- Modelos modernos
- Variación por tipo de pregunta (Yes/No vs What vs Where)

---

# **7. Resultados Esperados**
- Incremento notable en BLEU y CIDEr al usar modelos modernos.  
- Mejor desempeño en preguntas abiertas (“What/Where”) con modelos tipo BLIP-2.  
- Modelos tipo LLaVA superan a CLIP+GPT-2 especialmente en razonamiento clínico simple.

---

# **8. Conclusión**
Esta versión del proyecto permite:

- Reproducir experimentalmente elementos del paper.
- Aprovechar un dataset público completo.
- Reducir la carga de trabajo eliminando OCR y anotaciones manuales de la versión original del proyecto final del curso.
- Implementar modelos multimodales actuales.
- Evaluar de forma directa y objetiva mediante BLEU y CIDEr.


