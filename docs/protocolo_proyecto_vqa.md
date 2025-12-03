# **Modelos Multimodales para Visual Question Answering en Imágenes Histopatológicas**

## **Protocolo Oficial del Proyecto**

---

## **1. Resumen General**

Este documento presenta el protocolo oficial para el desarrollo del proyecto final de la materia **Procesamiento de Texto e Imágenes con Deep Learning**, cuyo objetivo es implementar y evaluar un sistema multimodal moderno de **Visual Question Answering (VQA)** aplicado a imágenes histopatológicas.

El proyecto toma como referencia conceptual el paper: *Evaluating Low-Cost Multimodal (Visual and Textual) Language Models for Automated Image Understanding in Computational Pathology*, utilizando únicamente los elementos permitidos:

* El dataset **PathVQA**.
* Las métricas **BLEU** y **CIDEr**.

Sin embargo, se propone una mejora metodológica mediante el uso de modelos multimodales modernos basados en **Vision–Language Models (VLMs)**, específicamente **LLaVA 1.5 (SigLIP + LLaMA‑3) con LoRA**, eliminando propuestas anteriores que incluían etapas pesadas como OCR y creación manual de datasets.

El protocolo define la estructura oficial del proyecto, su motivación, metodología, métricas, alcance y diseño experimental, enfocandose en el nucleo multimodal:

```bash
Imagen histopatológica + Pregunta → Modelo Multimodal → Respuesta generada
```

---

## **2. Objetivo General**

Desarrollar, implementar y evaluar un sistema multimodal moderno para **Visual Question Answering** en imágenes histopatológicas, utilizando arquitecturas recientes de visión–lenguaje (principalmente **LLaVA 1.5**) y el dataset **PathVQA**, para producir respuestas clínicas plausibles ante preguntas derivadas de imágenes histopatológicas.

---

## **3. Motivación y Justificación**

El campo de la patología digital ha avanzado significativamente gracias al uso de modelos multimodales capaces de integrar señales visuales y lingüísticas. Sin embargo, muchos estudios académicos aún se basan en arquitecturas limitadas como **CLIP + GPT‑2**, las cuales presentan:

* baja capacidad de razonamiento,
* escasa compatibilidad entre espacios visuales y lingüísticos,
* integración superficial,
* dificultades para generalizar en tareas clínicas.

Modelos modernos como **LLaVA 1.5**, basados en **SigLIP** y **LLaMA‑3**, resuelven estas limitaciones gracias a:

* un encoder visual SOTA,
* un proyector multimodal específico,
* un LLM moderno con fuertes capacidades de generación,
* fine‑tuning eficiente mediante **LoRA**.

El uso de estos modelos permite construir un sistema robusto, reproducible y con resultados superiores en la tarea de VQA.

---

## **4. Definición del Problema**
El problema de VQA consiste en que dado un par:
* **Imagen histopatológica**
* **Pregunta en lenguaje natural**

el sistema debe generar una **respuesta correcta y clínicamente coherente**.

Los tipos de preguntas presentes en el dataset PathVQA incluyen:

* Sí/No,
* What / Where / How,
* identificación anatómica,
* conteo de elementos,
* hallazgos diagnósticos.

**Ejemplo simulado:**

```bash
Imagen: Corte histológico de riñón
Pregunta: "¿Se observan membranas basales engrosadas?"
Respuesta esperada: "Sí"
```

---

## **5. Dataset: PathVQA**

**Fuente:** Hugging Face → `flaviagiammarino/path-vqa`. 
[https://huggingface.co/datasets/flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

Características principales:

* ~5,000 imágenes histopatológicas,
* 32,799 pares pregunta–respuesta,
* Preguntas de diversos tipos (binarias, abiertas, anatómicas, etc.),
* División estándar train/validation/test.

Ejemplos reales del dataset:

```bash
Pregunta: "where are liver stem cells (oval cells) located?"
Respuesta: "in the canals of hering"
```

```bash
Pregunta: "is embolus derived from a lower-extremity venous thrombus lodged in a pulmonary..."
Respuesta: "yes"
```

---

## **6. Metodología**

### **6.1. Modelo Moderno Propuesto (LLaVA 1.5)**

El modelo seleccionado para este proyecto es **LLaVA 1.5**, una arquitectura moderna de visión–lenguaje compuesta por:

* **SigLIP** como encoder visual SOTA,
* **MLP multimodal** como proyector imagen → lenguaje,
* **LLaMA‑3** como modelo de lenguaje generativo,
* **LoRA** para fine‑tuning eficiente.

**Pipeline:**

```bash
Imagen → SigLIP → Proyector MLP → LLaMA‑3 → Respuesta
```

Este modelo ofrece un desempeño notable en tareas de VQA, con una arquitectura más robusta y moderna que el enfoque clásico del paper (CLIP + GPT‑2 + prefix‑tuning), a cual presenta limitaciones de compatibilidad, capacidad de razonamiento y desempeño general.

---

### **6.2. Baseline (Opcional)**

Para fines comparativos se puede incluir un baseline minimalista inspirado en el paper original:

* Encoder visual: **CLIP/OpenCLIP**,
* Proyector lineal MLP,
* Decoder pequeño (GPT‑2 o similar).

Esto establece un piso de desempeño desde el cual evaluar la ganancia obtenida con LLaVA.

---

## **7. Métricas de Evaluación**

### **7.1. Preguntas Binarias (Sí/No)**

Estas preguntas requieren métricas específicas:

* **Accuracy**,
* **F1‑score (macro)**.

### **7.2. Preguntas Abiertas (What/Where/How)**

Las métricas sugeridas por el profesor serán utilizadas:

* **BLEU**,
* **CIDEr**.

Métricas adicionales opcionales:

* **ROUGE‑L**,
* **BERTScore**.

Dividir el análisis por tipo de pregunta evita conclusiones erróneas, ya que métricas como BLEU y CIDEr no son adecuadas para respuestas cortas como “yes” o “no”.

---

## **8. Diseño Experimental**

### **Experimento 1 — Baseline (Opcional)**

Implementación mínima con CLIP + GPT‑2 o CLIP + MLP.

### **Experimento 2 — Modelo Moderno (LLaVA 1.5)**

Evaluación e Implementación de **LLaVA 1.5 + LoRA**, completa sobre **PathVQA**.

### **Experimento 3 — Comparativa Final**

* Baseline vs LLaVA 1.5 (opcional o comparando con resultados reportados en el paper original).
* Desempeño por tipo de pregunta.
* Análisis de errores y casos difíciles.

---

## **9. Resultados Esperados**

* Mejoras claras en BLEU y CIDEr usando LLaVA 1.5 frente al baseline del paper original.
* Alto desempeño en preguntas binarias mediante Accuracy y F1.
* Modelos modernos capaces de razonamiento clínico básico.

---

## **10. Conclusión Oficial**

Este protocolo presenta la versión oficial del proyecto final. El enfoque propuesto es moderno, eficiente y alineado con los lineamientos del curso, integrando modelos de visión‑lenguaje SOTA como **(SigLIP + LLaMA‑3 + LoRA)** y métricas adecuadas para cada tipo de pregunta.

El resultado será un estudio multimodal robusto, comparativo y completamente reproducible.

---

## **Estructura propuesta de carpetas en el proyecto**

```bash
multimodal_vqa_project/
├── configs/               # Archivos .yaml con hiperparámetros (LR, batch_size, LoRA r, etc.)
├── checkpoints/           # Aquí se guardan los pesos (modelos .pt, adaptadores LoRA)
├── data/
│   ├── raw/                 # Datos originales (imágenes y JSONs de PathVQA)
│   └── processed/           # Datasets tokenizados o tensores pre-procesados
├── docs/
│   ├── Protocolo_Proyecto_VQA.md
│   ├── Bitacora_tecnica.md
│   └── Indicaciones_Proyecto_final.md
├── logs/                    # Logs de entrenamiento 
├── notebooks/               # EDA y prototipado rápido
├── results/                 # Salidas finales: Gráficas generadas, tablas de métricas, CSVs de predicciones
├── src/
│   ├── __init__.py
│   ├── data/                # Loaders, transformaciones y clases Dataset custom
│   ├── models/              # Definición de la arquitectura (LLaVA interface, peft config)
│   ├── training/            # Bucles de entrenamiento (Trainer class, validación)
│   ├── evaluation/          # Scripts de métricas (BLEU, CIDEr, Accuracy)
│   └── utils/               # Funciones auxiliares (seeding, visualización, logger setup)
├── scripts/                 # Scripts de bash para ejecutar experimentos en cluster de cómputo
├── train.py                 # Script principal de ejecución para entrenar
├── inference.py             # Script para generar respuestas sobre el test set
├── README.md
└── environment.yml          # Dependencias del proyecto
```

---
