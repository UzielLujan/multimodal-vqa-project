# **Modelos Multimodales para Visual Question Answering en Imágenes Histopatológicas**

## **Protocolo Oficial del Proyecto**

---

## **1. Resumen General**

Este documento presenta el protocolo oficial para el desarrollo del proyecto final de la materia **Procesamiento de Texto e Imágenes con Deep Learning**, cuyo objetivo es implementar y evaluar un sistema multimodal moderno de **Visual Question Answering (VQA)** aplicado a imágenes histopatológicas.

El proyecto toma como referencia conceptual el paper compartido por el profesor: *Evaluating Low-Cost Multimodal (Visual and Textual) Language Models for Automated Image Understanding in Computational Pathology*, utilizando únicamente los elementos permitidos:

* El dataset **PathVQA**.
* Las métricas **BLEU** y **CIDEr**.

Sin embargo, se propone una mejora metodológica mediante el uso de modelos multimodales modernos basados en **Vision–Language Models (VLMs)**, especialmente **LLaVA 1.5 (SigLIP + LLaMA‑3) con LoRA**, eliminando etapas pesadas como OCR y creación manual de datasets.

El protocolo define la estructura oficial del proyecto, su motivación, metodología, métricas, alcance y diseño experimental.

---

## **2. Objetivo General**

Desarrollar, implementar y evaluar un sistema multimodal avanzado para **Visual Question Answering** en imágenes histopatológicas, utilizando arquitecturas modernas de visión–lenguaje (principalmente **LLaVA 1.5**) y el dataset **PathVQA**, comparando el rendimiento frente a un baseline clásico.

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

Dado un par **(imagen histopatológica, pregunta en lenguaje natural)**, el sistema debe generar una **respuesta correcta y clínicamente coherente**.

Los tipos de preguntas presentes en PathVQA incluyen:

* Sí/No,
* What / Where / How,
* identificación anatómica,
* conteo de estructuras,
* hallazgos diagnósticos.

**Ejemplo:**

```text
Imagen: corte histológico de riñón
Pregunta: "¿Se observan membranas basales engrosadas?"
Respuesta esperada: "Sí"
```

---

## **5. Dataset: PathVQA**

**Fuente:** Hugging Face → `flaviagiammarino/path-vqa`

Características principales:

* ~5,000 imágenes histopatológicas,
* 32,799 pares pregunta–respuesta,
* preguntas de diversos tipos (binarias, abiertas, anatómicas, etc.),
* división estándar train/validation/test.

Ejemplos reales:

```text
Pregunta: "where are liver stem cells (oval cells) located?"
Respuesta: "in the canals of hering"
```

```text
Pregunta: "is embolus derived from a lower-extremity venous thrombus lodged in a pulmonary..."
Respuesta: "yes"
```

---

## **6. Metodología**

### **6.1. Modelo Moderno Propuesto (LLaVA 1.5)**

El modelo principal del proyecto será **LLaVA 1.5**, compuesto por:

* **SigLIP** como encoder visual,
* **MLP multimodal** como proyector imagen→lenguaje,
* **LLaMA‑3** como modelo de lenguaje generativo,
* **LoRA** para fine‑tuning eficiente.

**Pipeline:**

```
Imagen → SigLIP → Proyector MLP → LLaMA‑3 → Respuesta
```

Este modelo ofrece un desempeño notable en tareas de VQA, con una arquitectura más robusta y moderna que el enfoque clásico del paper (CLIP + GPT‑2 + prefix‑tuning).

---

### **6.2. Baseline (Opcional)**

Para fines comparativos se puede incluir un baseline minimalista inspirado en el paper original:

* Encoder: **CLIP/OpenCLIP**,
* Proyector lineal,
* Decoder pequeño (GPT‑2 o similar).

Esto establece un piso de desempeño desde el cual evaluar la ganancia obtenida con LLaVA.

---

## **7. Métricas de Evaluación**

### **7.1. Preguntas Binarias (Sí/No)**

Estas preguntas requieren métricas específicas:

* **Accuracy**,
* **F1‑score (macro)**.

### **7.2. Preguntas Abiertas (What/Where/How)**

Las métricas solicitadas por el profesor serán utilizadas:

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

Evaluación completa del pipeline multimodal con LoRA.

### **Experimento 3 — Comparativa Final**

* Baseline vs LLaVA 1.5,
* Desempeño por tipo de pregunta,
* Análisis de errores y casos difíciles.

---

## **9. Resultados Esperados**

* Mejoras tangibles en BLEU y CIDEr usando LLaVA 1.5.
* Alto desempeño en preguntas binarias mediante Accuracy y F1.
* Modelos modernos capaces de razonamiento clínico básico.

---

## **10. Conclusión Oficial**

Este protocolo presenta la versión formal del proyecto final. El enfoque propuesto es moderno, eficiente y alineado con los lineamientos del profesor, integrando modelos de visión‑lenguaje SOTA como **LLaVA 1.5 (SigLIP + LLaMA‑3 + LoRA)** y métricas adecuadas para cada tipo de pregunta.

El resultado será un estudio multimodal robusto, comparativo y completamente reproducible.

---

## **Documento oficial del proyecto**

Este archivo constituye la referencia central para todas las etapas de desarrollo e implementación del proyecto.

---
