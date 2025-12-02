# **Visual Question Answering en Imágenes Histopatológicas**

### **Proyecto Final – Procesamiento de Texto e Imágenes con Deep Learning**

---

## **Descripción General**

Este repositorio contiene la implementación del proyecto final del curso **Procesamiento de Texto e Imágenes con Deep Learning**, cuyo objetivo es desarrollar un sistema moderno de **Visual Question Answering (VQA)** aplicado a **imágenes histopatológicas**.

El proyecto utiliza modelos multimodales recientes de visión–lenguaje, en particular **LLaVA 1.5 (SigLIP + LLaMA-3) con LoRA**, evaluados sobre el dataset **PathVQA**.

---

## **Objetivo del Proyecto**

Construir y evaluar un sistema multimodal capaz de responder preguntas en lenguaje natural basadas en imágenes histológicas, comparando:

* Un baseline clásico (opcional) basado en CLIP + GPT-2.
* Un modelo moderno VLM: **LLaVA 1.5**.

---

## **Dataset**

**PathVQA** (HuggingFace):

* ~5,000 imágenes histopatológicas
* 32,799 pares Pregunta–Respuesta
* Preguntas: Sí/No, What/Where/How, hallazgos diagnósticos, etc.

Link: [https://huggingface.co/datasets/flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

---

## **Modelo Principal**

El modelo seleccionado es **LLaVA 1.5**, compuesto por:

* **SigLIP** como encoder visual
* **MLP** como módulo de fusión visión→lenguaje
* **LLaMA-3** como modelo generador
* **LoRA** para fine-tuning eficiente

Pipeline general:

```
Imagen → SigLIP → MLP multimodal → LLaMA-3 → Respuesta generada
```

---

## **Métricas**

Se emplean métricas separadas por tipo de pregunta:

### Para Sí/No:

* Accuracy
* F1-score (macro)

### Para preguntas abiertas:

* BLEU
* CIDEr
* (Opcional) ROUGE-L, BERTScore

---

## **📁 Estructura Propuesta del Repositorio**

```
VQA/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── protocolo_proyecto_vqa.md
│   └── documentacion.md
├── src/
│   ├── loaders/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── notebooks/
├── results/
├── README.md
└── environment.yml
```

---

## **Estado del Proyecto**

El proyecto se encuentra en fase de organización inicial. Próximos pasos:

* Implementar carga de PathVQA.
* Configurar modelo LLaVA 1.5.
* Añadir soporte para LoRA.
* Definir experimentos y métricas.

---

## **Autores**

- **Uziel Isaí Luján López**  
- **Diego Paniagua Molina**  

