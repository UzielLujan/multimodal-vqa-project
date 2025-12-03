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

## **Estructura Propuesta del Repositorio**

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
├── scripts/                 # Scripts de bash slurm para ejecutar experimentos en cluster de cómputo
├── train.py                 # Script principal de ejecución para entrenar
├── inference.py             # Script para generar respuestas sobre el test set
├── README.md
└── environment.yml          # Dependencias del proyecto
```

* `configs/`: Archivos YAML de configuración.
* `data/`: Datos procesados (PathVQA).
* `checkpoints/`: Pesos del encoder visual (SigLIP).
* `src/`: Código fuente modular.
* `scripts/`: Scripts SLURM para el clúster Lab-SB.

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

##  Estado

En desarrollo – versión inicial del proyecto.  

## Despliegue en Clúster (Lab-SB)

### 1. Preparación de Datos (Local)
Los datos y el encoder visual ya están descargados en `data/raw` y `checkpoints/siglip_vision_tower`. Subir la carpeta completa `multimodal_vqa_project`.

### 2. Configuración de LLaMA-3
Como el clúster no tiene internet, **no intentes descargar LLaMA**.
Edita `configs/train_config.yaml` y cambia la ruta de `llm_model_path` a la ubicación absoluta de los pesos en el clúster.

```yaml
paths:
  # Ejemplo:
  llm_model_path: "/home/est_posgrado_uziel.lujan/modelos/llama3-8b-hf"