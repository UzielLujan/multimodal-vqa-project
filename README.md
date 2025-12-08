# **Visual Question Answering en Imágenes Histopatológicas**

### **Proyecto Final – Procesamiento de Texto e Imágenes con Deep Learning**

---

## **Descripción General**

Este repositorio contiene la implementación final del proyecto para el curso **Procesamiento de Texto e Imágenes con Deep Learning**. El objetivo es un sistema de **Visual Question Answering (VQA)** eficiente aplicado a **imágenes histopatológicas**.

Debido a restricciones de hardware en el entorno de cómputo (GPUs de 24GB sin acceso a internet), el proyecto evolucionó de una propuesta inicial basada en LLaMA-3 hacia una arquitectura optimizada utilizando **TinyLlama-1.1B** y **CLIP-ViT-Large**, entrenada mediante **Full Fine-Tuning**.

---

## **Arquitectura del Modelo**

El sistema implementado sigue el paradigma LLaVA pero adaptado para eficiencia:

*   **Encoder Visual:** `CLIP-ViT-Large-patch14-336` (Frozen).
*   **Conector:** MLP (Proyector) entrenable.
*   **Modelo de Lenguaje:** `TinyLlama-1.1B-Chat` (Full Fine-Tuning).
*   **Estrategia de Entrenamiento:** Full Fine-Tuning del LLM y el proyector, manteniendo el encoder visual congelado.

---

## **Dataset**

**PathVQA** (HuggingFace):
*   ~5,000 imágenes histopatológicas.
*   32,632 pares Pregunta–Respuesta.
*   Tipos de preguntas: Binarias (Yes/No) y Abiertas (Open-ended).

---

## **Estructura del Repositorio**

```bash
multimodal-vqa-project/
├── configs/               # Configuraciones YAML (entrenamiento e inferencia)
├── data/                  # Datos (raw y processed)
├── docs/                  # Documentación y bitácoras del proyecto
├── notebooks/             # Análisis Exploratorio de Datos (EDA)
├── report/                # Código fuente LaTeX del reporte final
├── results/               # Salidas: logs, plots, predicciones y métricas
│   ├── plots/
│   ├── predictions/
│   └── summary/
├── scripts/               # Scripts Bash para ejecución en clúster/local
├── src/                   # Código fuente Python
│   ├── data/              # Dataset y Dataloaders
│   ├── evaluation/        # Scripts de métricas y visualización
│   ├── models/            # Definición de la arquitectura (TinyLlama + CLIP)
│   ├── training/          # Loop de entrenamiento y callbacks
│   └── utils/             # Utilidades generales
└── environment.yml        # Dependencias Conda
```

---

## **Resultados Principales**

El modelo final (TinyLlama-CLIP-768) superó al baseline (versión 1024 tokens) en estabilidad y métricas clave:

| Métrica | TinyLlama-CLIP-768 (Final) |
| :--- | :--- |
| **Accuracy (Yes/No)** | **86.18%** |
| **Keyword Accuracy** | **57.76%** |
| **Flexible Accuracy** | **32.90%** |

---

## **Instrucciones de Uso**

### 1. Configuración del Entorno
```bash
conda env create -f environment.yml
conda activate vqa-tiny-env
```

### 2. Preparación de Datos
Descarga y preprocesamiento del dataset PathVQA:
```bash
bash scripts/run_data.sh
```

### 3. Entrenamiento
Ejecutar el entrenamiento (Full Fine-Tuning):
```bash
bash scripts/run_train.sh
```

### 4. Inferencia y Evaluación
Generar predicciones y calcular métricas sobre el set de validación:
```bash
bash scripts/run_inference_eval.sh
```

---

## **Autores**

- **Uziel Isaí Luján López**
- **Diego Paniagua Molina**

---
*Centro de Investigación en Matemáticas (CIMAT) - 2025*
