# **Documentación Técnica: Estado Actual del Sistema VQA (Pre-Migración)**

## **1. Resumen del Sistema**

Este documento describe la arquitectura, implementación y estado del sistema de **Visual Question Answering (VQA)** desarrollado hasta la fecha. El sistema fue diseñado para procesar imágenes y generar respuestas textuales detalladas utilizando un LLM de última generación.

Estado Actual: Deprecado/En Pausa debido a limitaciones de hardware.  
Próximo Paso: Migración del componente LLM (Llama-3 $\\to$ TinyLlama).

## **2. Arquitectura del Pipeline**

El sistema sigue una arquitectura modular secuencial. El flujo actual es:

$$\text{Imagen} (H \times W \times 3) \xrightarrow{\text{SigLIP}} \text{Embeddings Visuales} \xrightarrow{\text{MLP}} \text{Tokens Visuales} \xrightarrow{\text{Concat}} \text{Llama-3-8B}  + \text{LoRA} \rightarrow \text{Texto}$$


### **Módulos Principales (A rescatar)**

1. **Vision Encoder (Congelado):** google/siglip-so400m-patch14-384.  
   * *Justificación:* Seleccionado por superar a CLIP en eficiencia y métricas *zero-shot*.  
   * *Output:* Embeddings de alta calidad.  
2. **Proyector (Entrenable):** Perceptrón Multicapa (MLP).  
   * *Función:* Adaptador de dimensiones. Transforma el espacio latente de SigLIP al espacio de entrada del LLM.  
   * *Configuración:* Capas Lineales \+ Activación GELU.

### **Módulo Problemático (A reemplazar)**

3. **LLM Engine:** meta-llama/Meta-Llama-3-8B.  
   * *Configuración:* Cargado con BitsAndBytes (4-bit quantization) y adaptadores LoRA.

## **3. Estructura del Proyecto (File System)**

El proyecto reside actualmente en el clúster bajo la siguiente estructura de directorios. **El objetivo es tratar de mantener esta estructura intacta durante la migración.**. Sin embargo, cualquier cambio necesario que se requiera será implementado para asegurar funcionamiento óptimo. Se anticipan algunos cambios en `src/models/` para adaptar el LLM y el proyector. 

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
## **4. Diagnóstico de Fallas y Limitaciones**

Durante las pruebas de entrenamiento e inferencia en el clúster, se identificaron los siguientes bloqueos críticos que motivan la migración:

| Error / Síntoma | Causa Raíz | Impacto |
| :---- | :---- | :---- |
| **CUDA OOM (Out Of Memory)** | El modelo base Llama-3 (8B) \+ Optimizer States exceden la VRAM disponible, incluso en 4-bit. | Imposibilidad de completar un *forward pass* con un batch\_size \> 1\. |
| **Loss \= NaN / Inf** | Inestabilidad numérica provocada por la cuantización agresiva (NF4) combinada con gradientes explosivos en el proyector. | El modelo deja de aprender y diverge inmediatamente. |
| **Inferencia Lenta** | El tamaño del modelo satura el ancho de banda de memoria de la GPU. | Latencia inaceptable para pruebas iterativas rápidas. |

## **5. Tecnologías y Librerías Implementadas**

El nuevo sistema debe ser compatible con este stack tecnológico para "rescatar" el entorno:

* **Framework:** PyTorch 2.x. 
* **NumPy:** Numpy 1.26.4 
* **Hugging Face:** transformers, tokenizers, accelerate.  
* **Optimización:** bitsandbytes (para cuantización), peft (para LoRA).  
* **Tracking:** wandb (Weights & Biases) o Tensorboard.

## **6. Decisiones de Diseño (Vigentes)**

A pesar de cambiar el LLM, las siguientes decisiones de diseño se mantienen para la Fase 2 (Migración):

1. **Uso de LoRA:** Se seguirá utilizando *Low-Rank Adaptation* en lugar de *Full Fine-Tuning* para mantener la modularidad y reducir el peso de los checkpoints.  
2. **Dataset Personalizado:** El formato de entrada (Imagen \+ Prompt de Instrucción) se mantiene por su buen desempeño y compatibilidad con el pipeline actual ( lógica en `dataset.py`).  
3. **Métricas:** Se mantienen las metricas de evaluación actuales (BLEU, CIDEr,../) para consistencia.

### **Instrucciones para el Asistente (Contexto de Migración)**

* **Objetivo:** Reemplazar Meta-Llama-3-8B por TinyLlama-1.1B dentro de src/model.py.  
* **Restricción:** Mantener src/dataset.py y el encoder SigLIP sin cambios.  
* **Acción Requerida:** Ajustar las dimensiones del **Proyector MLP** en src/model.py para alinear con el nuevo hidden\_size de TinyLlama.