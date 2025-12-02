# 📘 EduIA – Versión 2 (Enfoque OCR + QA Generativo)

## 🎯 Título Tentativo
**EduIA: Asistente Educativo Multimodal Basado en OCR y Generación de Explicaciones**

## 🧩 Idea General
EduIA será un sistema multimodal capaz de **leer una pregunta escrita en una imagen** (foto, captura de pantalla, PDF, ejercicio impreso) mediante un módulo **OCR** y generar una **explicación educativa en texto** utilizando un modelo de lenguaje.

El sistema funciona como un “profesor IA” que interpreta imágenes con enunciados y produce respuestas claras y razonadas.

---

## 🎯 Objetivo
Desarrollar un pipeline multimodal (Imagen → Texto → Explicación) que:

1. **Extraiga la pregunta contenida en una imagen** usando OCR basado en deep learning.  
2. **Interprete la pregunta y genere una explicación** usando un modelo generativo open-source.  
3. Evalúe el desempeño del sistema de forma objetiva mediante métricas para OCR y QA generativo.

---

## 🔧 Componentes Principales

### **1. Módulo OCR (Procesamiento de Imágenes)**
- Detecta y extrae texto presente en imágenes.
- Modelos posibles:
  - **EasyOCR** (baseline)
  - **TrOCR** (Transformer OCR, académico)
  - **Donut** (Document Understanding Transformer, opcional)
- Métricas:
  - **CER (Character Error Rate)**
  - **WER (Word Error Rate)**

### **2. Módulo QA Generativo (Procesamiento de Texto)**
- Recibe la pregunta extraída por OCR.
- Genera una explicación educativa en español.
- Modelos evaluados:
  - **FLAN-T5-XL**
  - **BLOOMZ-3B**
  - **LLaMA 3** (opcional)
- Evaluación:
  - **ROUGE**, **BERTScore**, coherencia semántica.

### **3. Integración Multimodal**
- Pipeline unificado:
    **Imagen** → *OCR* → **texto** → *QA generativo* → **explicación.**
    
- Validación de coherencia final entre texto extraído y explicación generada.

---

## 📊 Evaluación
El proyecto se evaluará con tres componentes:

1. **OCR**  
 - CER, WER sobre un subconjunto controlado de imágenes.

2. **QA Generativo**  
 - Comparación con respuestas de referencia (SQuAD-es, MLQA-es).  
 - BERTScore o ROUGE para medir calidad.

3. **Evaluación global**  
 - Exactitud OCR → calidad de respuesta.  
 - Caso de estudio donde imagen y explicación se relacionan correctamente.

---

## 🌟 Entregable Final
- Demo en interfaz web simple (Gradio/Streamlit).  
- Pipeline modular reproducible (imagen → texto → explicación).  
- Comparación entre modelos OCR y QA.  
- Análisis de métricas y discusión académica.

---

## 📝 Conclusión
Esta versión del proyecto conserva el enfoque multimodal exigido por la materia, reduce la complejidad técnica respecto a la generación de imágenes, permite una evaluación clara y objetiva, y mantiene un espíritu de “Asistente Educativo Inteligente” sólido y demostrable.

