# Bitácora Final del Proyecto Multimodal VQA

## 1. Estado previo del pipeline

Al retomar el proyecto, el estado general del sistema de inferencia y evaluación era funcional solo de manera parcial. La estructura base del repositorio ya incluía módulos para entrenamiento, carga de datos y algunos scripts de inferencia iniciales, pero la segunda mitad del pipeline —es decir, la parte crítica relacionada con inferencia en modelos entrenados, generación de predicciones masivas, métricas avanzadas y análisis comparativo— estaba incompleta o no operativa.

En términos generales, las observaciones iniciales fueron:

* Existía un script `inference_demo.py`, pero no era robusto, no era reproducible y no podía utilizarse sobre el dataset **PathVQA** en su formato Arrow.
* No existía una ruta clara para ejecutar inferencia masiva ni para guardar salidas ordenadas.
* El código de evaluación carecía de métricas apropiadas para VQA médico, dependiendo únicamente de BLEU, el cual no es adecuado para respuestas cortas como el caso del dataset **PathVQA**.
* No se contaba con una forma automatizada de comparar modelos entre sí.
* Faltaba una herramienta que permitiera analizar ejemplos individuales de predicción en profundidad.
* No había mecanismo para generar visualizaciones de resultados.
* No existía documentación sobre cómo debía ejecutarse la inferencia o el análisis final.

A partir de este estado, se construyó y refinó un pipeline completo de:

* Inferencia por muestra (visual y validable manualmente)
* Inferencia masiva en cluster
* Métricas avanzadas diseñadas especialmente para VQA médico
* Análisis detallado y comparación entre modelos
* Visualización de métricas
* Documentación para reproducibilidad



---

## 2. Pipeline actualizado: Inferencia, Evaluación y Métricas

Esta sección describe el **pipeline completo y unificado** que se desarrolló para la segunda mitad del proyecto. Aquí se explica cómo funcionan los módulos de inferencia, cómo se realiza la evaluación masiva, qué métricas se utilizan y cómo se integran en un flujo coherente. Este bloque representa el corazón del sistema final de VQA multimodal sobre PathVQA.

El pipeline completo consta de cuatro etapas principales:

1. **Inferencia por muestra** (inspección cualitativa)
2. **Inferencia masiva** (predicción sobre todo el split)
3. **Cálculo de métricas avanzadas** (evaluación cuantitativa)
4. **Comparación y visualización** (análisis entre modelos)

---

### 2.1 Inferencia por muestra — `inference_sample.py`

Este módulo permite inspeccionar visualmente el comportamiento del modelo.

**Características principales:**

* Carga imágenes desde PathVQA en formato Arrow.
* Obtiene la pregunta y la respuesta real.
* Genera una predicción textual con el modelo.
* Produce un panel tipo "paper mode":

  * Imagen de entrada
  * Pregunta
  * Ground Truth (GT)
  * Predicción del modelo
* Guarda cada panel en `results/samples/`.

**Propósitos:**

* Verificar cualitativamente el modelo.
* Identificar patrones de error.
* Generar ejemplos listos para el reporte.

---

### 2.2 Inferencia masiva — `inference_eval.py`

Este módulo es responsable de evaluar el modelo sobre miles de muestras.

**Salida generada:**

* CSV con todas las predicciones:
  `results/predictions/predictions_<modelo>_<split>.csv`
* JSON con métricas globales:
  `results/summary/metrics_<modelo>_<split>.json`

**Características clave:**

* Ejecución optimizada en SLURM + GPU.
* Decodificación determinista (sin temperatura, sin sampling).
* Control de longitud con `max_new_tokens` y `trim_words`.
* Manejo seguro de imágenes truncadas (`PIL.TiffImagePlugin`).

Este módulo es el que produce la base para todo análisis cuantitativo y comparativo.

---

### 2.3 Métricas avanzadas — definidas en `metrics.py`

Las métricas originales del proyecto eran insuficientes para PathVQA, ya que BLEU no captura correctamente respuestas cortas ni preguntas binarias. Por ello se implementó un conjunto de métricas **específicas para VQA médico**, junto con sus definiciones matemáticas.

A continuación se presentan.

---

#### **1. Accuracy Yes/No** — para preguntas binarias

Compara solo el **primer token** de la predicción.

Sea $y_i^{GT} \in \{\text{yes},\text{no}\}$ y $f_i^{pred}$ la secuencia generada:


$$Accuracy_{Yes/No}
= \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big( y_i^{GT} = \text{first}(f_i^{pred}) \big)$$


**Intuición:** si la predicción comienza con "yes" o "no" correctamente, se considera correcta aunque el modelo genere texto adicional.

---

#### **2. Keyword Accuracy** — concepto principal correcto

Evalúa si la palabra clave del ground truth aparece en la predicción.

Sea $k_i$ la palabra clave del GT:

$$
Accuracy_{Keyword}
= \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\big( k_i \subseteq f_i^{pred} \big)
$$

**Motivación:** muchas respuestas de PathVQA son etiquetas anatómicas de una sola palabra.

---

#### **3. Accuracy Flexible** — coincidencia parcial o semántica

Permite matching exacto, parcial o basado en la primera palabra.

Sea la función auxiliar:

$$
match(y_i^{GT}, f_i^{pred}) =
\begin{cases}
1 & y_i^{GT} = f_i^{pred} \\
1 & y_i^{GT} \subseteq f_i^{pred} \\
1 & \text{first}(y_i^{GT}) = \text{first}(f_i^{pred}) \\
0 & \text{otro caso}
\end{cases}
$$

Entonces:

$$
Accuracy_{Flexible} =
\frac{1}{N} \sum_{i=1}^{N} match(y_i^{GT}, f_i^{pred})
$$

**Intuición:** mide si la predicción preserva correctamente la idea principal.

---

#### **4. BLEU-short (máx. 5 palabras)** — BLEU adaptado a VQA

BLEU clásico penaliza fuertemente respuestas largas; por ello se creó BLEU-short.

$$
BLEU_{short} = BP \cdot \exp\left( \sum_{n=1}^4 w_n \log p_n^{(5)} \right)
$$

donde:

* $p_n^{(5)}$: precisión de n-gramas evaluando solo los primeros 5 tokens.
* $w_n = 1/4$: pesos uniformes.
* $BP$: brevity penalty.

---

#### **5. BLEU clásico** — por completitud

Incluido solo para comparación con literatura previa.

$$
BLEU = BP \cdot \exp\left( \sum_{n=1}^4 w_n \log p_n \right)
$$

---

### 2.4 Herramientas de análisis — `evaluation_tools.py`

Este módulo integra las funciones de evaluación sin calcular directamente las métricas.

Incluye:

* `load_predictions()` — carga y normaliza el CSV.
* `compute_all_metrics()` — invoca funciones de `metrics.py`.
* `find_top_errors()` — identifica ejemplos representativos.
* `analyze_predictions()` — genera un análisis completo para un solo modelo.

---

### 2.5 Comparación entre modelos — `evaluation_batch.py`

Automatiza el proceso de comparar distintos experimentos.

Produce:

* `summary_models.csv`
* `summary_models.json`

Utiliza automáticamente todos los archivos de `results/predictions/`.

---

### 2.6 Visualización — `plot_metrics.py`

Genera figuras para incluir directamente en el reporte:

* Gráficas de barras
* Gráficas de dispersión (scatter)

Se guardan en:

```
results/plots/
```

Estas visualizaciones ayudan a interpretar diferencias entre modelos, como:

* estabilidad
* precisión en preguntas binarias
* identificación de palabras clave
* comportamiento generativo


Con estas herramientas combinadas, el pipeline ofrece análisis:

* Cuantitativo
* Cualitativo
* Comparativo
* Visual

El resultado es un sistema completo y reproducible para evaluar modelos VQA multimodales entrenados en PathVQA.

---


## 4. Resultados principales de los modelos evaluados

Esta sección presenta los resultados cuantitativos obtenidos mediante el pipeline de inferencia y evaluación. Se comparan directamente los dos modelos entrenados:

* **TinyLlama-CLIP 768 tokens** (modelo final con regularización y ajustes maduros)
* **TinyLlama-CLIP 1024 tokens** (primer experimento funcional, sin regularización y sin loss de evaluación correcto)

Las métricas provienen de los archivos JSON generados por `inference_eval.py` y del archivo comparativo `summary_models.csv`.

---

### 4.1 Métricas individuales por modelo

#### **Modelo: TinyLlama-CLIP 768** ("final_model")

Fuente: `metrics_tinyllama-clip-768_validation.json`

* **Samples:** 6259
* **Accuracy Yes/No:** 0.86176
* **Accuracy Flexible:** 0.32897
* **Keyword Accuracy:** 0.57757
* **BLEU-short:** 0.03843
* **BLEU:** 0.02205

Este modelo muestra un desempeño sólido: estabilidad en preguntas binarias y capacidad razonable para identificar palabras clave relevantes.

---

#### **Modelo: TinyLlama-CLIP 1024**

Fuente: `metrics_tinyllama-clip-1024_validation.json`

* **Samples:** 6259
* **Accuracy Yes/No:** 0.82272
* **Accuracy Flexible:** 0.32578
* **Keyword Accuracy:** 0.56926
* **BLEU-short:** 0.03831
* **BLEU:** 0.02433

El modelo 1024, pese a haber sido la primera versión funcional, presenta métricas ligeramente inferiores debido a:

* falta de regularización,
* longitud máxima mayor en entrenamiento inicial,
* ausencia de evaluación de pérdida durante el entrenamiento.

---

### 4.2 Comparación general entre modelos

Los valores siguientes provienen de `summary_models.csv`.

#### **Tabla comparativa de métricas**

| Métrica           | TinyLlama-CLIP 768 | TinyLlama-CLIP 1024 |
| ----------------- | ------------------ | ------------------- |
| Accuracy Yes/No   | 0.86176            | 0.82272             |
| Accuracy Flexible | 0.32897            | 0.32578             |
| Keyword Accuracy  | 0.57757            | 0.56926             |
| BLEU-short        | 0.03843            | 0.03831             |
| BLEU clásico      | 0.02205            | 0.02433             |

---

### 4.3 Interpretación de resultados

1. **El modelo TinyLlama-CLIP 768 es consistentemente superior** en casi todas las métricas relevantes para VQA médico.
2. **Yes/No Accuracy** muestra la diferencia más clara: el modelo 768 es más estable en decisiones binarias.
3. **Keyword Accuracy** evidencia que el modelo 768 identifica mejor el concepto anatómico principal.
4. **Accuracy Flexible** es comparable en ambos modelos, lo que indica similitud en coincidencias parciales.
5. **BLEU-short y BLEU clásico** no son métricas fuertes para este dominio, pero ayudan a comprobar comportamiento generativo.
6. El modelo 1024 muestra valores razonables, pero no supera al 768 debido a:

   * falta de regularización,
   * ausencia de métricas de evaluación durante su entrenamiento,
   * mayor tendencia a generar respuestas largas.

---

### 4.4 Conclusión breve

El análisis cuantitativo indica que:

> **El modelo TinyLlama-CLIP 768 representa la mejor versión entrenada del sistema, superando consistentemente al modelo 1024 en métricas críticas para VQA médico.**

Esta sección sirve como base para el análisis de errores (Sección 5) y para la discusión final del reporte.


## 5. Análisis de errores (Top‑K representativos por modelo)

Esta sección presenta un **análisis cualitativo de errores**, utilizando los archivos producidos por `analyze_single.py` para ambos modelos. El objetivo es identificar patrones comunes en las fallas, ilustrar ejemplos concretos y destacar diferencias entre modelos.

Se seleccionaron **5 errores representativos por modelo**, priorizando claridad y diversidad de patrones.

---

## 5.1 Errores representativos — Modelo TinyLlama‑CLIP 768

**Archivo fuente:** `analysis_tinyllama-clip-768_validation.json`

A continuación se presentan cinco casos donde el modelo falla por razones distintas: respuestas incompletas, descripciones largas, confusiones anatómicas o incapacidad para capturar detalles relevantes.

### **Ejemplo 1** — Confusión anatómica sutil

* **Pregunta:** *where is this?*
* **GT:** *oral cavity*
* **Predicción:** *oral cavity and pharyngeal region with soft tissue involvement*
* **Análisis:** El modelo identifica correctamente la región general, pero añade términos adicionales que no forman parte del GT, lo cual penaliza las métricas basadas en coincidencia exacta.

### **Ejemplo 2** — Respuesta redundante con ruido semántico

* **Pregunta:** *what organ is shown?*
* **GT:** *liver*
* **Predicción:** *liver with metastatic appearance and surrounding capsule involvement*
* **Análisis:** La palabra clave está presente, pero la extensión innecesaria del texto degrada puntuaciones como BLEU y Accuracy Flexible.

### **Ejemplo 3** — Error por sobre‑explicación

* **Pregunta:** *is urinary present?*
* **GT:** *yes*
* **Predicción:** *yes this is a simple cystadenoma...*
* **Análisis:** El primer token es correcto (beneficia Yes/No Accuracy), pero el resto de la respuesta contiene narrativa clínica no relacionada.

### **Ejemplo 4** — Interpretación alternativa del mismo concepto

* **Pregunta:** *what tissue is this?*
* **GT:** *hepatobiliary*
* **Predicción:** *hepatobiliary system with metastatic cancer of the liver and spleen...*
* **Análisis:** Coincide la palabra clave, pero la adición de condiciones clínicas inventadas causa una penalización fuerte.

### **Ejemplo 5** — Error por descripción demasiado general

* **Pregunta:** *where is this located?*
* **GT:** *renal cortex*
* **Predicción:** *kidney region*
* **Análisis:** El modelo identifica correctamente el órgano, pero no la sub‑estructura, lo que afecta Keyword Accuracy.

---

## 5.2 Errores representativos — Modelo TinyLlama‑CLIP 1024

**Archivo fuente:** `analysis_tinyllama-clip-1024_validation.json`

El modelo 1024 tiende a cometer errores similares al 768, pero de forma más marcada: respuestas más largas, mayor ruido semántico y menor estabilidad en preguntas binarias.

### **Ejemplo 1** — Predicción larga con ruido clínico

* **Pregunta:** *what is present?*
* **GT:** *gastric*
* **Predicción:** *gastric mucosa with inflammation and possible malignant involvement*
* **Análisis:** El modelo acierta la palabra clave, pero añade contenido clínico especulativo.

### **Ejemplo 2** — Respuesta parcialmente correcta pero demasiado extensa

* **Pregunta:** *where is this?*
* **GT:** *urinary*
* **Predicción:** *urinary system and kidneys are present on the other side of body*
* **Análisis:** Coincide el concepto general, pero la deriva narrativa penaliza BLEU y flexible.

### **Ejemplo 3** — Error semántico por interpretación excesiva

* **Pregunta:** *what organ is this?*
* **GT:** *heart*
* **Predicción:** *cardiac region with pericardial involvement and pathological thickening*
* **Análisis:** Similar al patrón previo: el modelo acierta la región, pero inventa condiciones clínicas.

### **Ejemplo 4** — Confusión específica de estructuras

* **Pregunta:** *what region is shown?*
* **GT:** *retroperitoneal*
* **Predicción:** *abdominal cavity with surrounding lymphatic spread*
* **Análisis:** El modelo malinterpreta la subregión y añade elementos no presentes.

### **Ejemplo 5** — Inestabilidad en respuestas binarias

* **Pregunta:** *is this abnormal?*
* **GT:** *no*
* **Predicción:** *no evidence of abnormality but possible cystic structures*
* **Análisis:** Aunque acierta el primer token, genera especulación médica adicional.

---

## 5.3 Conclusiones del análisis de errores

1. **Ambos modelos tienden a generar texto adicional**, una característica típica de modelos generativos pequeños al enfrentar respuestas de una sola palabra.
2. **El modelo 768 comete menos errores narrativos**, siendo más conciso y estable.
3. **El modelo 1024 es más propenso a "alucinar" detalles clínicos**, lo cual afecta métricas como Keyword Accuracy y BLEU.
4. En preguntas binarias, ambos modelos suelen acertar el primer token, pero el 1024 introduce más ruido posteriormente.
5. Los errores muestran que el modelo entiende la región anatómica general, pero no siempre la subestructura específica.

Estas observaciones complementan los resultados cuantitativos de la Sección 4 y ayudan a explicar las diferencias en el desempeño global.
