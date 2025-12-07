# Informe Técnico: Implementación Exitosa del Pipeline VQA Multimodal

**Estado:** ✅ Entrenamiento Completado   
**Arquitectura:** TinyLlama-1.1B + CLIP-Large-336 + MLP Projector  
**Estrategia:** Full Fine-Tuning (FFT)

---

## 0. Objetivo Alcanzado

Se ha logrado desplegar y entrenar exitosamente un modelo Visual Question Answering (VQA) en el clúster del CIMAT, superando restricciones severas de compatibilidad y hardware.  
El modelo es capaz de procesar imágenes médicas y responder preguntas en lenguaje natural, fusionando la capacidad visual de **CLIP** con el razonamiento de **TinyLlama**.



## 1. El Problema Original

El objetivo inicial era construir un modelo VQA utilizando componentes modernos SOTA (State of the Art). Sin embargo, el proyecto enfrentó bloqueos críticos de infraestructura y compatibilidad.

### Intentos Fallidos
1. **Original. Llama-3 (8B) + SigLIP + LoRA:**
    - *Fallo:* CUDA Out of Memory (OOM). Incluso con cuantización de 4-bit, los gradientes y estados del optimizador excedían la VRAM disponible en el clúster aunque sí se realizaba el entrenamiento.
2. **Primera reducción. TinyLlama (1.1B) + SigLIP + LoRA:**
    - *Fallo:* Incompatibilidad de Arquitectura (`NoneType errors`). SigLIP no exponía correctamente los `hidden_states` necesarios para LLaVA debido a discrepancias en la API de transformers y configuraciones internas del procesador.
    - *Fallo Silencioso recurrente:* "Modelo Fantasma". La inyección de pesos duplicaba los parámetros (2.8B ~ 3B en lugar de 1.5B ~ 1.1B), sugiriendo que LoRA se aplicaba a una instancia no conectada del modelo lo que no permitía aplicar LoRA correctamente (hipótesis).

---

## 2. Diagnóstico de Raíz y Desafíos Críticos Superados

Durante la fase de implementación, tras depuración exhaustiva (Smoke Tests), se identificaron y resolvieron cuatro causas raíz que impedían el entrenamiento:

### A. El "Modelo Fantasma" (Duplicidad de Parámetros)
Al instanciar `LlavaForConditionalGeneration`, la clase crea internamente un modelo de lenguaje aleatorio. Nuestro script de carga inyectaba el TinyLlama pre-entrenado encima del existente pero sin reemplazar la referencia interna profunda.
- **Problema:** El modelo reportaba 2.8B parámetros, 1.1B basura + 1.1B pre-entrenado + 0.6B visión, el doble de lo esperado, indicando que se cargaban dos instancias del LLM en memoria. 
- **Impacto en LoRA (hipótesis):** LoRA probablemente se adhería a las capas incorrectas (las "basura"), impidiendo el aprendizaje y su correcta aplicación.
 
- **Solución:** Se implementó una "Cirugía de Inyección Profunda" en `model_factory.py`, reemplazando explícitamente el atributo `model.model.language_model` para eliminar la instancia residual.  
- **Resultado:** Reducción a 1.5B parámetros, liberando memoria y asegurando que los gradientes fluyan al modelo correcto.

### B. Desalineación Matemática de Tokens (577 vs 576)
El codificador visual (CLIP/SigLIP) genera $N$ parches + 1 token CLS (Clasificación).
- **Cálculo:** $(336 / 14)^2 = 576$ parches.
- **Salida Real:** 577 vectores.
- **Error:** LLaVA esperaba estrictamente 576.
- **Solución:** Cambiar la estrategia `vision_feature_select_strategy` de `full` a `default`, lo que recorta automáticamente el token CLS sobrante.

### C. Conflicto de Tipos de Datos (Float vs BFloat16)
La falta de GPU en el entorno local de desarrollo forzaba a PyTorch a usar Float32, mientras que partes del código forzaban BFloat16. Esto causaba errores de multiplicación de matrices.

- **Problema:** `RuntimeError: mat1 and mat2 must have the same dtype`. Discrepancia entre la CPU (Float32) y la GPU (BFloat16) al inicializar el proyector.  
- **Solución:** Implementación de "Casting Estricto" (`model.to(dtype=...)`) al final del ensamblaje del modelo y detección automática de hardware en los scripts de prueba.


### D. Seguridad y Formatos (Safetensors vs Bin)**  
- **Problema:** Bloqueo de seguridad CVE-2025-32434 en PyTorch al cargar pesos .bin antiguos de OpenAI.  
- **Solución:**  
  - Actualización del entorno a PyTorch 2.4 + CUDA 12.1.  
  - Conversión manual de los pesos de CLIP a formato Safetensors (script `convert_to_safetensors.py`).








---

## 3. La Solución Definitiva (Arquitectura Actual)

Se optó por una simplificación radical para garantizar estabilidad y reproducibilidad.

### 3.1. Cambio de Componentes
- **Cerebro:** Se mantiene TinyLlama-1.1B (Eficiente y moderno).
- **Ojos:** Se migra de SigLIP a CLIP-ViT-Large-Patch14-336.
  - *Justificación:* CLIP es el estándar nativo de LLaVA 1.5, eliminando la necesidad de "wrappers" o parches manuales que causaban inestabilidad.
  - *Dificultades presentadas en la implementación de CLIP:* 
    - Aunque se consigió descargar el modelo CLIP desde HuggingFace no teniamos acceso directo a los pesos en el formato safetensors solo bin lo que requirió un proceso manual de conversión, el flujo fue:
      - Descargar pesos binarios `dowload_clip.py`.
      - Convertir a safetensors usando scripts de conversión `convert_to_safetensors.py`.
      - Cargar los pesos convertidos en el modelo CLIP dentro del pipeline del modelo `model_factory.py`.


### 3.2. Arquitectura Final del Sistema

El sistema implementado difiere significativamente del diseño original (Llama-3 + SigLIP + LoRA) debido a optimizaciones necesarias:

| Componente      | Selección Final                | Justificación                                                                 |
|-----------------|-------------------------------|-------------------------------------------------------------------------------|
| **LLM (Cerebro)**      | TinyLlama-1.1B                  | Equilibrio óptimo entre capacidad de razonamiento y consumo de VRAM.           |
| **Vision Encoder**     | CLIP-ViT-Large-Patch14-336      | Estándar nativo de LLaVA 1.5. Mayor estabilidad y compatibilidad que SigLIP.   |
| **Entrenamiento**      | Full Fine-Tuning (FFT)          | Permite un ajuste profundo de los pesos sin la inestabilidad de cuantización.  |
| **Optimizador**        | AdamW (Nativo)                  | Elimina dependencias externas propensas a fallos en el clúster.                |


### 3.2. Estrategia de Entrenamiento: Full Fine-Tuning (FFT)
Se abandona LoRA en favor del ajuste completo (Full Fine-Tuning).
- **Justificación:**
  - El modelo total (~1.5B) es lo suficientemente pequeño para caber en la VRAM moderna (~16GB+) sin necesidad de adaptadores.
  - Elimina la complejidad de peft y los errores de inyección de capas.
  - Asegura que el modelo "aprenda" realmente a usar los nuevos embeddings visuales sin restricciones de rango bajo.
- **Posible experimentación:** Ahora que el sistema es estable (eliminamos 'Modelo Fantasma'), se puede experimentar con LoRA para optimizar recursos y aplicar técnicas que mejoresn el rendimiento.

### 3.3. Corrección del "Factory" (`model_factory.py`)
Se reescribió el ensamblador del modelo con:
- **Inyección Quirúrgica:** Reemplazo explícito de `model.language_model` para eliminar el "Fantasma".
- **Sincronización de Vocabulario:** `resize_token_embeddings` ejecutado post-trasplante para evitar `IndexError`.
- **Agnosticismo de Hardware:** Detección automática de CUDA/CPU para permitir pruebas locales sin romper tipos de datos.

---

## 4. Estado actual del Proyecto

El pipeline ha pasado exitosamente las pruebas de integridad ("Smoke Tests") en local y en el clúster.:
- **Parámetros Totales:** ~1.5 Billones (Correcto).
- **Forward Pass:** Exitoso (Loss calculado).
- **Backward Pass:** Exitoso (Gradientes fluyendo).
- **Entorno:** Configurado con PyTorch 2.4 + CUDA 12.1.

El sistema ha sido desplegado en el clúster y está siendo entrenado. Con las siguientes configuraciones:

```bash
training:
  batch_size: 2    # 2 para evitar OOM
  grad_accumulation: 4    # Batch efectivo = 8
  epochs: 3
  learning_rate: 2.0e-5   # FFT requiere learning rate más bajo que LoRA
  # Optimizaciones
  fp16: false    # Usar bf16 si la GPU es (A100/A10/3090), si no fp16: true
  bf16: true     # Recomendado para estabilidad numérica en entrenamiento
  max_length: 768 # Longitud máxima de secuencia (ajustar según necesidad)          
  gradient_checkpointing: false # True para ahorrar VRAM en FFT pero se detectó un problema de estabilidad, se dejó en false
```

---

### 4.1 Estado Actual del Entrenamiento

- **Trabajo:** En ejecución estable en el nodo g-0-6 (GPU).
- **Métrica Inicial (Loss):** 52.60 (Step 10, inicio real del entrenamiento).
- **Progreso:** El loss disminuye consistentemente:
    - Step 100: 41.47
    - Step 200: 27.31
    - Step 300: 0.23
    - Step 1220: 0.13
    - **Eval Loss final (Epoch 1):** 0.0405
- **Grad Norm:** De 416.0 (inicio) a valores bajos (<2) al final.
- **Learning Rate:** Incrementa hasta ~2e-5 y luego desciende gradualmente.
- **Velocidad:** ~12 s/it (estimado, sin cambios reportados).
- **Checkpointing:** Guardado automático cada epoch.
- **Mejoras implementadas:** 
  - En la tercera prueba se corrigió un bug en el cálculo del eval loss que causaba que no se mostraran los valores correctamente. 
  - Se ajustó el parámetro de `gradient_checkpointing` a false para evitar inestabilidades detectadas.
  - Se optimizó el `batch_size = 2`, `grad_accumulation = 4` y `max_length = 768` para maximizar el uso de VRAM sin causar OOM.
  - Se implementó ajuste de rutas en `train_config.yaml` para diferenciar entre pruebas y evitar sobreescrituras accidentales de checkpoints.
  - Se implemento regularizacion `weight_decay` y  `warmup_ratio` para mejorar la estabilidad del entrenamiento y evitar sobreajuste.

**Próximos Pasos Post-Entrenamiento:**
- Generación de inferencias de prueba con `inference_demo.py` para verificación rápida de calidad de respuestas a imágenes y `inference_eval.py` para evaluación cuantitativa con las métricas establecidas.
- Cálculo de métricas BLEU/ROUGE para el reporte final.
- **Observación:** Aparentemente, el modelo está aprendiendo correctamente, con una reducción significativa del loss y gradientes estables. Hay que verificar que el valor tan bajo de eval loss (0.0405) no sea indicativo de sobreajuste o fuga de datos.

---

## 5. **Conclusión:**  
La infraestructura es sólida, el código es robusto y el modelo está aprendiendo. A partir de aquí, se pueden explorar optimizaciones adicionales como LoRA o técnicas de regularización para mejorar el rendimiento sin comprometer la estabilidad alcanzada.

