# Bitácora de Ingeniería: Resolución de Arquitectura VQA Multimodal

---

## 1. El Problema Original

El objetivo inicial era construir un modelo VQA utilizando componentes SOTA (State of the Art). Sin embargo, el proyecto enfrentó bloqueos críticos de infraestructura y compatibilidad.

### Intentos Fallidos
- **Llama-3 (8B) + SigLIP + LoRA:**
  - *Fallo:* CUDA Out of Memory (OOM). Incluso con cuantización de 4-bit, los gradientes y estados del optimizador excedían la VRAM disponible en el clúster.
- **TinyLlama (1.1B) + SigLIP + LoRA:**
  - *Fallo:* Incompatibilidad de Arquitectura (`NoneType errors`). SigLIP no exponía correctamente los `hidden_states` necesarios para LLaVA debido a discrepancias en la API de transformers y configuraciones internas del procesador.
  - *Fallo Silencioso:* "Modelo Fantasma". La inyección de pesos duplicaba los parámetros (2.8B ~ 3B en lugar de 1.5B ~ 1.1B), sugiriendo que LoRA se aplicaba a una instancia no conectada del modelo.

---

## 2. Diagnóstico de Raíz

Tras depuración exhaustiva (Smoke Tests), se identificaron tres causas raíz que impedían el entrenamiento:

### A. El "Modelo Fantasma" (Duplicidad de Parámetros)
Al instanciar `LlavaForConditionalGeneration`, la clase crea internamente un modelo de lenguaje aleatorio. Nuestro script de carga inyectaba el TinyLlama pre-entrenado encima del existente pero sin reemplazar la referencia interna profunda.
- **Consecuencia:** El modelo tenía 2.8 Billones de parámetros (1.1B basura + 1.1B pre-entrenado + 0.6B visión).
- **Impacto en LoRA:** LoRA probablemente se adhería a las capas incorrectas (las "basura"), impidiendo el aprendizaje.

### B. Desalineación Matemática de Tokens (577 vs 576)
El codificador visual (CLIP/SigLIP) genera $N$ parches + 1 token CLS (Clasificación).
- **Cálculo:** $(336 / 14)^2 = 576$ parches.
- **Salida Real:** 577 vectores.
- **Error:** LLaVA esperaba estrictamente 576.
- **Solución:** Cambiar la estrategia `vision_feature_select_strategy` de "full" a "default", lo que recorta automáticamente el token CLS sobrante.

### C. Conflicto de Tipos de Datos (Float vs BFloat16)
La falta de GPU en el entorno local de desarrollo forzaba a PyTorch a usar Float32, mientras que partes del código forzaban BFloat16. Esto causaba errores de multiplicación de matrices (`mat1 and mat2 must have the same dtype`).

---

## 3. La Solución Definitiva (Arquitectura Actual)

Se optó por una simplificación radical para garantizar estabilidad y reproducibilidad.

### 3.1. Cambio de Componentes
- **Cerebro:** Se mantiene TinyLlama-1.1B (Eficiente y moderno).
- **Ojos:** Se migra de SigLIP a CLIP-ViT-Large-Patch14-336.
  - *Justificación:* CLIP es el estándar nativo de LLaVA 1.5, eliminando la necesidad de "wrappers" o parches manuales que causaban inestabilidad.

### 3.2. Estrategia de Entrenamiento: Full Fine-Tuning (FFT)
Se abandona LoRA en favor del ajuste completo (Full Fine-Tuning).
- **Justificación:**
  - El modelo total (~1.5B) es lo suficientemente pequeño para caber en la VRAM moderna (~16GB+) sin necesidad de adaptadores.
  - Elimina la complejidad de peft y los errores de inyección de capas.
  - Asegura que el modelo "aprenda" realmente a usar los nuevos embeddings visuales sin restricciones de rango bajo.

### 3.3. Corrección del "Factory" (`model_factory.py`)
Se reescribió el ensamblador del modelo con:
- **Inyección Quirúrgica:** Reemplazo explícito de `model.language_model` para eliminar el "Fantasma".
- **Sincronización de Vocabulario:** `resize_token_embeddings` ejecutado post-trasplante para evitar `IndexError`.
- **Agnosticismo de Hardware:** Detección automática de CUDA/CPU para permitir pruebas locales sin romper tipos de datos.

---

## 4. Estado Final

El pipeline ha pasado exitosamente las pruebas de integridad ("Smoke Tests"):
- **Parámetros Totales:** ~1.5 Billones (Correcto).
- **Forward Pass:** Exitoso (Loss calculado).
- **Backward Pass:** Exitoso (Gradientes fluyendo).
- **Entorno:** Configurado con PyTorch 2.4 + CUDA 12.1.

El sistema ha sido desplegado en el clúster y está siendo entrenado. Con las siguientes configuraciones:

```bash
training:
  # Configuración para GPU (Aprovechando tu nodo g-0-6)
  batch_size: 2             # TinyLlama + CLIP caben bien con batch 2 en ~16GB VRAM
  grad_accumulation: 4      # Batch efectivo = 8
  epochs: 3
  learning_rate: 2.0e-5     # FFT requiere learning rate más bajo que LoRA
```