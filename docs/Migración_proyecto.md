# Diseño del Proyecto: Migración a TinyLlama

## 1. Objetivo General

Migrar el flujo de trabajo de procesamiento de lenguaje natural (NLP) actual hacia **TinyLlama-1.1B**. El objetivo es establecer un modelo base que sea computacionalmente viable en la infraestructura actual sin sacrificar drásticamente la capacidad de razonamiento y generación de texto.

---

## 2. Justificación y Análisis de Antecedentes

La decisión de migrar hacia una arquitectura de 1.1 billones de parámetros surge de un análisis exhaustivo de las limitaciones de hardware y el desempeño de modelos previos.

### 2.1. Evaluación de Llama 3 (8B)

Inicialmente, se intentó implementar Llama 3 (versión 8B) debido a su estado del arte (SOTA). Sin embargo, se encontraron barreras insuperables con la infraestructura disponible:

- **Insuficiencia de Cómputo y Memoria:** Los recursos de GPU disponibles no fueron capaces de alojar los pesos del modelo y los estados del optimizador simultáneamente.
- **Errores Críticos:** Se reportaron consistentemente errores de Out of Memory (OOM) durante la fase de carga e inferencia.
- **Inestabilidad Numérica:** Debido a intentos de cuantización agresiva para ajustar el modelo a la VRAM, se observaron fallas de cálculo numérico, degradando la integridad de las salidas.

### 2.2. Evaluación de GPT-2

Se exploró la migración hacia la familia GPT-2 como una alternativa ligera y probada.

- **Ventaja:** Extremadamente ligero y compatible con hardware modesto.
- **Desventaja Crítica:** La arquitectura, aunque histórica, carece de las optimizaciones modernas. Las pruebas populares han mostrado un desempeño deficiente en tareas de razonamiento complejo y una perplejidad ($PPL$) demasiado alta para los estándares actuales del proyecto.

---

## 3. Solución Propuesta: TinyLlama-1.1B

TinyLlama se selecciona como el punto medio óptimo. Ofrece una arquitectura moderna, diseñada explícitamente para la eficiencia.

### 3.1. Especificaciones Técnicas

| Característica   | Detalle                                                                 |
|------------------|-------------------------------------------------------------------------|
| Parámetros       | $\approx 1.1 \times 10^9$ (1.1B)                                        |
| Contexto         | 2048 tokens (extensible con RoPE)                                       |
| Arquitectura     | Llama 2 (GQA, RoPE, SwiGLU)                                             |
| Entrenamiento    | 3 Trillones de tokens (Chinchilla optimal)                              |
| Atención         | Flash Attention 2.0 integrado                                           |

### 3.2. Ventajas Esperadas

- **Viabilidad de Memoria:** El modelo en float16 ocupa aproximadamente 2.2 GB de VRAM. Incluso con los estados del optimizador (ej. AdamW), cabe cómodamente en una GPU de gama media (ej. 16GB o incluso 8GB con optimizaciones).
- **Rendimiento:** A pesar de su tamaño, supera a GPT-2 y es comparable a Llama 2 (7B) en varias métricas académicas.
- **Flexibilidad de Entrenamiento:** Su ligereza permite elegir entre Full Fine-Tuning o PEFT, facilitando experimentos rápidos.

---

## 4. Plan de Implementación

### 4.1. Estrategia de Migración y Compatibilidad

El enfoque principal es maximizar la reutilización del sistema actual ya desplegado en el clúster. Se priorizará la adaptación de los scripts de entrenamiento e inferencia existentes para minimizar el tiempo de inactividad.

- **Preservación del Entorno:** Se intentará mantener las versiones actuales de librerías y drivers CUDA, realizando cambios solo si son estrictamente necesarios para soportar Flash Attention en TinyLlama.
- **Plan de Contingencia:** En caso de incompatibilidad crítica (ej. conflictos de dependencias profundas), se optará por soluciones de implementación profunda para evitar caer en bucles de parches, es decir, si es necesario, refactorizar todo el sistema base del clúster.

### 4.2. Arquitectura del Pipeline Multimodal

La estructura lógica del pipeline multimodal se intentará mantener intacta, sustituyendo únicamente el componente de generación de texto final. El flujo de datos será:

$$
\text{Imagen} \xrightarrow{\text{Codificador}} \textbf{SigLIP} \xrightarrow{\text{Adaptación}} \textbf{Proyector MLP} \xrightarrow{\text{Generación}} \textbf{TinyLlama-1.1B} \rightarrow \text{Respuesta}
$$

- **Encoder Visual:** Se mantiene SigLIP por su eficiencia y precisión en embeddings imagen-texto y por tenerlo listo en el clúster.
- **Conector:** El Proyector MLP seguirá encargándose de alinear las características visuales al espacio latente del LLM.  
  **Nota:** Será necesario re-entrenar o ajustar este proyector para que coincida con las dimensiones de entrada de TinyLlama (2048 dims vs 4096 de Llama-3).
- **LLM:** TinyLlama reemplaza a Llama-3 como el cerebro generativo.

### 4.3. Fases de Ejecución

**Fase 1: Ajuste del Entorno**
- Verificación de compatibilidad de bitsandbytes y accelerate con la arquitectura TinyLlama.
- Ajuste de hiperparámetros de memoria en los scripts de slurm/bash del clúster.

**Fase 2: Adaptación del Proyector**
- Modificación de la capa de salida del MLP para mapear al hidden_size de TinyLlama.
- Entrenamiento inicial (warm-up) del proyector congelando el LLM y el encoder visual.

**Fase 3: Fine-Tuning (Estrategia LoRA)**
- A pesar de que TinyLlama permite un ajuste completo (Full Fine-Tuning) debido a su tamaño reducido, se mantendrá la implementación de LoRA (Low-Rank Adaptation) tal como existe en el sistema actual.

  **Justificación:**
  - **Consistencia del Pipeline:** Permite reutilizar los scripts de entrenamiento existentes sin refactorización mayor.
  - **Eficiencia:** Reduce el uso de VRAM para los estados del optimizador, permitiendo aumentar el tamaño del batch (batch size) para una convergencia más estable.
  - **Modularidad:** Facilita la gestión de diferentes versiones del modelo sin duplicar los pesos base.

  **Acción:** Ajustar los `target_modules` en la configuración de PEFT para asegurar que se apliquen a las capas de query y value (y opcionalmente output) de la arquitectura TinyLlama.

**Fase 4: Validación**
- Validar el desempeño del pipeline completo como se detalla en el documento de Protocolo del Proyecto VQA, asegurando que las métricas de evaluación se mantengan o mejoren respecto a la versión previa con Llama-3.