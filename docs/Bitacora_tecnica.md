# **Bitácora Técnica del Proyecto**

### *multimodal-vqa-project*

---
Este documento funciona como **registro técnico completo** del proyecto, incluyendo:

* diseño del pipeline,
* decisiones arquitectónicas,
* experimentos realizados,
* resultados parciales,
* problemas encontrados,
* reflexiones,
* pasos siguientes.

Es un documento vivo: se actualizará conforme avancemos en el desarrollo.

---
---

## **Diseño del Pipeline Técnico**

> Aquí va la definición progresiva del pipeline, desde la teoría hasta la implementación.

### **2.1. Pipeline general del sistema VQA**

```
Imagen → SigLIP → Proyector MLP → LLaMA-3 → Respuesta
```

### **2.2. Componentes principales**

* **SigLIP**: encoder visual SOTA entrenado con contrastive learning.
* **Proyector multimodal (MLP)**: adapta embeddings visuales al espacio del LLM.
* **LLaMA-3**: modelo generativo para producir la respuesta.
* **LoRA**: técnica de fine-tuning eficiente aplicada sobre el LLM o el MLP.

### **2.3. Decisiones arquitectónicas**

> Espacio para detallar las decisiones específicas:

* Tamaño exacto del modelo LLaMA-3 seleccionado (8B, 70B, etc.).
* Configuración del SigLIP.
* Dónde aplicar LoRA.
* Parámetros de entrenamiento.

---

## **Carga y Procesamiento de Datos**

> Aquí documenta todo lo relacionado con PathVQA y cualquier preprocesamiento.

* Ubicación en HuggingFace.
* Formato de las imágenes.
* Tipos de preguntas.
* Separación entre Yes/No y preguntas abiertas.
* Transformaciones necesarias (si aplica).
* Estructura final del `Dataset` en PyTorch.

Puedes usar bloques así:

```
### Observaciones:
- ...

### Decisiones:
- ...
```

---

## **Experimentos**

Aquí se documentan los experimentos oficiales del proyecto, incluyendo parámetros exactos.

### **4.1. Experimento 1 — Baseline (opcional)**

Descripción, parámetros, resultados parciales.

### **4.2. Experimento 2 — Modelo Moderno (LLaVA 1.5)**

* Hiperparámetros
* LR, batch size
* LoRA-rank
* Número de steps
* Hardware usado (GPU del cluster)
* Tiempo de entrenamiento

### **4.3. Experimento 3 — Comparativa Final**

* Tabla por tipo de pregunta
* BLEU, CIDEr, Accuracy, F1
* Gráficas

Puedes documentar así:

```bash
# Resultados
- BLEU: ...
- CIDEr: ...
- Accuracy: ...
- F1: ...

# Interpretación
- ...
```

---

## **Problemas y Soluciones**

Usa este espacio para registrar errores técnicos, bugs o inconsistencias.

Formato sugerido:

```bash
# Problema:
Descripción del problema.

# Diagnóstico:
Qué crees que lo causa.

# Solución aplicada:
Detalles exactos o enlaces a commits.
```

---

## **Notas Importantes y Observaciones**

Cualquier detalle relevante que pueda servir en análisis o escritura del reporte final.

Ejemplos:

* Comportamientos extraños del modelo.
* Hipótesis sobre resultados.
* Posibles mejoras.
* Preguntas para discutir con el profesor.

---

## **Pendientes Generales del Proyecto**

Lista acumulativa de tareas no terminadas.

Formato sugerido:

```
- [ ] Implementar loader de PathVQA
- [ ] Generar primer prototipo con LLaVA
- [ ] Añadir métrica CIDEr
- [ ] Configurar LoRA
- [ ] Entrenar modelo final
- [ ] Redactar sección experimental del reporte
```

---

## **Cierre y Reflexiones Finales**

Este apartado se completa al finalizar el proyecto para insertar directo en el reporte final.

Puedes incluir:

* Qué aprendiste
* Qué funcionó mejor
* Limitaciones del proyecto
* Recomendaciones futuras

---

## **Fin de la Bitácora Técnica**
