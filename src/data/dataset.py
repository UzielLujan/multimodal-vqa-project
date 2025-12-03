import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
import os

class PathVQADataset(Dataset):
    def __init__(self, data_path, processor, split='train', model_max_length=2048):
        """
        Args:
            data_path (str): Ruta a la carpeta donde guardaste el dataset (data/raw/path_vqa_hf).
            processor (LlavaProcessor): El procesador de Hugging Face (maneja img + texto).
            split (str): 'train', 'validation', o 'test'.
            model_max_length (int): Longitud máxima de tokens.
        """
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.model_max_length = model_max_length
        
        # 1. Cargar datos en modo OFFLINE (vital para tu clúster)
        print(f"Cargando split '{split}' desde disco: {data_path}...")
        self.dataset = load_from_disk(data_path)[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Obtener imagen y asegurar formato RGB
        image = item['image'].convert('RGB')
        
        # 2. Obtener pregunta y respuesta
        question = item['question']
        answer = item['answer']
        
        # 3. Construir el Prompt (Formato Conversación)
        # Adaptado para que LLaMA-3 entienda que es un turno de usuario/asistente
        # Nota: <image> es el token especial donde el modelo "verá" la imagen.
        text_input = f"<|user|>\n<image>\n{question}<|end_header_id|>\n<|assistant|>\n"
        
        # Si estamos entrenando, añadimos la respuesta para que aprenda
        if self.split == 'train':
            text_input += f"{answer}<|end_of_text|>"
            
        # 4. Procesar con el Processor de LLaVA/LLaMA
        # Esto convierte la imagen a tensores (SigLIP) y el texto a tokens (LLaMA-3)
        inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length
        )

        # 5. Limpieza de dimensiones extra (el processor añade batch dim=1 por defecto)
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        pixel_values = inputs.pixel_values.squeeze(0)
        
        # 6. Crear Labels (para entrenamiento)
        # En HuggingFace, labels = input_ids, pero ponemos -100 donde no queremos calcular loss
        labels = input_ids.clone()
        
        # Si NO estamos entrenando, no necesitamos labels (o devolvemos dummy)
        if self.split != 'train':
            labels = torch.full_like(input_ids, -100)
        else:
            # Opcional: Aquí podrías enmascarar la pregunta para que solo aprenda la respuesta.
            # Por ahora, dejamos que aprenda todo el flujo para simplificar (standard causal LM).
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }