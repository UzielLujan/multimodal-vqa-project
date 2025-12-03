import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from pathlib import Path
from typing import Optional, Union

class PathVQADataset(Dataset):
    def __init__(
        self, 
        data_path: Union[str, Path], 
        processor, 
        split: str = 'train', 
        model_max_length: int = 2048
    ):
        # Aseguramos que data_path sea string para load_from_disk (por compatibilidad)
        self.data_path = str(data_path) 
        self.processor = processor
        self.split = split
        self.model_max_length = model_max_length
        
        print(f"📂 [Dataset] Cargando split '{split}' desde: {self.data_path}")
        # Carga offline robusta
        try:
            full_dataset = load_from_disk(self.data_path)
            self.dataset = full_dataset[split]
        except Exception as e:
            raise RuntimeError(f"Error cargando dataset desde {self.data_path}. ¿Está corrupto o la ruta está mal?") from e
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Imagen (Convertir a RGB siempre por seguridad)
        image = item['image'].convert('RGB')
        
        # 2. Prompt Template (Estilo LLaVA 1.5 / LLaMA 3)
        question = item['question']
        answer = item['answer']
        
        # Estructura de conversación estándar
        text_input = f"<|user|>\n<image>\n{question}<|end_header_id|>\n<|assistant|>\n"
        
        if self.split == 'train':
            text_input += f"{answer}<|end_of_text|>"

        # 3. Procesamiento Multimodal
        inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length
        )

        input_ids = inputs.input_ids.squeeze(0)
        
        # 4. Labels
        labels = input_ids.clone()
        if self.split != 'train':
            labels = torch.full_like(input_ids, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": labels
        }