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
        self.data_path = str(data_path) 
        self.processor = processor
        self.split = split
        self.model_max_length = model_max_length
        
        print(f"📂 [Dataset] Cargando split '{split}' desde: {self.data_path}")
        try:
            full_dataset = load_from_disk(self.data_path)
            self.dataset = full_dataset[split]
        except Exception as e:
            raise RuntimeError(f"Error cargando dataset. Verifica la ruta: {self.data_path}") from e
            
        # Token especial para separar turnos (TinyLlama Chat format)
        self.eos_token = processor.tokenizer.eos_token # </s>

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Imagen (PIL)
        image = item['image'].convert('RGB')
        
        # 2. Textos
        question = item['question']
        answer = str(item['answer']) # Asegurar string
        
        # 3. Construcción del Prompt (TinyLlama Chat Format)
        # Separamos el input (lo que ve el modelo) del target (lo que debe predecir)
        
        system_msg = "You are an expert pathologist. Answer the question based on the image provided."
        
        # Parte A: Contexto (System + User + Image) -> NO se entrena (Loss = 0)
        prompt_context = f"<|system|>\n{system_msg}</s>\n<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"
        
        # Parte B: Respuesta -> SÍ se entrena
        full_text = prompt_context + answer + self.eos_token

        # 4. Tokenización Inteligente
        # Tokenizamos el contexto solo para saber su longitud y enmascararlo
        context_tokens = self.processor.tokenizer(
            prompt_context, 
            add_special_tokens=False, 
            return_tensors='pt'
        ).input_ids.squeeze(0)
        
        # Procesamos todo junto (Imagen + Texto Completo)
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        pixel_values = inputs.pixel_values.squeeze(0)
        
        # 5. Creación de Labels (Masking Estratégico)
        labels = input_ids.clone()
        
        # A) En validación/test, no queremos labels (el modelo genera libremente)
        if self.split != 'train':
            labels = torch.full_like(input_ids, -100)
        else:
            # B) En train, enmascaramos el contexto (preguntas) con -100
            # Solo calculamos loss sobre la respuesta del asistente.
            context_len = len(context_tokens)
            
            # Nota: Si hubo truncamiento, context_len podría ser mayor que input_ids,
            # así que usamos min() para evitar errores de índice.
            mask_len = min(context_len, self.model_max_length)
            labels[:mask_len] = -100
            
            # También enmascaramos el padding
            labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels
        }