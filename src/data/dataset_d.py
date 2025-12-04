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
        self.tokenizer = processor.tokenizer
        self.split = split
        self.model_max_length = model_max_length
        self.tokenizer.padding_side = "right"

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
        question = item['question'].strip()
        answer = item['answer'].strip()

        # Conversación siguiendo plantilla oficial LLaVA/LLaMA-3
        user_content = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]

        if self.split == 'train':
            conversation = user_content + [
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer}
                ]}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            # Prompt sin respuesta para enmascarar labels
            user_only_prompt = self.tokenizer.apply_chat_template(
                user_content,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            conversation = user_content
            prompt_text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

        # 3. Procesamiento Multimodal
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length
        )

        input_ids = inputs.input_ids.squeeze(0)

        # 4. Labels con enmascarado de la parte del usuario
        if self.split == 'train':
            labels = input_ids.clone()
            prompt_only_ids = self.tokenizer(
                user_only_prompt,
                truncation=True,
                max_length=self.model_max_length,
                return_tensors="pt"
            ).input_ids.squeeze(0)

            pad_id = self.tokenizer.pad_token_id
            prompt_len = int((prompt_only_ids != pad_id).sum())
            labels[:prompt_len] = -100
        else:
            labels = torch.full_like(input_ids, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": labels
        }
