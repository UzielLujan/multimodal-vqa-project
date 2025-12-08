import os
from huggingface_hub import hf_hub_download

def download_clip_bin():
    repo_id = "openai/clip-vit-large-patch14-336"
    output_dir = os.path.join("checkpoints", "clip-vit-large-patch14-336")

    print(f"Recuperando archivo original (.bin) para: {repo_id}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin", # El archivo que sí existe
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print("Archivo pytorch_model.bin descargado correctamente.")
    except Exception as e:
        print(f"Error descarga: {e}")

if __name__ == "__main__":
    download_clip_bin()
