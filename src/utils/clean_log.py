import json

def clean_jsonl(input_path, output_path):
    seen_steps = {}
    with open(input_path, "r") as fin:
        for line in fin:
            try:
                entry = json.loads(line)
                step = entry.get("step")
                loss = entry.get("loss")
                # Solo guardar si tiene step y loss
                if step is not None and loss is not None:
                    # Si hay duplicados, guarda el último (puedes cambiar a 'if step not in seen_steps' para el primero)
                    seen_steps[step] = entry
            except Exception:
                continue

    # Guardar limpio
    with open(output_path, "w") as fout:
        for step in sorted(seen_steps):
            fout.write(json.dumps(seen_steps[step]) + "\n")

if __name__ == "__main__":
    clean_jsonl(
        "results/tinyllama-clip-1024/training_logs.jsonl",
        "results/tinyllama-clip-1024/training_logs_clean.jsonl"
    )