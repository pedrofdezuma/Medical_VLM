import os
import json

def split_metadata():
    # Rutas definidas por el usuario
    metadata_path = r"D:\axial_flair_metadata.jsonl"
    train_dir = r"D:\data-axial-flair\train"
    val_dir = r"D:\data-axial-flair\val"
    output_dir = r"D:\data-axial-flair"

    # Obtener listas de archivos en los directorios
    print(f"Obteniendo nombres de archivos de {train_dir} y {val_dir}...")
    train_filenames = set(os.listdir(train_dir))
    val_filenames = set(os.listdir(val_dir))

    train_count = 0
    val_count = 0
    ignored_count = 0

    print("Procesando metadatos...")
    
    # Abrir archivos de salida
    train_jsonl_path = os.path.join(output_dir, "train.jsonl")
    val_jsonl_path = os.path.join(output_dir, "val.jsonl")

    with open(metadata_path, 'r', encoding='utf-8') as f_in, \
         open(train_jsonl_path, 'w', encoding='utf-8') as f_train, \
         open(val_jsonl_path, 'w', encoding='utf-8') as f_val:
        
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                file_name = data.get("file_name").lower().strip()
                
                if file_name in train_filenames:
                    f_train.write(json.dumps(data) + "\n")
                    train_count += 1
                elif file_name in val_filenames:
                    f_val.write(json.dumps(data) + "\n")
                    val_count += 1
                else:
                    ignored_count += 1
            except json.JSONDecodeError:
                print(f"Error al decodificar linea: {line[:100]}...")

    print(f"Completado.")
    print(f"- Entradas de Train: {train_count}")
    print(f"- Entradas de Val: {val_count}")
    print(f"- Entradas no encontradas en directorios: {ignored_count}")
    print(f"Archivos guardados en: {output_dir}")

if __name__ == "__main__":
    split_metadata()
