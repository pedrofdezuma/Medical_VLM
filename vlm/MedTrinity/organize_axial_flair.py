import os
import shutil
import random
import re
from pathlib import Path

def get_patient_id(filename):
    """
    Extracts a clean Patient ID from MedTrinity-25M filenames to prevent data leakage.
    Returns the core ID without study, shard, or slice suffixes.
    """
    filename = filename.strip().lower()
    # Case 1: Standard MedTrinity/BraTS with double dashes "--"
    if "--" in filename:
        parts = filename.split("--")
        if len(parts) >= 3:
            raw_id = parts[2]
            # Remove trailing numeric patterns like '_0001_35566'
            clean_id = re.sub(r'_\d{4,}(_\d+)$', '', raw_id)
            return clean_id
    # Case 2: BraTS-GoAT or other formats with simple dashes
    match_brats = re.search(r"(brats-[a-z]+-\d+)", filename)
    if match_brats:
        return match_brats.group(1)
    # Case 3: ISPY1 format
    match_ispy = re.search(r"(ispy1_\d+)_", filename)
    if match_ispy:
        return match_ispy.group(1)
    return None

def organize_dataset(src_dir, dest_root, train_ratio=0.8):
    """
    Organiza las imágenes en train y val basándose en el volumen de archivos.
    Garantiza que todos los archivos de un paciente estén en el mismo split.
    Los archivos sin ID (None) van siempre a Train.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.nii', '.nii.gz')
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_root, split), exist_ok=True)

    patient_groups = {}
    unidentified_files = []
    total_files_found = 0
    
    print(f"Escaneando archivos en {src_dir}...")
    
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(root, f)
                pid = get_patient_id(f)
                total_files_found += 1
                if pid:
                    if pid not in patient_groups:
                        patient_groups[pid] = []
                    patient_groups[pid].append(full_path)
                else:
                    unidentified_files.append(full_path)

    if total_files_found == 0:
        print("Error: No se encontraron archivos de imagen.")
        return

    # Calculamos el objetivo de archivos para Train (80% del total)
    target_train_count = int(total_files_found * train_ratio)
    
    # Todos los unidentified van a Train por defecto
    train_files_list = list(unidentified_files)
    val_files_list = []
    
    current_train_count = len(train_files_list)
    
    # Mezclamos los pacientes para una distribución aleatoria
    pids = list(patient_groups.keys())
    random.seed(42)
    random.shuffle(pids)

    # Asignamos pacientes a Train hasta alcanzar el 80% del volumen total de archivos
    train_pids_count = 0
    val_pids_count = 0
    
    for pid in pids:
        pid_files = patient_groups[pid]
        # Si aún no hemos llegado al cupo del 80%, añadimos el paciente a Train
        if current_train_count < target_train_count:
            train_files_list.extend(pid_files)
            current_train_count += len(pid_files)
            train_pids_count += 1
        else:
            val_files_list.extend(pid_files)
            val_pids_count += 1

    print(f"\nResumen de la organización por VOLUMEN DE ARCHIVOS:")
    print(f"Total archivos detectados: {total_files_found}")
    print(f"Objetivo Train (~{train_ratio*100}%): {target_train_count} archivos")
    print("-" * 30)
    print(f"Archivos asignados a TRAIN: {len(train_files_list)} ({len(train_files_list)/total_files_found:.1%})")
    print(f"   - Pacientes en Train: {train_pids_count}")
    print(f"   - Archivos sin ID: {len(unidentified_files)}")
    print(f"Archivos asignados a VAL: {len(val_files_list)} ({len(val_files_list)/total_files_found:.1%})")
    print(f"   - Pacientes en Val: {val_pids_count}")

    def perform_copy(file_list, split_name):
        count = 0
        for src_path in file_list:
            dest_path = os.path.join(dest_root, split_name, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dest_path)
                count += 1
            except Exception as e:
                print(f"Error copiando {src_path}: {e}")
        return count

    print(f"\nCopiando archivos a {dest_root}/train...")
    t_actual = perform_copy(train_files_list, 'train')
    
    print(f"Copiando archivos a {dest_root}/val...")
    v_actual = perform_copy(val_files_list, 'val')

    print(f"\nProceso finalizado con éxito.")
    print(f"Carpeta de destino: {os.path.abspath(dest_root)}")

if __name__ == "__main__":
    SOURCE = r"D:\axial-flair-dataset"
    DESTINATION = r"D:\data-axial-flair"
    
    if os.path.exists(SOURCE):
        organize_dataset(SOURCE, DESTINATION)
    else:
        print(f"Error: La carpeta de origen {SOURCE} no existe.")
