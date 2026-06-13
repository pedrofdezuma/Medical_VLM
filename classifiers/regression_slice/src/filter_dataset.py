import os
import re
import pickle
import csv
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# --- CONFIGURACIÓN ---
SOURCE_DIR = r"D:\axial-flair-dataset"
OUTPUT_PICKLE = r"D:\axial_flair_organized.pkl"
OUTPUT_CSV = r"D:\axial_flair_report.csv"
# ---------------------

abbreviations = {
    "t1-w": {"t1", "t1n", "t1w"},
    "t1-ce": {"t1c", "t1ce", "t1gd"},
    "t2": {"t2w", "t2"},
    "flair": {"t2f", "flair", "t2flair"}
}

def get_patient_id(filename):
    """
    Extracts a clean Patient ID from MedTrinity-25M filenames to prevent data leakage.
    Returns the core ID without study, shard, or slice suffixes.
    """
    filename = filename.strip()

    # Case 1: Standard MedTrinity/BraTS with double dashes "--"
    if "--" in filename:
        parts = filename.split("--")
        if len(parts) >= 3:
            raw_id = parts[2]
            # Remove trailing numeric suffixes only if they have two groups (e.g., _0001_35566)
            # This preserves single ID numbers (e.g., brats2021_01012)
            clean_id = re.sub(r'_\d{4,}(_\d+)$', '', raw_id)
            return clean_id

    # Case 2: BraTS-GoAT or other formats with simple dashes
    # Capture brats-XXX-YYYYY (3 parts) to avoid including study/slice suffixes
    match_brats = re.search(r"(brats-[a-z0-9]+-[a-z0-9]+)", filename, re.IGNORECASE)
    if match_brats:
        return match_brats.group(1)

    # Case 3: ISPY1 format
    match_ispy = re.search(r"(ispy1_\d+)_", filename, re.IGNORECASE)
    if match_ispy:
        return match_ispy.group(1)

    return None

def get_slice_number(filename):
    """
    Extracts the slice number from specific MRI filename formats.
    """
    match_slice = re.search(r'slice_(\d+)', filename, re.IGNORECASE)
    if match_slice:
        return int(match_slice.group(1))

    match_xyz = re.search(r'[xyz]_(\d+)', filename, re.IGNORECASE)
    if match_xyz:
        return int(match_xyz.group(1))

    return None

def get_view_letter(filename):
    fn_lower = filename.lower()
    match_med = re.search(r"--([xyz])_\d+--", fn_lower)
    if match_med:
        return match_med.group(1)
    match_brats = re.search(r"_([xyz])(?:\.|$|_)", fn_lower)
    if match_brats:
        return match_brats.group(1)
    return None

def get_sequence(filename: str) -> str:
    fn_lower = filename.lower()
    match_mr = re.search(r'^mr_([a-z0-9]+)--', fn_lower)
    if match_mr:
        return match_mr.group(1)
    match_brats = re.search(r'^[a-z0-9-]+_([a-z0-9]+)_', fn_lower)
    if match_brats:
        return match_brats.group(1)
    return None

def is_axial(filename: str, view_letter: str) -> bool:
    if not view_letter:
        return False
    # Formato A (MedTrinity): 'x' es axial
    if "--" in filename:
        return view_letter.lower() == 'x'
    # Formato B (BraTS): 'z' es axial
    return view_letter.lower() == 'z'

def is_flair(sequence: str) -> bool:
    if not sequence:
        return False
    return sequence.lower() in abbreviations["flair"]

def filter_and_organize():
    print(f"\n>>> Iniciando escaneo y organización en: {SOURCE_DIR}")
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: El directorio {SOURCE_DIR} no existe.")
        return

    # Estructura: patient_id -> list of slices {path, slice_num, original_filename}
    organized_data = defaultdict(list)
    total_processed = 0
    total_matched = 0

    shards = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    for shard in shards:
        shard_path = os.path.join(SOURCE_DIR, shard)
        print(f"Procesando shard: {shard}")
        
        for root, _, files in os.walk(shard_path):
            for file in tqdm(files, desc=f"  Analizando {os.path.basename(root)}", leave=False):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                total_processed += 1
                seq = get_sequence(file)
                view = get_view_letter(file)

                if is_flair(seq) and is_axial(file, view):
                    patient_id = get_patient_id(file)
                    slice_num = get_slice_number(file)
                    
                    if patient_id:
                        total_matched += 1
                        info = {
                            "path": os.path.abspath(os.path.join(root, file)),
                            "slice": slice_num,
                            "filename": file
                        }
                        organized_data[patient_id].append(info)

    # Guardar resultados
    print(f"\n>>> Filtrado completado.")
    print(f"Total imágenes analizadas: {total_processed}")
    print(f"Total imágenes Axial-Flair: {total_matched}")
    print(f"Total pacientes detectados: {len(organized_data)}")

    # 1. Guardar Pickle (Desactivado por petición del usuario)
    # try:
    #     with open(OUTPUT_PICKLE, 'wb') as f:
    #         pickle.dump(dict(organized_data), f)
    #     print(f"[OK] Organización completa guardada en pickle: {OUTPUT_PICKLE}")
    # except Exception as e:
    #     print(f"[ERROR] No se pudo guardar el pickle: {e}")

    # 2. Guardar CSV (Reporte para análisis humano)
    try:
        # Ordenamos pacientes por número de imágenes (opcional, de más a menos)
        sorted_patients = sorted(organized_data.items(), key=lambda x: len(x[1]), reverse=True)
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Patient_ID', 'Axial_Flair_Count'])
            for p_id, slices in sorted_patients:
                writer.writerow([p_id, len(slices)])
        print(f"[OK] Reporte CSV generado en: {OUTPUT_CSV}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el CSV: {e}")

    # Mostrar muestra por consola
    if organized_data:
        print("\nResumen de los 5 pacientes con más imágenes:")
        for p_id, slices in sorted_patients[:5]:
            print(f"  - {p_id}: {len(slices)} imágenes")

if __name__ == "__main__":
    filter_and_organize()
