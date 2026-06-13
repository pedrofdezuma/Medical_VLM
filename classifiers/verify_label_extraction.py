import os
import re

# --- CONFIGURACIÓN DE RUTAS ---
REGRESSION_DATA_DIR = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"

def get_view_label(path):
    path_lower = path.lower()
    classes = ['axial', 'coronal', 'sagittal', 'non_brain_mri']
    if 'axial' in path_lower: return classes[0]
    if 'coronal' in path_lower: return classes[1]
    if 'sagittal' in path_lower: return classes[2]
    if 'non_brain' in path_lower or 'non_mri' in path_lower: return classes[3]
    return "UNKNOWN"

def get_sequence_label(path):
    path_lower = path.lower()
    classes = ['t1-w', 't1-ce', 't2-w', 'flair', 'others']
    # Mapeo flexible para nombres de carpetas como T1w, T1-ce, FLAIR, etc.
    if 't1-w' in path_lower or 't1w' in path_lower: return classes[0]
    if 't1-ce' in path_lower or 't1ce' in path_lower: return classes[1]
    if 't2-w' in path_lower or 't2w' in path_lower or 't2' in path_lower: return classes[2]
    if 'flair' in path_lower: return classes[3]
    if 'others' in path_lower: return classes[4]
    return "UNKNOWN"

def verify_logic():
    if not os.path.exists(REGRESSION_DATA_DIR):
        print(f"ERROR: No se encuentra la ruta base: {REGRESSION_DATA_DIR}")
        return

    print(f"Buscando muestras en diversas carpetas de: {REGRESSION_DATA_DIR}...")
    print(f"{'VISTA':<12} | {'SECUENCIA':<10} | {'RUTA (últimos 80 chars)':<80}")
    print("-" * 110)

    total_shown = 0
    max_per_folder = 3 # Mostramos 3 de cada carpeta para ver variedad

    for root, dirs, fnames in os.walk(REGRESSION_DATA_DIR):
        # Filtramos archivos de imagen
        imgs = [f for f in fnames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Solo procesamos si hay imágenes y estamos en una ruta de split
        if imgs and ('imagesTr' in root or 'imagesTs' in root):
            folder_count = 0
            for fname in imgs:
                full_path = os.path.abspath(os.path.join(root, fname))
                
                view = get_view_label(full_path)
                seq = get_sequence_label(full_path)
                
                display_path = ("..." + full_path[-77:]) if len(full_path) > 80 else full_path
                print(f"{view:<12} | {seq:<10} | {display_path:<80}")
                
                folder_count += 1
                total_shown += 1
                
                if folder_count >= max_per_folder:
                    break # Pasamos a la siguiente carpeta
            
            if total_shown >= 50: # Límite total de la tabla
                break

    if total_shown == 0:
        print("AVISO: No se encontraron imágenes en el formato esperado.")

if __name__ == "__main__":
    verify_logic()
