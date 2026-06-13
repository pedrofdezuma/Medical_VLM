import os
import pickle
from dataset import get_regression_loaders

# --- CONFIGURACIÓN ---
# Ajusta esta ruta a donde tengas los datos localmente
DATA_DIR = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "manual_split_export"))

def export_subjects():
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No se encontró el directorio de datos en: {DATA_DIR}")
        print("Por favor, edita la variable DATA_DIR en este script con la ruta correcta.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Cargando loaders desde: {DATA_DIR}...")
    # Usamos batch_size=1 y num_workers=0 para que sea rápido y ligero
    loaders = get_regression_loaders(DATA_DIR, batch_size=1, num_workers=0)
    
    subjects_path = os.path.join(OUTPUT_DIR, 'subjects_split.pkl')
    
    data_to_save = {
        'train_subjects': loaders['train_subjects'],
        'val_subjects': loaders['val_subjects']
    }
    
    with open(subjects_path, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print("-" * 30)
    print(f"¡Éxito! Listas de sujetos exportadas.")
    print(f"Archivo: {subjects_path}")
    print(f"Entrenamiento: {len(data_to_save['train_subjects'])} sujetos")
    print(f"Validación:    {len(data_to_save['val_subjects'])} sujetos")
    print("-" * 30)

if __name__ == "__main__":
    export_subjects()
