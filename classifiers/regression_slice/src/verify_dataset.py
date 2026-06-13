import os
import torch
from dataset import get_regression_loaders

def verify():
    # Ruta de los datos
    DATA_PATH = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra la ruta {DATA_PATH}")
        return

    # Usamos la función del loader para obtener las listas ya divididas por sujetos
    print("Escaneando sujetos y calculando etiquetas...")
    loaders = get_regression_loaders(DATA_PATH, batch_size=1)
    
    train_ds = loaders['train'].dataset
    val_ds = loaders['val'].dataset
    
    if len(train_ds) == 0:
        print("No se encontraron imágenes en 'imagesTr'.")
        return

    print("\n" + "="*40)
    print("   VERIFICACIÓN DE ETIQUETAS (REGRESIÓN)")
    print("="*40)
    
    # Mostrar una pequeña muestra del set de entrenamiento
    print(f"\n--- Muestra de Entrenamiento ({len(train_ds)} imágenes) ---")
    for i in range(min(10, len(train_ds))):
        path, label = train_ds.samples[i]
        fname = os.path.basename(path)
        # La etiqueta es un float, la mostramos con 4 decimales
        print(f"File: {fname:25} | Label: {label:7.4f}")

    # Mostrar una pequeña muestra del set de validación
    print(f"\n--- Muestra de Validación ({len(val_ds)} imágenes) ---")
    for i in range(min(5, len(val_ds))):
        path, label = val_ds.samples[i]
        fname = os.path.basename(path)
        print(f"File: {fname:25} | Label: {label:7.4f}")

    # Estadísticas globales
    all_labels = [s[1] for s in train_ds.samples] + [s[1] for s in val_ds.samples]
    print("\n" + "="*40)
    print(f"Mínima etiqueta: {min(all_labels):.4f} (Debería ser -1.0)")
    print(f"Máxima etiqueta: {max(all_labels):.4f} (Debería ser 1.0)")
    print(f"Media etiquetas: {sum(all_labels)/len(all_labels):.4f} (Debería rondar 0.0)")
    print("="*40)

if __name__ == '__main__':
    verify()
