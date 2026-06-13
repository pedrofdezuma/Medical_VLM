import torch
from torchvision import models
import os

# --- CONFIGURACIÓN ---
# Carpeta donde se guardarán los pesos localmente en Windows
SAVE_DIR = r"D:\classifiers\model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_and_save_weights():
    """
    Descarga los pesos preentrenados oficiales de PyTorch y los guarda en archivos locales
    para poder subirlos a Picasso (donde no hay internet).
    """
    models_to_download = {
        "resnet50": models.resnet50(weights="ResNet50_Weights.DEFAULT"),
        "efficientnet_v2_s": models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.DEFAULT")
    }

    print("="*50)
    print("  DESCARGA DE PESOS PARA ENTORNOS OFFLINE (PICASSO)")
    print("="*50)

    for name, model in models_to_download.items():
        save_path = os.path.join(SAVE_DIR, f"{name}-weights.pth")
        print(f"\nDescargando y guardando pesos para: {name}...")
        
        try:
            # Guardamos solo el state_dict para máxima compatibilidad
            torch.save(model.state_dict(), save_path)
            print(f" [OK] Guardado en: {save_path}")
            print(f"      Tamaño: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f" [ERROR] No se pudo guardar {name}: {e}")

    print("\n" + "="*50)
    print("  SIGUIENTES PASOS")
    print("="*50)
    print(f"1. Localiza los archivos .pth en: {SAVE_DIR}")
    print("2. Súbelos a Picasso a la siguiente ruta:")
    print("   /mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/classifiers/model_weights/")
    print("3. Cambia MODEL_NAME en main.py y lanza el entrenamiento.")
    print("="*50)

if __name__ == "__main__":
    download_and_save_weights()
