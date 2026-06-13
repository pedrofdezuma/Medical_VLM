import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import pickle

from dataset import get_regression_loaders
from model import get_regression_model
from train import train_model

# --- CONFIGURACIÓN Y HIPERPARÁMETROS ---
PICASSO = True  # Cambiar a False para entorno Local (Windows)

# Modelos disponibles: "resnet18", "resnet50", "efficientnet_v2_s", "vgg16", "mobilenet_v2"
MODEL_NAME = "resnet18"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
UNFREEZE_EPOCH = 9
PATIENCE = 5
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GESTIÓN DE RUTAS DINÁMICAS ---
if PICASSO:
    # Ruta en el supercomputador Picasso
    DATA_DIR = r"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/2D_MRI_ms_ep_control_FLAIR_T1"
    # Ruta base para checkpoints en Picasso
    BASE_CHECKPOINT_PATH = r"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/classifiers/regression_slice/checkpoints"
else:
    # Ruta local en Windows
    DATA_DIR = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
    # Ruta base para checkpoints local (un nivel arriba de src)
    BASE_CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))

def main():
    # 1. Cargar Datos primero para saber el tamaño del dataset
    print(f"Cargando loaders desde: {DATA_DIR}")
    loaders = get_regression_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Calcular tamaño de entrenamiento en 'k' (miles)
    num_train_imgs = len(loaders['train'].dataset)
    train_k = f"{num_train_imgs // 1000}k"
    
    # 2. Definir rutas dinámicas de checkpoint
    # Ejemplo: resnet18-98k
    checkpoint_name = f"{MODEL_NAME}-{train_k}"
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Guardar lista de sujetos split (Train/Val)
    subjects_path = os.path.join(checkpoint_dir, 'subjects_split.pkl')
    with open(subjects_path, 'wb') as f:
        pickle.dump({
            'train_subjects': loaders['train_subjects'],
            'val_subjects': loaders['val_subjects']
        }, f)
    print(f"Lista de sujetos guardada en: {subjects_path}")
    
    model_path = os.path.join(checkpoint_dir, 'best_mri_regression.pth')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pth') 
    history_path = os.path.join(checkpoint_dir, 'history.pth')
    times_path = os.path.join(checkpoint_dir, 'times_epochs.pth')

    print(f"Resultados se guardarán en: {checkpoint_dir}")

    # 3. Inicializar Modelo
    model = get_regression_model(MODEL_NAME, pretrained=True, picasso=PICASSO)
    
    # Congelado inicial (entrenar solo la cabeza al principio)
    for param in model.parameters():
        param.requires_grad = False
    
    # Seleccionar la capa final correcta según la arquitectura
    if hasattr(model, 'fc'):
        head = model.fc
    elif hasattr(model, 'classifier'):
        head = model.classifier
    else:
        raise AttributeError(f"No se reconoce la capa final para {MODEL_NAME}")

    for param in head.parameters():
        param.requires_grad = True
        
    model = model.to(DEVICE)

    # 4. Configurar Optimización
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # --- LÓGICA DE REANUDACIÓN (Resume) ---
    old_history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'lr': []}
    old_times = {}

    if os.path.exists(model_path):
        print(f" >>> Reanudando: Cargando pesos desde {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        if os.path.exists(optimizer_path):
            print(" >>> Cargando estado del optimizador...")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))

        if os.path.exists(history_path):
            old_history = torch.load(history_path)
            print(f" >>> Historial cargado ({len(old_history['train_loss'])} epochs previos).")
        
        if os.path.exists(times_path):
            old_times = torch.load(times_path)

    # 4. Entrenar
    start_epoch = len(old_history['train_loss'])
    model, new_history, new_times = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        mae_criterion=mae_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        unfreeze_epoch=UNFREEZE_EPOCH,
        patience=PATIENCE,
        device=DEVICE,
        start_epoch=start_epoch
    )

    # --- MERGE DE HISTORIAL Y GUARDADO ---
    for key in old_history:
        old_history[key].extend(new_history.get(key, []))
    
    old_times.update(new_times)

    print(f"Guardando resultados finales...")
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(old_history, history_path)
    torch.save(old_times, times_path)
    
    print(f"Proceso finalizado. Checkpoint guardado en: {checkpoint_dir}")

if __name__ == '__main__':
    main()
