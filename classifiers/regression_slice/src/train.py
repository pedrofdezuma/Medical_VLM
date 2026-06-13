import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from tqdm import tqdm

def train_model(model, loaders, criterion, mae_criterion, optimizer, scheduler, unfreeze_epoch, patience, device, start_epoch=0):
    """
    Entrenamiento con bucle infinito y conocimiento del epoch de inicio.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'lr': []}
    times_training_epochs = {}
    
    print(f"Inicio/Reanudación del entrenamiento en epoch {start_epoch + 1} sobre {device}")
    
    epoch = start_epoch
    try:
        while True:
            epoch_start_time = time.time()
            print(f'Epoch {epoch + 1}')
            print('-' * 10)

            # --- LÓGICA DE DESCONGELADO (Fine-Tuning) ---
            # Si ya pasamos el epoch de unfreeze (ej: al reanudar), se activa inmediatamente
            if epoch >= unfreeze_epoch:
                # Solo imprimimos el mensaje la primera vez que ocurre en esta sesión
                if any(p.requires_grad == False for p in model.parameters()):
                    print(">>> Fine-Tuning: Descongelando todas las capas...")
                    for param in model.parameters():
                        param.requires_grad = True
                    current_lr = optimizer.param_groups[0]['lr']
                    optimizer = optim.Adam(model.parameters(), lr=min(1e-5, current_lr))

            phase_times = {}
            for phase in ['train', 'val']:
                phase_start_time = time.time()
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_mae = 0.0

                pbar = tqdm(loaders[phase], desc=f"{phase.capitalize()}", unit="batch", leave=False)
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        mae = mae_criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_mae += mae.item() * inputs.size(0)
                    pbar.set_postfix(loss=loss.item())

                dataset_size = len(loaders[phase].dataset)
                epoch_loss = running_loss / dataset_size
                epoch_mae = running_mae / dataset_size

                print(f'{phase.capitalize()} Final -> Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}')
                
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_mae'].append(epoch_mae)
                phase_times[phase] = time.time() - phase_start_time

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        if epoch >= unfreeze_epoch:
                            patience_counter = 0
                    else:
                        if epoch >= unfreeze_epoch:
                            patience_counter += 1

            history['lr'].append(optimizer.param_groups[0]['lr'])
            epoch_end_time = time.time()
            times_training_epochs[epoch] = {
                "train": phase_times['train'],
                "val": phase_times['val'],
                "total": epoch_end_time - epoch_start_time
            }
            
            if epoch >= unfreeze_epoch and patience_counter >= patience:
                print(f"Early stopping activado después de {epoch + 1} épocas.")
                break
            
            epoch += 1

    except KeyboardInterrupt:
        print("\n[!] Interrumpido por usuario. Sincronizando historial...")
        min_len = min(len(history['train_loss']), len(history['val_loss']))
        for k in history: 
            if isinstance(history[k], list):
                history[k] = history[k][:min_len]

    model.load_state_dict(best_model_wts)
    return model, history, times_training_epochs
