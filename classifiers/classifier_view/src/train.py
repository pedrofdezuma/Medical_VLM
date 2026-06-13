import torch
import time
import copy
from tqdm import tqdm # For a real progress bar
import sys

def train_model(model,dataloaders,criterion,optimizer,scheduler,unfreeze_epoch,patience,pretrained,device="cuda"):
    train_start_time=time.time()
    
    # This variable stores the best model weights. The model with these weights will be returned.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Store the time each epoch took to execute
    times_training_epochs={}
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    patience_counter = 0
    epoch = 0
    
    try:
        while True:
           
            epoch_start_time=time.time() # Start counting the epoch's training duration
            
            print(f'Epoch {epoch + 1}')
            print('-' * 10)
            
            
             # ---  UNFREEZING LOGIC  ---
            if pretrained and epoch == unfreeze_epoch:
                print("Unfreezing layers for Fine-Tuning...")
                for param in model.parameters():
                    # All parameters will be learned from now on
                    param.requires_grad = True
                # Lower the learning rate to 1e-5 or the lr previously held by the optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=min(1e-5,optimizer.param_groups[0]['lr']))
            
            
           
            for phase in ['train','val']:
                if phase=='train':
                    phase_train_start_time=time.time()
                    model.train()
                else:
                    phase_val_start_time=time.time()
                    model.eval()
    
                current_loss = 0.0
                current_corrects = 0
    
                # Progress bar
                # dataloaders[phase] contains all batches. If we have 3200 images and a batch_size of 32, we iterate 100 times.
                # desc indicates the current phase
                # leave=False removes the bar from the console once finished
                pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}", unit="batch", leave=False)
        
                # dataloaders=(images, labels)
                for inputs,labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Before analyzing a new batch, clear what was learned from the previous one for clean calculations.
                    optimizer.zero_grad()
                    
                    # Forward pass
                    # Enable or disable gradient calculation depending on the phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)   # e.g.: [-2.34, 4.12, -0.15]
                        _, preds = torch.max(outputs, 1) # preds is the index of the highest class value
                        loss = criterion(outputs, labels)
    
                        # Backward pass + Optimize only in training
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    current_loss += loss.item() * inputs.size(0)
                    current_corrects += torch.sum(preds == labels.data)
                    
                    # Update info in the batch progress bar
                    pbar.set_postfix(loss=loss.item())
                
                dataset_size = len(dataloaders[phase].dataset)
                epoch_loss = current_loss / dataset_size
                epoch_acc = current_corrects.double() / dataset_size
    
                print(f'{phase.capitalize()} Final -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
                # Save in history
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                    phase_train_end_time=time.time()
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    
                    # UPDATE SCHEDULER
                    # ReduceLROnPlateau needs the validation accuracy
                    scheduler.step(epoch_acc)
    
                    # Save the best model
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        if epoch >= unfreeze_epoch:
                            patience_counter = 0
                    else:
                        if epoch >= unfreeze_epoch:
                            patience_counter += 1
                    
                    phase_val_end_time=time.time()
                        
            history['lr'].append(optimizer.param_groups[0]['lr'])
            epoch_end_time=time.time()
            epoch_elapsed_time=epoch_end_time-epoch_start_time
            
            times_training_epochs[epoch]={
                "train":phase_train_end_time-phase_train_start_time,
                "val":phase_val_end_time-phase_val_start_time,
                "total":epoch_elapsed_time
                
                }
            
            if epoch >= unfreeze_epoch and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
                
            epoch += 1
            
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C). Saving the best model achieved so far...")
        
        # Ensure all history lists have the same length in case of an interruption during the epoch
        min_len = min(len(history['train_loss']), len(history['val_loss']), len(history['train_acc']), len(history['val_acc']), len(history['lr']))
        history['train_loss'] = history['train_loss'][:min_len]
        history['val_loss'] = history['val_loss'][:min_len]
        history['train_acc'] = history['train_acc'][:min_len]
        history['val_acc'] = history['val_acc'][:min_len]
        history['lr'] = history['lr'][:min_len]
        
        keys_to_remove = [k for k in times_training_epochs.keys() if k >= min_len]
        for k in keys_to_remove:
            del times_training_epochs[k]

    total_time = time.time() - train_start_time
    print('+'*30)
    print(f'Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')
    print('+'*30)
    
    model.load_state_dict(best_model_wts)
    
    return model,history,times_training_epochs
    