import torch.optim as optim
from torch.optim import lr_scheduler
import torch 
import torch.nn as nn
import os
import random
import numpy as np

from dataset import get_mri_loaders
from model import get_mri_model
from train import train_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---  CONFIGURATION AND HYPERPARAMETERS ---
NUM_TRAIN=str(4)+"k"

BATCH_SIZE = 32
UNFREEZE_EPOCH = 9         
PATIENCE = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED=True
NON_ROTATED=True
NO_AUGMENTATION=True

MODEL_NAME="mobilenet_v2" # Can be either "resnet18" or "vgg16" or "mobilenet_v2"
#DATA_DIR = f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/dataset/classifier_view/{NUM_TRAIN}"
DATA_DIR = f'D:/classifiers/classifier_view/data/{NUM_TRAIN}'
if NON_ROTATED:
    DATA_DIR += '-EN'  #CAMBIADO


def main():
    set_seed(42)
    
    loaders, sizes, classes = get_mri_loaders(DATA_DIR, batch_size=BATCH_SIZE, no_augmentation=NO_AUGMENTATION)
    model = get_mri_model(model_name=MODEL_NAME,
                          pretrained=PRETRAINED,
                          num_classes=len(classes)).to(DEVICE)
                        
    checkpoint_dir=f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/classifiers/classifier_view/checkpoints/{model.model_name}-{NUM_TRAIN}"
    #checkpoint_dir = f'D:/classifiers/classifier_view/checkpoints/{model.model_name}-{NUM_TRAIN}'
    
    if PRETRAINED:
        checkpoint_dir+='-PT'
    if NON_ROTATED:
        checkpoint_dir+='-EN' #CAMBIADO
    if NO_AUGMENTATION:
        checkpoint_dir+='-NA'
    
    
    model_path = os.path.join(checkpoint_dir, 'best_mri_classifier.pth')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pth') 
    history_path = os.path.join(checkpoint_dir, 'history.pth')
    times_path = os.path.join(checkpoint_dir, 'times_epochs.pth')



    # Apply weights to the CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # --- ENHANCED LOADING LOGIC ---
    old_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    old_times = {}

    """
    if os.path.exists(model_path):
        print(f" Resuming: Loading weights...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        if os.path.exists(optimizer_path):
            print(f" Loading optimizer state...")
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

        if os.path.exists(history_path):
            old_history = torch.load(history_path, weights_only=True)
            old_times = torch.load(times_path, weights_only=True)
            print(f" History loaded ({len(old_history['train_loss'])} previous epochs). Data Augmentation: {'No' if NO_AUGMENTATION else 'Yes'}")
     """
    try:
        model, new_history, new_times = train_model(
                                            model,
                                            loaders,
                                            criterion,
                                            optimizer,
                                            scheduler, 
                                            unfreeze_epoch=UNFREEZE_EPOCH,
                                            patience=PATIENCE,
                                            pretrained=PRETRAINED,
                                            device=DEVICE
                                            )
    except Exception:
        print("\n[!] main.py caught Exception. Saving currently returned state from train_model...")
        # Since train_model also catches it, it will return the state up to the interruption.
        # We just need to ensure the variables are assigned properly.
        # However, if train_model catches it successfully, it returns normally, so this except block might only trigger if the interrupt happens outside train_model or during the return.
        pass # Actually, if train_model handles it, we don't strictly need it here, but it's good for safety.

    # --- MERGING AND SAVING ---
    if 'new_history' in locals():
        for key in old_history:
            old_history[key].extend(new_history[key])
        
        total_times = old_times | new_times
    else:
        print("Training did not return successfully. Saving only old history if any.")
        total_times = old_times

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path) # <--- SAVE OPTIMIZER 
    torch.save(old_history, history_path)
    torch.save(total_times, times_path)
    
    print(f" Process finished. Total accumulated epochs saved: {len(old_history['train_loss'])}")
    
if __name__=='__main__':
    main()
