import torch.optim as optim
from torch.optim import lr_scheduler
import torch 
import torch.nn as nn
import os

from dataset import get_mri_loaders
from model import get_mri_model
from train import train_model

# ---  CONFIGURATION AND HYPERPARAMETERS ---
NUM_TRAIN=str(4)+"k"

BATCH_SIZE = 32
UNFREEZE_EPOCH = 9         
PATIENCE = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED=True
NON_ROTATED=True
MODEL_NAME="vgg16" # You can change this to "vgg16" or "mobilenet_v2"

DATA_DIR = f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/dataset/classifier_sequence/{NUM_TRAIN}"
#DATA_DIR = f'D:/classifiers/classifier_sequence/data/{NUM_TRAIN}'
if NON_ROTATED:
    DATA_DIR += '-NR'

def main():
    loaders, sizes, classes = get_mri_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    model = get_mri_model(model_name=MODEL_NAME,
                          pretrained=PRETRAINED,
                          num_classes=len(classes)).to(DEVICE)
                        
    checkpoint_dir = f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/classifiers/classifier_sequence/checkpoints/{model.model_name}-{NUM_TRAIN}"
    #checkpoint_dir = f'D:/classifiers/classifier_sequence/checkpoints/{model.model_name}-{NUM_TRAIN}'
    
    if PRETRAINED:
        checkpoint_dir+='-PT'
    if NON_ROTATED:
        checkpoint_dir+='-NR'
    
    model_path = os.path.join(checkpoint_dir, 'best_mri_classifier.pth')
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pth') 
    history_path = os.path.join(checkpoint_dir, 'history.pth')
    times_path = os.path.join(checkpoint_dir, 'times_epochs.pth')

    # Initial configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # --- ENHANCED LOADING LOGIC ---
    old_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    old_times = {}

    
   
    if os.path.exists(model_path):
        print(f" Resuming: Loading weights...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        if os.path.exists(optimizer_path):
            print(f" Loading optimizer state...")
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

        if os.path.exists(history_path):
            old_history = torch.load(history_path, weights_only=True)
            old_times = torch.load(times_path, weights_only=True)
            print(f" History loaded ({len(old_history['train_loss'])} previous epochs).")
     

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
    except KeyboardInterrupt:
        print("\n[!] main.py caught KeyboardInterrupt. Saving currently returned state from train_model...")
        pass

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
