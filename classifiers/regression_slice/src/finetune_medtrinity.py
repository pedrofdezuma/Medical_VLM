import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import re
import random
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Import existing components
from model import get_regression_model
from train import train_model
from utils import plot_learning_curves, evaluate_regression

# --- CONFIGURATION ---
MODEL_NAME = "resnet50"
PRETRAINED_WEIGHTS = r"D:\classifiers\regression_slice\checkpoints\resnet50-98k\best_mri_regression.pth"
DATA_DIR = r"D:\medtrinity-profundidad"
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 30
UNFREEZE_EPOCH = 0
PATIENCE = 7
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET LOGIC ---

def extract_info_medtrinity(filename):
    """
    Extracts prefix and slice number from MedTrinity filenames.
    Formats:
    - brats-goat-00450_t2f_slice_104_z.png
    - MRIe_001_046.png
    """
    # Format: brats-goat-XXXXX_t2f_slice_YYY_z.png
    match1 = re.search(r'(.+)_slice_(\d+)_z\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match1:
        return match1.group(1), int(match1.group(2))
    
    # Format: MRIe_XXX_YYY.png
    match2 = re.search(r'(.+)_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match2:
        return match2.group(1), int(match2.group(2))
    
    return None, None

class MedTrinityRegressionDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.samples = file_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

def get_medtrinity_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, seed=42, num_workers=0):
    random.seed(seed)
    
    all_files = []
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, fname))

    # Group by subject (folder name)
    subject_grouping = defaultdict(list)
    for path in all_files:
        subject_id = os.path.basename(os.path.dirname(path))
        prefix, s_num = extract_info_medtrinity(os.path.basename(path))
        if prefix:
            subject_grouping[subject_id].append((path, s_num))

    processed_samples_by_subject = defaultdict(list)
    for subj_id, slices in subject_grouping.items():
        if not slices: continue
        slice_nums = [s[1] for s in slices]
        s_min, s_max = min(slice_nums), max(slice_nums)
        
        center = (s_max + s_min) / 2.0
        h_range = (s_max - s_min) / 2.0 if s_max != s_min else 1.0
        
        for path, s_val in slices:
            label = (s_val - center) / h_range
            processed_samples_by_subject[subj_id].append((path, float(label)))

    all_subjects = list(processed_samples_by_subject.keys())
    random.shuffle(all_subjects)
    split_idx = int(len(all_subjects) * (1 - val_split))
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]

    train_list = []
    for s in train_subjects: train_list.extend(processed_samples_by_subject[s])
    val_list = []
    for s in val_subjects: val_list.extend(processed_samples_by_subject[s])

    tfs = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_ds = MedTrinityRegressionDataset(train_list, transform=tfs['train'])
    val_ds = MedTrinityRegressionDataset(val_list, transform=tfs['eval'])

    print(f"Total Subjects: {len(all_subjects)}")
    print(f"Train: {len(train_subjects)} sujetos ({len(train_list)} imágenes)")
    print(f"Val:   {len(val_subjects)} sujetos ({len(val_list)} imágenes)")

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return loaders

# --- MAIN TRAINING LOOP ---

def main():
    # 1. Setup paths
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"{MODEL_NAME}-medtrinity"))
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'best_mri_regression.pth')
    history_path = os.path.join(output_dir, 'history.pth')
    times_path = os.path.join(output_dir, 'times_epochs.pth')

    # 2. Load Data
    print(f"Loading MedTrinity loaders from: {DATA_DIR}")
    loaders = get_medtrinity_loaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=0) # 0 for Windows stability

    # 3. Initialize Model and Load Pretrained Weights
    print(f"Initializing {MODEL_NAME}...")
    model = get_regression_model(MODEL_NAME, pretrained=False) # We load specific weights next
    
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f" >>> Loading weights from: {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True))
    else:
        print(f" ERROR: Pretrained weights not found at {PRETRAINED_WEIGHTS}")
        return

    model = model.to(DEVICE)

    # 4. Configure Optimization
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 5. Train
    model, history, times = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        mae_criterion=mae_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        unfreeze_epoch=UNFREEZE_EPOCH,
        patience=PATIENCE,
        device=DEVICE,
        start_epoch=0
    )

    # 6. Save Results
    print(f"Saving fine-tuned model to: {output_dir}")
    torch.save(model.state_dict(), model_path)
    torch.save(history, history_path)
    torch.save(times, times_path)

    # 7. Generate plots
    plot_learning_curves(history, f"{MODEL_NAME}-medtrinity", "MedTrinity", output_dir)
    
    # 8. Final Evaluation
    evaluate_regression(model, loaders['val'], DEVICE, output_dir, split='val_medtrinity')

if __name__ == '__main__':
    main()
