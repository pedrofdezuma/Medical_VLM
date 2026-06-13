import os
import re
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from collections import defaultdict

# --- FUNCIONES DE APOYO (Definidas a nivel global para que Windows pueda serializarlas) ---

def extract_info(filename):
    # Captura el prefijo completo: ej MRIms_541 de MRIms_541_050.png
    match = re.search(r'(.+)_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def rotate_0(x): return x
def rotate_90(x): return TF.rotate(x, 90)
def rotate_180(x): return TF.rotate(x, 180)
def rotate_270(x): return TF.rotate(x, 270)

class SliceRegressionDataset(Dataset):
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

def get_regression_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, seed=42, num_workers=4):
    random.seed(seed)
    
    # 1. Escaneo de todos los archivos en imagesTr
    all_tr_files = []
    for root, _, fnames in os.walk(data_dir):
        if 'imagesTr' in root:
            for fname in fnames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_tr_files.append(os.path.join(root, fname))

    # 2. Agrupar por Sujeto y Volumen para calcular labels
    vol_grouping = defaultdict(list)
    for path in all_tr_files:
        prefix, s_num = extract_info(os.path.basename(path))
        if prefix:
            vol_key = os.path.join(os.path.dirname(path), prefix)
            vol_grouping[vol_key].append((path, s_num, prefix))

    # 3. Calcular labels y organizar por Sujeto Único
    processed_samples_by_subject = defaultdict(list)
    for vol_key, slices in vol_grouping.items():
        slice_nums = [s[1] for s in slices]
        s_min, s_max = min(slice_nums), max(slice_nums)
        subj_id = slices[0][2]
        
        center = (s_max + s_min) / 2.0
        h_range = (s_max - s_min) / 2.0 if s_max != s_min else 1.0
        
        for path, s_val, _ in slices:
            label = (s_val - center) / h_range
            if 'sagittal' in path.lower():
                label = abs(label)
            processed_samples_by_subject[subj_id].append((path, float(label)))

    # 4. División por Sujetos
    all_subjects = list(processed_samples_by_subject.keys())
    random.shuffle(all_subjects)
    split_idx = int(len(all_subjects) * (1 - val_split))
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]

    train_list = []
    for s in train_subjects: train_list.extend(processed_samples_by_subject[s])
    val_list = []
    for s in val_subjects: val_list.extend(processed_samples_by_subject[s])

    # 5. Cargar imagesTs
    test_list = []
    for root, _, fnames in os.walk(data_dir):
        if 'imagesTs' in root:
            temp_vol = defaultdict(list)
            for fname in fnames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    p = os.path.join(root, fname)
                    pref, s_n = extract_info(fname)
                    if pref:
                        v_k = os.path.join(root, pref)
                        temp_vol[v_k].append((p, s_n))
            for v_k, slips in temp_vol.items():
                s_nums = [s[1] for s in slips]
                mi, ma = min(s_nums), max(s_nums)
                ce, hr = (ma + mi) / 2.0, (ma - mi) / 2.0 if ma != mi else 1.0
                for p, sv in slips:
                    label = (sv - ce)/hr
                    if 'sagittal' in p.lower():
                        label = abs(label)
                    test_list.append((p, float(label)))

    # Transformaciones con funciones nombradas (para compatibilidad con Windows)
    tfs = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.Lambda(rotate_0),
                transforms.Lambda(rotate_90),
                transforms.Lambda(rotate_180),
                transforms.Lambda(rotate_270),
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_ds = SliceRegressionDataset(train_list, transform=tfs['train'])
    val_ds = SliceRegressionDataset(val_list, transform=tfs['eval'])
    test_ds = SliceRegressionDataset(test_list, transform=tfs['eval'])

    print(f"Sujetos totales en imagesTr: {len(all_subjects)}")
    print(f"Train: {len(train_subjects)} sujetos ({len(train_list)} imágenes)")
    print(f"Val:   {len(val_subjects)} sujetos ({len(val_list)} imágenes)")
    print(f"Test:  {len(test_list)} imágenes (desde imagesTs)")

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        'train_subjects': train_subjects,
        'val_subjects': val_subjects
    }
    return loaders
