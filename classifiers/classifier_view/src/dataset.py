import os
import torch
import pickle
import re
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

class MrIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, transform=None, all_to_test=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.all_to_test = all_to_test
        self.classes = ['axial', 'coronal', 'sagittal','non_brain_mri']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} #axial: 0, coronal:1, sagittal:2, non_mri:3
        self.samples = self._make_dataset()

    def _make_dataset(self):
        instances = []
        
        # If all_to_test is True, only 'test' split contains data
        if self.all_to_test and self.split != 'test':
            return []

        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            
            # Determine which splits to look into
            if self.all_to_test:
                splits_to_check = ['train', 'val', 'test']
            else:
                splits_to_check = [self.split]

            for s in splits_to_check:
                # Construct path: data/axial/train
                target_dir = os.path.join(self.root_dir, target_class, s)
                if not os.path.isdir(target_dir):
                    continue
                    
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if self._is_valid_file(path):
                            instances.append((path, class_index))
        return instances

    def _is_valid_file(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # Load image
        from PIL import Image
        sample = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target

class RegressionSliceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=None, pkl_path=None, all_to_test=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.all_to_test = all_to_test
        self.classes = ['axial', 'coronal', 'sagittal','non_brain_mri']
        
        # Load splits from pkl
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                split_data = pickle.load(f)
            self.train_subjects = set(split_data['train_subjects'])
            self.val_subjects = set(split_data['val_subjects'])
        else:
            self.train_subjects = set()
            self.val_subjects = set()

        self.samples = self._make_dataset()

    def _make_dataset(self):
        instances = []
        def extract_subject_id(filename):
            match = re.search(r'(.+)_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
            return match.group(1) if match else None

        def get_label_from_path(path):
            path_lower = path.lower()
            if 'axial' in path_lower: return 0
            if 'coronal' in path_lower: return 1
            if 'sagittal' in path_lower: return 2
            if 'non_brain' in path_lower or 'non_mri' in path_lower: return 3
            return 0 # Fallback

        # Búsqueda recursiva flexible (igual que en el cargador original de regresión)
        if self.all_to_test:
            if self.split != 'test':
                return []
            target_split_folders = ['imagesTs', 'imagesTr']
        else:
            target_split_folders = ['imagesTs'] if self.split == 'test' else ['imagesTr']
            
        target_subjects = self.train_subjects if self.split == 'train' else self.val_subjects

        if os.path.isdir(self.data_dir):
            for root, _, fnames in os.walk(self.data_dir):
                # Check if current root is in one of the target split folders
                if any(tsf in root for tsf in target_split_folders):
                    for fname in fnames:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            full_path = os.path.join(root, fname)
                            
                            # Si es train/val (y no estamos en all_to_test), filtramos por sujeto
                            if not self.all_to_test and self.split != 'test':
                                subj_id = extract_subject_id(fname)
                                if subj_id in target_subjects:
                                    instances.append((full_path, get_label_from_path(full_path)))
                            else:
                                # Si es test (o all_to_test), incluimos todo lo que esté en las carpetas seleccionadas
                                instances.append((full_path, get_label_from_path(full_path)))
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        from PIL import Image
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class RotateTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        if self.angle == 0:
            return x
        return TF.rotate(x, self.angle)

def get_mri_loaders(data_dir, batch_size=32, no_augmentation=False, use_regression_split=False, pkl_path=None, all_to_test=False):
    """
    Creates and returns DataLoaders for training and validation.
    """

    if no_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),      # Standard size for ResNet
            transforms.RandomHorizontalFlip(),  # Standard data augmentation

            # Apply only 0, 90, 180, or 270 degree rotations
            transforms.RandomChoice([
                RotateTransform(0),
                RotateTransform(90),
                RotateTransform(180),
                RotateTransform(270),
            ]),

            transforms.ToTensor(),              # Converts image to PyTorch tensor [0, 1]        
            # Standard ImageNet normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_transforms = {
        'train': train_transforms,
    
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

    #-------------------------------------------------------------------------------------
    # Adaptation for folder structure: data/{class}/{split}
    
    # Create datasets
    if use_regression_split:
        image_datasets = {
            x: RegressionSliceDataset(data_dir, x, data_transforms[x], pkl_path, all_to_test=all_to_test)
            for x in ['train', 'val', 'test']
        }
    else:
        image_datasets = {
            x: MrIDataset(data_dir, x, data_transforms[x], all_to_test=all_to_test)
            for x in ['train', 'val','test']
        }

    #------------------------------------------------------------------------------------
    # Create dataloaders
    dataloaders = {}
    for x in ['train', 'val', 'test']:
        if len(image_datasets[x]) > 0:
            dataloaders[x] = DataLoader(
                image_datasets[x], 
                batch_size=batch_size, 
                shuffle=True if x == 'train' else False, # Only shuffle in training
                num_workers=0 # Set to 0 on Windows to avoid WinError 1455
            )
        else:
            if not all_to_test or x == 'test':
                print(f"Warning: Dataset for split '{x}' is empty. Loader will not be created.")

    # Extract useful information
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes 

    return dataloaders, dataset_sizes, class_names