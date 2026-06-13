import os
import shutil
import pickle
import re
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_BASE = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1\MRIms_kde"
DEST_SEQ = r"D:\classifiers\classifier_sequence\data\4k-NR"
DEST_VIEW = r"D:\classifiers\classifier_view\data\4k-NR"

PKL_TRAIN = r"classifiers\train_subjects_ms.pkl"
PKL_VAL = r"classifiers\val_subjects_ms.pkl"

# Modality mapping
MOD_MAP = {
    'FLAIR': 'flair',
    'T1w': 't1-w'
}

# --- LOAD SUBJECTS ---
with open(PKL_TRAIN, 'rb') as f:
    train_subjects = set(pickle.load(f))
with open(PKL_VAL, 'rb') as f:
    val_subjects = set(pickle.load(f))

def extract_subject_id(filename):
    match = re.search(r'(.+)_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    return match.group(1) if match else None

def process():
    files_to_copy = []
    
    # 1. Collect all files from source
    for modality in ['FLAIR', 'T1w']:
        mod_dir = os.path.join(SOURCE_BASE, modality)
        if not os.path.exists(mod_dir): continue
        
        for folder in ['imagesTr', 'imagesTs']:
            folder_dir = os.path.join(mod_dir, folder)
            if not os.path.exists(folder_dir): continue
            
            for view in ['axial', 'coronal', 'sagittal']:
                view_dir = os.path.join(folder_dir, view)
                if not os.path.exists(view_dir): continue
                
                for filename in os.listdir(view_dir):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    
                    src_path = os.path.join(view_dir, filename)
                    
                    # Determine split
                    if folder == 'imagesTs':
                        split = 'test'
                    else:
                        subj_id = extract_subject_id(filename)
                        if subj_id in train_subjects:
                            split = 'train'
                        elif subj_id in val_subjects:
                            split = 'val'
                        else:
                            # Skip if subject not in our split (should not happen given previous steps)
                            continue
                    
                    files_to_copy.append({
                        'src': src_path,
                        'filename': filename,
                        'split': split,
                        'modality': MOD_MAP[modality],
                        'view': view
                    })

    print(f"Total files identified to add: {len(files_to_copy)}")
    
    # 2. Perform copy
    for item in tqdm(files_to_copy, desc="Adding images"):
        # Copy to Sequence Classifier
        d1 = os.path.join(DEST_SEQ, item['modality'], item['split'])
        os.makedirs(d1, exist_ok=True)
        shutil.copy2(item['src'], os.path.join(d1, item['filename']))
        
        # Copy to View Classifier
        d2 = os.path.join(DEST_VIEW, item['view'], item['split'])
        os.makedirs(d2, exist_ok=True)
        shutil.copy2(item['src'], os.path.join(d2, item['filename']))

if __name__ == "__main__":
    process()
    print("Done!")
