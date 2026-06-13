import pickle
import os

pkl_path = 'classifiers/subjects_split.pkl'
output_dir = 'classifiers'

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    train_subjects_all = sorted(list(data.get('train_subjects', [])))
    val_subjects_all = sorted(list(data.get('val_subjects', [])))
    
    # Filter only Multiple Sclerosis (MS/EM) subjects
    train_subjects_ms = [s for s in train_subjects_all if 'ms' in s.lower()]
    val_subjects_ms = [s for s in val_subjects_all if 'ms' in s.lower()]
    
    # Save MS training subjects to a separate pkl
    train_output_path = os.path.join(output_dir, 'train_subjects_ms.pkl')
    with open(train_output_path, 'wb') as f:
        pickle.dump(train_subjects_ms, f)
    
    # Save MS validation subjects to a separate pkl
    val_output_path = os.path.join(output_dir, 'val_subjects_ms.pkl')
    with open(val_output_path, 'wb') as f:
        pickle.dump(val_subjects_ms, f)
    
    print(f"Archivos creados en {output_dir} (SOLO Esclerosis Múltiple):")
    print(f"  - train_subjects_ms.pkl ({len(train_subjects_ms)} sujetos)")
    print(f"  - val_subjects_ms.pkl ({len(val_subjects_ms)} sujetos)")
    
    print(f"\nLista de Validación MS:")
    print(", ".join(val_subjects_ms))
else:
    print(f"File not found: {pkl_path}")
