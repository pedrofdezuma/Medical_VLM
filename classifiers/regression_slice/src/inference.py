import torch
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import get_regression_loaders, extract_info
from model import get_regression_model
from utils import analyze_regression_errors

# --- CONFIGURACIÓN ---
MODEL_NAME = "efficientnet_v2_s"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_TO_EVAL = 'test'  # Opciones: 'train', 'val', 'test'

# --- CONFIGURACIÓN CLASIFICADOR (OPCIONAL) ---
USE_PLANE_CLASSIFIER = True  # Si es True, usa el modelo para detectar el plano. Si False, usa el Path.
PLANE_MODEL_NAME = "resnet18"
PLANE_CHECKPOINT_PATH = r"D:\classifiers\classifier_view\checkpoints\resnet18-4k-PT-NR\best_mri_classifier.pth"

# Intentar detectar si estamos en Picasso o Local
PICASSO = os.path.exists("/mnt2/fscratch")

if PICASSO:
    DATA_DIR = r"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/2D_MRI_ms_ep_control_FLAIR_T1"
    BASE_CHECKPOINT_PATH = r"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/classifiers/regression_slice/checkpoints"
else:
    DATA_DIR = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
    #DATA_DIR = r"D:\axial-flair-dataset\shard_57"
    BASE_CHECKPOINT_PATH = r"D:\classifiers\regression_slice\checkpoints"

def load_plane_classifier():
    """Carga el modelo de clasificación de planos evitando conflictos de nombres."""
    import importlib.util
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    model_path = os.path.join(project_root, "classifiers", "classifier_view", "src", "model.py")
    
    # Carga dinámica del módulo para evitar conflicto con el 'model.py' local
    spec = importlib.util.spec_from_file_location("model_classifier", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    print(f" >>> Cargando Clasificador de Planos desde: {PLANE_CHECKPOINT_PATH}")
    model = model_module.get_mri_model(PLANE_MODEL_NAME, pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(PLANE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def run_inference():
    print(f"Cargando datos desde: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"Error: No se encuentra el directorio de datos en {DATA_DIR}")
        return

    # Lógica especial para shard_57 (primero 1000 imágenes)
    if "shard_57" in DATA_DIR:
        from dataset import SliceRegressionDataset
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_files = sorted(all_files)[:1000] # Primeras 1000 imagenes
        
        samples = [(p, 0.0) for p in all_files] # Label dummy (no tenemos GT para este shard)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = SliceRegressionDataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        loaders = {SPLIT_TO_EVAL: loader}
        train_k = "98k" # Usamos el checkpoint de 98k por defecto
    else:
        loaders = get_regression_loaders(DATA_DIR, batch_size=BATCH_SIZE)
        num_train_imgs = len(loaders['train'].dataset)
        train_k = f"{num_train_imgs // 1000}k"
    
    checkpoint_name = f"{MODEL_NAME}-{train_k}"
    checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, checkpoint_name)
    
    if not os.path.exists(checkpoint_dir) and MODEL_NAME == "resnet18":
        checkpoint_dir = os.path.join(BASE_CHECKPOINT_PATH, "resnet18-90k")
    
    model_path = os.path.join(checkpoint_dir, 'best_mri_regression.pth')
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo en {model_path}")
        return

    print(f"Usando modelo regresor: {model_path}")

    model_reg = get_regression_model(MODEL_NAME, pretrained=False)
    model_reg.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model_reg.to(DEVICE)
    model_reg.eval()

    plane_classifier = None
    if USE_PLANE_CLASSIFIER:
        if os.path.exists(PLANE_CHECKPOINT_PATH):
            plane_classifier = load_plane_classifier()
        else:
            print(f"AVISO: No se encontró el clasificador en {PLANE_CHECKPOINT_PATH}. Se usará la lógica de PATH.")

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    mode_str = "CLS" if plane_classifier else "PATH"
    
    prefix = "MedTrinity_" if "shard_57" in DATA_DIR else ""
    folder_name = f"{prefix}{SPLIT_TO_EVAL.upper()}_{mode_str}_{timestamp}"
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(project_root, "classifiers", "regression_slice", "inference_results", f"{MODEL_NAME}_{train_k}", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    metrics, df_preds = evaluate_with_optional_classifier(
        model=model_reg,
        dataloader=loaders[SPLIT_TO_EVAL],
        device=DEVICE,
        save_dir=save_dir,
        split=SPLIT_TO_EVAL,
        plane_classifier=plane_classifier,
        all_loaders=loaders
    )

    print("\n" + "="*40)
    print(f"       RESULTADOS INFERENCIA ({SPLIT_TO_EVAL.upper()})")
    print(f"       MODO: {mode_str}")
    print("="*40)
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"R2 : {metrics['R2']:.4f}")
    print("="*40)

    analyze_regression_errors(df_preds, save_dir)
    print(f"\n[OK] Resultados guardados en: {save_dir}")

def get_metadata_from_path(path):
    """Extrae metadatos (plano, secuencia, enfermedad, etc.) a partir de la ruta del archivo."""
    filename = os.path.basename(path)
    p_n = path.lower().replace('\\', '/')
    
    # Soporte para formato shard_57: mr_flair--brats2021--brats2021_00587--x_0096--0007_002.png
    if '--' in filename:
        parts = filename.split('--')
        sequence = 'unknown'
        if 'flair' in parts[0].lower(): sequence = 'flair'
        elif 't1' in parts[0].lower(): sequence = 't1w'
        
        plane = 'unknown'
        real_slice = 0
        volume_prefix = 'unknown'
        
        for p in parts:
            if p.startswith('x_'): 
                plane = 'axial'
                try: real_slice = int(p.split('_')[1])
                except: pass
            elif p.startswith('y_'): 
                plane = 'coronal'
                try: real_slice = int(p.split('_')[1])
                except: pass
            elif p.startswith('z_'): 
                plane = 'sagittal'
                try: real_slice = int(p.split('_')[1])
                except: pass
        
        if len(parts) >= 3:
            volume_prefix = f"{parts[1]}--{parts[2]}"
            
        disease = 'Unknown'
        return {
            'filename': filename, 'plane': plane, 'sequence': sequence,
            'disease': disease, 'volume_prefix': volume_prefix, 'real_slice': real_slice
        }

    if '/axial/' in p_n: plane = 'axial'
    elif '/sagittal/' in p_n: plane = 'sagittal'
    elif '/coronal/' in p_n: plane = 'coronal'
    else: plane = 'unknown'
    
    if '/flair/' in p_n: sequence = 'flair'
    elif '/t1' in p_n: sequence = 't1w'
    else: sequence = 'unknown'
    
    if 'mricontrol' in p_n: disease = 'Control'
    elif 'mrims' in p_n: disease = 'EM'
    elif 'mrie' in p_n: disease = 'Epilepsia'
    else: disease = 'Unknown'
    
    volume_prefix, real_slice = extract_info(filename)
    
    return {
        'filename': filename,
        'plane': plane,
        'sequence': sequence,
        'disease': disease,
        'volume_prefix': volume_prefix,
        'real_slice': real_slice
    }


def evaluate_with_optional_classifier(model, dataloader, device, save_dir, split, plane_classifier=None, all_loaders=None):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    samples = dataloader.dataset.samples

    print(f"Evaluando... (Uso de Clasificador: {'SÍ' if plane_classifier else 'NO'})")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Inferencia")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.cpu().numpy().flatten()
            
            if plane_classifier:
                plane_logits = plane_classifier(inputs)
                plane_preds = torch.argmax(plane_logits, dim=1).cpu().numpy()
                for j in range(len(preds)):
                    if plane_preds[j] == 2: # Sagittal
                        preds[j] = abs(preds[j])
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            
            batch_size = inputs.size(0)
            start_idx = i * dataloader.batch_size
            all_paths.extend([s[0] for s in samples[start_idx:start_idx + batch_size]])

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    
    # Construir DataFrame para análisis detallado usando la función helper
    metadata_list = [get_metadata_from_path(p) for p in all_paths]
    df_preds = pd.DataFrame(metadata_list)
    df_preds['real'] = all_labels
    df_preds['pred'] = all_preds
    df_preds['error'] = np.abs(all_labels - all_preds)

    # --- CONSTRUIR REFERENCIA GLOBAL (Si se proporcionan todos los loaders) ---
    df_ref = None
    if all_loaders:
        print("Construyendo referencia global de todos los slices...")
        ref_list = []
        for s_key in ['train', 'val', 'test']:
            if s_key in all_loaders:
                for p, l in all_loaders[s_key].dataset.samples:
                    meta = get_metadata_from_path(p)
                    meta['real'] = l
                    ref_list.append(meta)
        df_ref = pd.DataFrame(ref_list)
    
    # 2. Función para encontrar el slice más cercano (SIN INTERPOLACIÓN)
    def find_closest_slice(group):
        if len(group) == 0: return group
        
        preds = group['pred'].values
        real_slices = group['real_slice'].values
        plane = group['plane'].iloc[0]
        
        # Buscar candidatos en la referencia global si existe
        if df_ref is not None:
            # Atributos del grupo para filtrar la referencia
            disease = group['disease'].iloc[0]
            v_prefix = group['volume_prefix'].iloc[0]
            plane = group['plane'].iloc[0]
            seq = group['sequence'].iloc[0]
            
            mask = (df_ref['disease'] == disease) & \
                   (df_ref['volume_prefix'] == v_prefix) & \
                   (df_ref['plane'] == plane) & \
                   (df_ref['sequence'] == seq)
            candidates = df_ref[mask]
            
            if not candidates.empty:
                c_reals = candidates['real'].values
                c_slices = candidates['real_slice'].values
                
                if plane == 'sagittal':
                    # Para Sagittal, las etiquetas suelen ser abs(label), lo que genera ambigüedad (dos slices con misma label).
                    # Disambiguamos eligiendo el candidato que esté en el mismo lado del centro que el real_slice.
                    center = (np.min(c_slices) + np.max(c_slices)) / 2.0
                    pred_slices = []
                    
                    for i in range(len(preds)):
                        p_val = preds[i]
                        r_slice = real_slices[i]
                        
                        # Calcular distancias a todos los candidatos
                        dists = np.abs(p_val - c_reals)
                        # Ordenar candidatos por cercanía en valor de label
                        sorted_indices = np.argsort(dists)
                        
                        # Buscar el mejor que esté en el mismo lado que el real_slice
                        r_side = (r_slice >= center)
                        best_idx = sorted_indices[0] # Fallback por si acaso
                        
                        for idx in sorted_indices:
                            c_slice = c_slices[idx]
                            c_side = (c_slice >= center)
                            if c_side == r_side:
                                best_idx = idx
                                break
                        pred_slices.append(c_slices[best_idx])
                    
                    group['pred_slice'] = pred_slices
                    return group
                else:
                    # Búsqueda normal del más cercano en otros planos
                    distances = np.abs(preds[:, np.newaxis] - c_reals[np.newaxis, :])
                    closest_indices = np.argmin(distances, axis=1)
                    group['pred_slice'] = c_slices[closest_indices]
                    return group

        # Fallback: buscar el más cercano entre los disponibles en el grupo actual (si no hay df_ref)
        reals = group['real'].values
        slices = group['real_slice'].values
        
        if plane == 'sagittal':
            center = (np.min(slices) + np.max(slices)) / 2.0
            pred_slices = []
            for i in range(len(preds)):
                p_val = preds[i]
                r_slice = real_slices[i]
                dists = np.abs(p_val - reals)
                sorted_indices = np.argsort(dists)
                r_side = (r_slice >= center)
                best_idx = sorted_indices[0]
                for idx in sorted_indices:
                    if (slices[idx] >= center) == r_side:
                        best_idx = idx
                        break
                pred_slices.append(slices[best_idx])
            group['pred_slice'] = pred_slices
        else:
            distances = np.abs(preds[:, np.newaxis] - reals[np.newaxis, :])
            closest_indices = np.argmin(distances, axis=1)
            group['pred_slice'] = slices[closest_indices]
            
        return group

    # Aplicar por volumen, plano y secuencia para evitar mezclar datos con normalizaciones distintas
    df_preds = df_preds.groupby(['disease', 'volume_prefix', 'plane', 'sequence'], group_keys=False).apply(find_closest_slice)
    
    # 3. Calcular error en slices
    df_preds['error_slice'] = (df_preds['real_slice'] - df_preds['pred_slice']).abs()
    
    # Eliminar columna auxiliar y reordenar
    cols = ['filename', 'plane', 'sequence', 'disease', 'real', 'pred', 'error', 'real_slice', 'pred_slice', 'error_slice']
    df_preds = df_preds[cols]
    
    df_preds.to_csv(os.path.join(save_dir, f"predictions_{split}.csv"), index=False)
    
    metrics = {
        'MAE': mean_absolute_error(all_labels, all_preds),
        'MSE': mean_squared_error(all_labels, all_preds),
        'R2': r2_score(all_labels, all_preds),
        'MAE_slice': df_preds['error_slice'].mean()
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, f"metrics_{split}.csv"), index=False)

    # --- GENERAR SCATTER PLOT ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_preds, x='real', y='pred', hue='plane', alpha=0.5, palette='Set1')
    plt.plot([-1, 1], [-1, 1], color='black', linestyle='--', lw=1, label='Ideal')
    plt.axhline(0, color='gray', lw=1, alpha=0.5)
    plt.axvline(0, color='gray', lw=1, alpha=0.5)
    plt.xlabel('Posición Real', fontsize=12)
    plt.ylabel('Posición Predicha', fontsize=12)
    plt.title(f'Evaluación de Regresión ({split.upper()})\nMAE = {metrics["MAE"]:.4f} | R² = {metrics["R2"]:.4f}', fontsize=14, fontweight='bold')
    plt.legend(title='Plano', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(os.path.join(save_dir, f"scatter_{split}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"scatter_{split}.eps"), format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics, df_preds

if __name__ == '__main__':
    run_inference()
