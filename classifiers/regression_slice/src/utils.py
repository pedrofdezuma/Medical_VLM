import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

def analyze_regression_errors(df_preds, save_dir):
    """
    Analyzes and plots regression errors grouped by Plane, Sequence, and Disease.
    """
    print("\n" + "="*40)
    print("   ANÁLISIS DE ERROR POR CATEGORÍAS")
    print("="*40)
    
    metrics_report = []

    # 1. Analizar por cada grupo
    group_map = {'plane': 'Plano', 'sequence': 'Secuencia', 'disease': 'Patología'}
    for group_col in ['plane', 'sequence', 'disease']:
        print(f"\n--- MAE por {group_col.upper()} ---")
        
        # Calcular MAE y MAE_slice (si existe) por cada valor único del grupo
        agg_dict = {'error': 'mean'}
        if 'error_slice' in df_preds.columns:
            agg_dict['error_slice'] = 'mean'
            
        group_stats = df_preds.groupby(group_col).agg(agg_dict).reset_index()
        
        if 'error_slice' in df_preds.columns:
            group_stats.columns = [group_col, 'MAE', 'MAE_slice']
        else:
            group_stats.columns = [group_col, 'MAE']
        
        # Ordenar por MAE para ver cuáles fallan más
        group_stats = group_stats.sort_values(by='MAE', ascending=False)
        
        for _, row in group_stats.iterrows():
            msg = f"  {row[group_col]:<12}: MAE={row['MAE']:.4f}"
            if 'MAE_slice' in group_stats.columns:
                msg += f" | MAE_slice={row['MAE_slice']:.2f}"
            print(msg)
            
            report_entry = {
                'category': group_col,
                'value': row[group_col],
                'MAE': row['MAE']
            }
            if 'MAE_slice' in group_stats.columns:
                report_entry['MAE_slice'] = row['MAE_slice']
            metrics_report.append(report_entry)

        # 2. Generar Boxplots de distribución de error
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_col, y='error', data=df_preds, palette='viridis', hue=group_col, legend=False)
        plt.title(f'Distribución de Error por {group_map[group_col]}')
        plt.ylabel('Error Absoluto')
        plt.xlabel(group_map[group_col])
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(save_dir, f'error_dist_{group_col}_regresion.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path.replace('.png', '.eps'), format='eps', dpi=300, bbox_inches='tight')
        plt.close()

    # Guardar reporte de métricas por grupo en CSV
    df_report = pd.DataFrame(metrics_report)
    df_report.to_csv(os.path.join(save_dir, 'grouped_mae_report.csv'), index=False)
    
    print("\n" + "="*40)
    print(f"Gráficas de distribución guardadas en: {save_dir}")

def plot_loss_curve(history, model_name, save_dir):
    """Generates an individual Loss plot."""
    plt.figure(figsize=(10, 6))
    epochs = range(len(history['train_loss']))
    plt.plot(epochs, history['train_loss'], label='Entrenamiento', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validación', linewidth=3)
    
    if len(epochs) > 9:
        plt.axvline(x=9, color='red', linestyle=':', label='Descongelado (Unfreeze)', alpha=0.8)

    plt.title(f'Evolución de la Pérdida ({model_name})', fontsize=18, fontweight='bold')
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Pérdida (MSE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve_regresion.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'loss_curve_regresion.eps'), format='eps', dpi=300)
    plt.close()

def plot_mae_curve(history, model_name, save_dir):
    """Generates an individual MAE plot."""
    if 'train_mae' not in history: return
    plt.figure(figsize=(10, 6))
    epochs = range(len(history['train_mae']))
    plt.plot(epochs, history['train_mae'], label='Entrenamiento', linewidth=2)
    plt.plot(epochs, history['val_mae'], label='Validación', linewidth=3)
    
    if len(epochs) > 9:
        plt.axvline(x=9, color='red', linestyle=':', label='Descongelado (Unfreeze)', alpha=0.8)

    plt.title(f'Evolución del MAE ({model_name})', fontsize=18, fontweight='bold')
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mae_curve_regresion.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'mae_curve_regresion.eps'), format='eps', dpi=300)
    plt.close()

def plot_lr_curve(history, model_name, save_dir):
    """Generates a Learning Rate evolution plot."""
    if 'lr' not in history or not history['lr']: return
    plt.figure(figsize=(10, 6))
    epochs = range(len(history['lr']))
    plt.plot(epochs, history['lr'], color='orange', linewidth=2, marker='o', markersize=4)
    
    plt.title(f'Evolución de la Tasa de Aprendizaje ({model_name})', fontsize=18, fontweight='bold')
    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Learning Rate (Log)', fontsize=14)
    plt.yscale('log') # Usually LR changes in orders of magnitude
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_curve_regresion.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'lr_curve_regresion.eps'), format='eps', dpi=300)
    plt.close()

def plot_learning_curves(history, model_name, train_k, save_dir):
    """
    Plots training and validation loss/MAE over epochs with professional formatting.
    Generates the combined file and also individual ones.
    """
    # 1. Generar individuales primero
    plot_loss_curve(history, model_name, save_dir)
    plot_mae_curve(history, model_name, save_dir)
    plot_lr_curve(history, model_name, save_dir)

    # 2. Generar el combinado
    plt.figure(figsize=(18, 8))
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs_range = range(len(train_loss))

    # --- Subplot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Entrenamiento', alpha=0.6, linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validación', linewidth=3)
    
    if len(epochs_range) > 9:
        plt.axvline(x=9, color='red', linestyle='--', label='Descongelado', alpha=0.8)

    # Mark the minimum Loss
    best_loss_idx = np.argmin(val_loss)
    best_loss_val = val_loss[best_loss_idx]
    plt.scatter(best_loss_idx, best_loss_val, color='red', zorder=5)
    plt.annotate(f'Mín: {best_loss_val:.4f}', 
                 xy=(best_loss_idx, best_loss_val), 
                 xytext=(best_loss_idx, best_loss_val + (max(val_loss)*0.05)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 ha='center', fontsize=14, fontweight='bold')

    plt.title(f'Evolución de la Pérdida ({model_name})', fontsize=20, fontweight='bold')
    plt.xlabel("Época", fontsize=18)
    plt.ylabel("Pérdida (MSE)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # --- Subplot 2: MAE ---
    plt.subplot(1, 2, 2)
    if 'train_mae' in history and 'val_mae' in history:
        plt.plot(epochs_range, history['train_mae'], label='Entrenamiento', alpha=0.6, linewidth=2)
        plt.plot(epochs_range, history['val_mae'], label='Validación', linewidth=3)
        
        if len(epochs_range) > 9:
            plt.axvline(x=9, color='red', linestyle='--', label='Descongelado', alpha=0.8)

        # Mark the minimum MAE
        best_mae_idx = np.argmin(history['val_mae'])
        best_mae_val = history['val_mae'][best_mae_idx]
        plt.scatter(best_mae_idx, best_mae_val, color='red', zorder=5)
        plt.annotate(f'Mín: {best_mae_val:.4f}', 
                     xy=(best_mae_idx, best_mae_val), 
                     xytext=(best_mae_idx, best_mae_val + (max(history['val_mae'])*0.05)),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     ha='center', fontsize=14, fontweight='bold')

        plt.title(f'Evolución del MAE ({model_name})', fontsize=20, fontweight='bold')
        plt.xlabel("Época", fontsize=18)
        plt.ylabel("MAE", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curves_regresion.png")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace('.png', '.eps'), format='eps', dpi=300)
    plt.close()
    print(f"Todas las gráficas de entrenamiento se han guardado en: {save_dir}")


def evaluate_regression(model, dataloader, device, save_dir, split='test'):
    """
    Evaluates the regression model and returns metrics and predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []

    # Accessing internal samples if possible for paths
    samples = dataloader.dataset.samples

    print(f"Evaluando conjunto de {split.upper()}...")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Eval {split.upper()}", unit="batch")
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Get paths for this batch
            batch_size = inputs.size(0)
            start_idx = i * dataloader.batch_size
            end_idx = start_idx + batch_size
            all_paths.extend([s[0] for s in samples[start_idx:end_idx]])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    # Save metrics to CSV
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, f"metrics_{split}.csv"), index=False)

    # Save detailed predictions to CSV
    filenames = [os.path.basename(p) for p in all_paths]
    
    planes = []
    sequences = []
    diseases = []
    
    for p in all_paths:
        # Normalizamos la ruta para buscar componentes específicos
        p_norm = p.lower().replace('\\', '/')
        filename = os.path.basename(p_norm)
        
        # Plane detection
        if '/axial/' in p_norm or 'axial' in filename: planes.append('axial')
        elif '/sagittal/' in p_norm or 'sagittal' in filename: planes.append('sagittal')
        elif '/coronal/' in p_norm or 'coronal' in filename: planes.append('coronal')
        else: planes.append('unknown')
        
        # Sequence detection (Buscamos como carpeta o en el nombre del archivo)
        if '/flair/' in p_norm or 'flair' in filename:
            sequences.append('flair')
        elif '/t1' in p_norm or 't1' in filename:
            sequences.append('t1w')
        else:
            sequences.append('unknown')
        
        # Disease detection (Buscamos patrones específicos de tus carpetas)
        if 'mricontrol' in p_norm: diseases.append('Control')
        elif 'mrims' in p_norm: diseases.append('EM')
        elif 'mrie' in p_norm: diseases.append('Epilepsia')
        else: diseases.append('Unknown')

    df_preds = pd.DataFrame({
        'filename': filenames,
        'plane': planes,
        'sequence': sequences,
        'disease': diseases,
        'real': all_labels,
        'pred': all_preds,
        'error': np.abs(all_labels - all_preds)
    })
    df_preds.to_csv(os.path.join(save_dir, f"predictions_{split}.csv"), index=False)

    # Scatter Plot colored by Plane
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_preds, x='real', y='pred', hue='plane', alpha=0.5, palette='Set1')
    
    # Ideal line
    plt.plot([-1, 1], [-1, 1], color='black', linestyle='--', lw=1, label='Ideal (Error 0)', alpha=0.7)
    
    plt.axhline(0, color='gray', lw=1, alpha=0.5)
    plt.axvline(0, color='gray', lw=1, alpha=0.5)
    
    plt.xlabel('Posición Real', fontsize=18)
    plt.ylabel('Posición Predicha', fontsize=18)
    
    split_es = {'train': 'ENTRENAMIENTO', 'val': 'VALIDACIÓN', 'test': 'PRUEBA'}.get(split.lower(), split.upper())
    plt.title(f'Evaluación de Regresión ({split_es})\nMAE = {mae:.4f} | R² = {r2:.4f}', fontsize=20, fontweight='bold', pad=15)
    plt.legend(title='Plano', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Adjust axes to see both ranges clearly
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    plt.savefig(os.path.join(save_dir, f"scatter_{split}_regresion.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"scatter_{split}_regresion.eps"), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    return metrics, df_preds

from matplotlib.ticker import MultipleLocator

def load_time_epochs(file_path:str):
    """
    Loads the training time data from a .pth file.
    """
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        return data
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def plot_models_comparison(checkpoint_dirs, model_names, save_dir):
    """
    Plots a comparison of MAE and loss (validation) for multiple regression models.
    """
    plt.figure(figsize=(16, 6))
    
    # 1. Loss Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(MultipleLocator(3))
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            val_loss = history['val_loss']
            epochs = range(len(val_loss))
            plt.plot(epochs, val_loss, label=f'{model_name}', linewidth=2)
            
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelado')
    plt.title('Comparativa Pérdida (Validación)', fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('Pérdida', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # 2. MAE Plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            if 'val_mae' in history:
                val_mae = history['val_mae']
                epochs = range(len(val_mae))
                plt.plot(epochs, val_mae, label=f'{model_name}', linewidth=2)
    
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelado')
    plt.title('Comparativa MAE (Validación)', fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "models_comparison_regresion.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, "models_comparison_regresion.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)
    plt.close()
    return save_path

    plt.close()
    return save_path

def plot_models_comparison_full(checkpoint_dirs, model_names, save_dir):
    """
    Plots a comparison of training AND validation metrics (MAE and loss) for multiple regression models.
    Each model uses the same color: training is solid/dark, validation is dashed/light.
    """
    plt.figure(figsize=(24, 10))
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(model_names))))
    
    def get_colors(m_name, idx):
        m_name_low = m_name.lower()
        if 'resnet18' in m_name_low:
            return 'darkblue', 'cornflowerblue'
        elif 'resnet50' in m_name_low:
            return 'darkorange', 'sandybrown'
        elif 'efficientnet' in m_name_low:
            return 'darkgreen', 'lightgreen'
        return colors[idx], colors[idx]
    
    # 1. Loss Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(MultipleLocator(3))
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    
    for i, (checkpoint_dir, model_name) in enumerate(zip(checkpoint_dirs, model_names)):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            epochs = range(len(history['val_loss']))
            train_color, val_color = get_colors(model_name, i)
            
            # Training Loss (Darker)
            plt.plot(epochs, history['train_loss'], label=f'{model_name} (Entr.)', 
                     color=train_color, linestyle='-', linewidth=2, alpha=1.0)
            # Validation Loss (Lighter, Dashed)
            plt.plot(epochs, history['val_loss'], label=f'{model_name} (Val.)', 
                     color=val_color, linestyle='--', linewidth=2, alpha=1.0)

    plt.axvline(x=9, color='red', linestyle=':', alpha=0.8, label='Descongelamiento')
    plt.title('Comparativa Pérdida: Entrenamiento vs Validación', fontsize=28, fontweight='bold', pad=15)

    plt.xlabel('Época', fontsize=24)
    plt.ylabel('Pérdida (MSE)', fontsize=24)
    plt.grid(True, alpha=0.3)
    
    # 2. MAE Plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    
    for i, (checkpoint_dir, model_name) in enumerate(zip(checkpoint_dirs, model_names)):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            if 'val_mae' in history:
                epochs = range(len(history['val_mae']))
                train_color, val_color = get_colors(model_name, i)
                
                # Training MAE (Darker)
                plt.plot(epochs, history['train_mae'], label=f'{model_name} (Entr.)', 
                         color=train_color, linestyle='-', linewidth=2, alpha=1.0)
                # Validation MAE (Lighter, Dashed)
                plt.plot(epochs, history['val_mae'], label=f'{model_name} (Val.)', 
                         color=val_color, linestyle='--', linewidth=2, alpha=1.0)
    
    plt.axvline(x=9, color='red', linestyle=':', alpha=0.8, label='Descongelamiento')
    plt.title('Comparativa MAE: Entrenamiento vs Validación', fontsize=28, fontweight='bold', pad=15)

    plt.xlabel('Época', fontsize=24)
    plt.ylabel('MAE', fontsize=24)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "models_comparison_full_regresion.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, "models_comparison_full_regresion.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)

    
    # Extraer y guardar la leyenda por separado
    handles, labels = ax1.get_legend_handles_labels()
    
    # Reordenar handles y labels para forzar el layout en columnas
    # Formato original: [A_train, A_val, B_train, B_val, C_train, C_val, Descongelamiento]
    # Queremos 2 filas. Col 1: A, Col 2: B, Col 3: C, Col 4: Descongelamiento
    # Orden necesario para plt.legend (que llena fila a fila):
    # [A_train, B_train, C_train, Descongelamiento, A_val, B_val, C_val, (vacío si es necesario)]
    
    num_models = len(model_names)
    train_handles = []
    val_handles = []
    desc_handle = None
    
    train_labels = []
    val_labels = []
    desc_label = None

    for h, l in zip(handles, labels):
        if 'Entr.' in l:
            train_handles.append(h)
            train_labels.append(l)
        elif 'Val.' in l:
            val_handles.append(h)
            val_labels.append(l)
        elif 'Descongelamiento' in l:
            desc_handle = h
            desc_label = l

    ordered_handles = train_handles + [desc_handle] + val_handles
    ordered_labels = train_labels + [desc_label] + val_labels
    
    # Añadir un espacio vacío al final para rellenar la cuadrícula si es impar
    if len(ordered_handles) % 2 != 0:
        import matplotlib.patches as mpatches
        ordered_handles.append(mpatches.Patch(color='none'))
        ordered_labels.append('')

    fig_leg = plt.figure(figsize=(30, 5))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    
    # ncol = número de modelos + 1 (para la columna de Descongelamiento)
    legend = ax_leg.legend(ordered_handles, ordered_labels, loc='center', fontsize=30, ncol=num_models + 1)
    
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    
    legend_save_path = os.path.join(save_dir, "models_comparison_full_legend_regresion.png")
    fig_leg.savefig(legend_save_path, dpi=300, bbox_inches=bbox.expanded(1.05, 1.05))
    legend_save_path_eps = os.path.join(save_dir, "models_comparison_full_legend_regresion.eps")
    fig_leg.savefig(legend_save_path_eps, format='eps', dpi=300, bbox_inches=bbox.expanded(1.05, 1.05))
    plt.close(fig_leg)
    plt.close()
    return save_path

def plot_time_comparison(checkpoint_dirs, model_names, save_dir):
    """
    Plots a comparison of execution times (total time per epoch) for multiple regression models.
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        times_path = os.path.join(checkpoint_dir, 'times_epochs.pth')
        if os.path.exists(times_path):
            data = load_time_epochs(times_path)
            if data:
                epochs = sorted(data.keys())
                total_times = [data[e]['total'] for e in epochs]
                cumulative_time = sum(total_times)
                
                mins = int(cumulative_time // 60)
                secs = int(cumulative_time % 60)
                time_label = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                
                plt.plot(epochs, total_times, label=f'{model_name} (Total: {time_label})', marker='o', markersize=4)
                
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelado')

    plt.title('Comparativa de Tiempo de Entrenamiento (por Época)', fontsize=20, fontweight='bold')
    plt.xlabel('Época', fontsize=18)
    plt.ylabel('Tiempo (segundos)', fontsize=18)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "time_comparison_regresion.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, "time_comparison_regresion.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)
    plt.close()
    return save_path

def visualize_regression_predictions(model, dataloader, device, save_dir, num_images=12):
    """
    Visualizes regression results with images.
    """
    model.eval()
    images_so_far = 0
    cols = 4
    rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)
    fig = plt.figure(figsize=(18, 4 * rows))

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Visualizando", unit="batch", leave=False)
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(rows, cols, images_so_far)
                ax.axis('off')

                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                error = abs(preds[j] - labels[j])
                # Color based on error (normalized 0 to 2, but usually error is small)
                # Let's say error > 0.1 is red-ish
                color = 'green' if error < 0.05 else ('orange' if error < 0.15 else 'red')
                
                ax.set_title(f'Real: {labels[j]:.3f}\nPred: {preds[j]:.3f}\nError: {error:.3f}', 
                             color=color, fontsize=10, fontweight='bold')
                
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, "samples_inference_regresion.png"), dpi=300)
                    plt.close()
                    return
