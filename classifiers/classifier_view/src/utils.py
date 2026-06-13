import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
import os

def plot_acc_curve(history,model_name):
    """
    Toma el diccionario de historial devuelto por el entrenamiento y genera un gráfico de Accuracy.
    """
    acc = history['train_acc']
    val_acc = history['val_acc']
    epochs_range = range(len(acc)+1)

    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))

    # 1. Accuracy Plot
    
    plt.plot(epochs_range, acc, label='Entrenamiento', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validación', linewidth=2,linestyle='--')
    plt.title(f'Evolución de Accuracy ({model_name}-{len(history["train_acc"])} épocas)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy [0-1]')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    save_dir=f"classifiers/classifier_view/figures/{model_name}"
    os.makedirs(save_dir,exist_ok=True)
    
    
    plt.savefig(f"{save_dir}/accuracy_curve.png", dpi=300) # dpi=300 for publication quality
    plt.show()

def plot_loss_curve(history,model_name):
    """
    Toma el diccionario de historial devuelto por el entrenamiento y genera un gráfico de la evolución de la Pérdida.
    """
    acc = history['train_loss']
    val_acc = history['val_loss']
    epochs_range = range(len(acc)+1)

    plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))

    
    plt.plot(epochs_range, acc, label='Entrenamiento', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validación', linewidth=2,linestyle='--')
    plt.title(f'Evolución de la Pérdida ({model_name}- {len(history["train_acc"])} épocas)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    save_dir=f"classifier/figures/{model_name}"
    os.makedirs(save_dir,exist_ok=True)
    
    
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300) # dpi=300 for publication quality
    plt.show()
    

def plot_learning_curves(history, model_name,num_train,pretrained:bool, non_rotated:bool=False, no_augmentation:bool=False):
    plt.figure(figsize=(14, 6))
    
    # Extract data for clarity
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    epochs_range = range(len(val_acc))

    # --- 1. Accuracy Plot ---
    ax1 = plt.subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    plt.plot(epochs_range, history['train_acc'], label='Entrenamiento', alpha=0.6)
    plt.plot(epochs_range, val_acc, label='Validación', linewidth=2)
    
    # Unfreezing line
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelación')
    
    # Find the maximum
    best_acc_idx = np.argmax(val_acc)
    best_acc_val = val_acc[best_acc_idx]
    
    # Mark the maximum point
    plt.scatter(best_acc_idx, best_acc_val, color='red', zorder=5)
    plt.annotate(f'Máx: {best_acc_val:.4f}', 
                 xy=(best_acc_idx, best_acc_val), 
                 xytext=(best_acc_idx, best_acc_val + 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 ha='center')

    title = f'Evolución de Accuracy ({model_name}'
    if no_augmentation: title += ' - Sin Aumento de Datos'
    title += ')'
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 2. Loss Plot ---
    ax2 = plt.subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    plt.plot(epochs_range, history['train_loss'], label='Entrenamiento', alpha=0.6)
    plt.plot(epochs_range, val_loss, label='Validación', linewidth=2)
    
    # Unfreezing line
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelación')
    
    # Find the minimum
    best_loss_idx = np.argmin(val_loss)
    best_loss_val = val_loss[best_loss_idx]
    
    # Mark the minimum point
    plt.scatter(best_loss_idx, best_loss_val, color='red', zorder=5)
    plt.annotate(f'Mín: {best_loss_val:.4f}', 
                 xy=(best_loss_idx, best_loss_val), 
                 xytext=(best_loss_idx, best_loss_val + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 ha='center')

    title = f'Evolución de la Pérdida ({model_name}'
    if no_augmentation: title += ' - Sin Aumento de Datos'
    title += ')'
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    save_dir = f"classifiers/classifier_view/figures/{model_name}-{num_train}"
    
    if pretrained:
        save_dir+="-PT" # Pretrained
    if non_rotated:
        save_dir+="-NR" # Non-Rotated
    if no_augmentation: # No Data Augmentation
        save_dir+="-NA"
    
   
    os.makedirs(save_dir, exist_ok=True)
    image_save_path=os.path.join(save_dir, "learning_curves.png")
    plt.savefig(image_save_path, dpi=300)
    
    plt.show()
    
    return image_save_path
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def evaluate_classification(model, dataloader, classes, model_name, num_train, pretrained:bool, device, non_rotated:bool=False,no_augmentation:bool=False, split:str='test', plot_cm:bool=True, base_output_dir:str=None):
    """
    Calcula y grafica una matriz de confusión para el modelo y los datos dados.
    """
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_filenames = []
    
    # Check if dataset has samples and paths
    has_paths = hasattr(dataloader.dataset, 'samples')

    # Get predictions
    with torch.no_grad():
        batch_start_idx = 0
        for inputs, labels in tqdm(dataloader, desc=f"Evaluando {split.upper()}", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

            # Track misclassified filenames
            if has_paths:
                for i in range(len(preds_np)):
                    if preds_np[i] != labels_np[i]:
                        sample_idx = batch_start_idx + i
                        if sample_idx < len(dataloader.dataset.samples):
                            path = dataloader.dataset.samples[sample_idx][0]
                            filename = os.path.basename(path)
                            misclassified_filenames.append({
                                'filename': filename,
                                'real': classes[labels_np[i]],
                                'pred': classes[preds_np[i]]
                            })
            
            batch_start_idx += len(inputs)
    
    # Confusion matrix calculation
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    
    # Save
    if base_output_dir:
        save_dir = os.path.join(base_output_dir, "figures", f"{model_name}-{num_train}")
    else:
        save_dir = f"classifiers/classifier_view/figures/{model_name}-{num_train}"
    
    if pretrained:
        save_dir+="-PT" # Pretrained
    if non_rotated:
        save_dir+="-NR" # Non-Rotated
    if no_augmentation: # No Data Augmentation
        save_dir+="-NA"
        
    os.makedirs(save_dir, exist_ok=True)
    
    image_save_path= os.path.join(save_dir, f"confusion_matrix_{split}.png")
    image_save_path_eps = os.path.join(save_dir, f"confusion_matrix_{split}.eps")

    if plot_cm:
        # 3. Visualization with Seaborn
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                         xticklabels=classes, yticklabels=classes,
                         annot_kws={"size": 19})
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=19)
        
        plt.title(f'Matriz de Confusión ({split.upper()}) - {model_name}', fontsize=19)
        plt.ylabel('Etiqueta Real', fontsize=19)
        plt.xlabel('Etiqueta Predicha', fontsize=19)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        
        plt.savefig(image_save_path, dpi=300)
        plt.savefig(image_save_path_eps, format='eps', dpi=300)
        
        if not base_output_dir:
            plt.show()
        plt.close()
    
    # Calculation of additional metrics
    report_dict = classification_report(all_labels, all_preds, labels=range(len(classes)), target_names=classes, output_dict=True)
    report = pd.DataFrame(report_dict).transpose()
    
    # Add per-class accuracy
    import numpy as np
    class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1)!=0)
    for i, cls in enumerate(classes):
        if cls in report.index:
            report.loc[cls, 'accuracy_per_class'] = class_acc[i]
    
    # Save Report
    if "figures" in save_dir:
        report_save_dir = save_dir.replace("figures", "reports")
    else:
        # Fallback if the path structure is different
        report_save_dir = os.path.join(os.path.dirname(save_dir), "reports", os.path.basename(save_dir))
        
    os.makedirs(report_save_dir, exist_ok=True)
    report_save_path = os.path.join(report_save_dir, f"classification_report_{split}.csv")
    report.to_csv(report_save_path)
    
    print("\n--- Informe de Clasificación ---")
    print(report)
    print(f"Informe de clasificación guardado en {report_save_path}")

    #  Get overall accuracy
    acc = accuracy_score(all_labels, all_preds)
   
    
    return cm, report, acc, image_save_path if plot_cm else None, report_save_path, all_labels, all_preds, misclassified_filenames
    
    
    
def visualize_predictions(model, dataloader, classes, device,model_name,num_train, pretrained:bool, non_rotated:bool=False, num_images=9):
    """
    Muestra una cuadrícula de imágenes con sus etiquetas predichas y reales.
    Las predicciones correctas son verdes, las incorrectas rojas.
    """
    model.eval()
    images_so_far = 0
    # Create a figure proportional to the number of images (e.g.: 3x3 for 9 images)
    cols = 3
    rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)
    fig = plt.figure(figsize=(15, 5 * rows))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(rows, cols, images_so_far)
                ax.axis('off')

                # --- DE-NORMALIZATION ---
                # Images come normalized for ResNet.
                # To view them correctly, we revert the Mean and Standard Deviation of ImageNet.
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1) # Ensure values are between 0 and 1

                # Title color: Green if correct, Red if incorrect
                is_correct = preds[j] == labels[j]
                title_color = 'green' if is_correct else 'red'
                
                ax.set_title(f'Real: {classes[labels[j]]}\nPred: {classes[preds[j]]}', 
                             color=title_color, fontsize=12, fontweight='bold')
                
                plt.imshow(img)

                if images_so_far == num_images:
                    model.train() # Return the model to training mode for safety
                    plt.tight_layout()
                    save_dir = f"classifiers/classifier_view/figures/{model_name}-{num_train}"
                    if pretrained:
                        save_dir += "-PT"
                    if non_rotated:
                        save_dir += "-NR"
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir,"samples_inference.png"))
                    plt.show()
                    return
                
                
def load_time_epochs(file_path:str):
    """
    Carga los datos del tiempo de entrenamiento desde un archivo .pth.
    """
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        return data
    except Exception as e:
        # English comment: Handle potential loading errors (e.g., file not found or corrupted)
        print(f"Error al cargar el archivo: {e}")
        return None

def plot_models_comparison(checkpoint_dirs, model_names, save_dir, pretrained:bool=True, non_rotated:bool=False, no_augmentation:bool=False):
    """
    Grafica una comparación de precisión y pérdida (validación) para múltiples modelos.
    """
    plt.figure(figsize=(16, 6))
    
    # Label suffix for title
    suffix = ""
    if no_augmentation: suffix += " - Sin Aumento de Datos"
    else: suffix += ""
    
    # 1. Accuracy Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(MultipleLocator(3))
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            val_acc = history['val_acc']
            epochs = range(len(val_acc))
            plt.plot(epochs, val_acc, label=f'{model_name}', linewidth=2)
    
    title_acc = 'Comparación de Accuracy en Validación'
    title_loss = 'Comparación de Pérdida en Validación'
    if suffix:
        title_acc += f' ({suffix.strip("- ").strip()})'
        title_loss += f' ({suffix.strip("- ").strip()})'
    
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelación')
    plt.title(title_acc, fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    # 2. Loss Plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            val_loss = history['val_loss']
            epochs = range(len(val_loss))
            plt.plot(epochs, val_loss, label=f'{model_name}', linewidth=2)
            
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelación')
    plt.title(title_loss, fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('Pérdida', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = "NA" if no_augmentation else "Aug"
    save_path = os.path.join(save_dir, f"models_comparison_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, f"models_comparison_{file_suffix}.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)
    plt.show()
    return save_path

def plot_models_comparison_full(checkpoint_dirs, model_names, save_dir, pretrained:bool=True, non_rotated:bool=False, no_augmentation:bool=False):
    """
    Grafica una comparación de métricas de entrenamiento Y validación (precisión y pérdida) para múltiples modelos.
    Cada modelo usa el mismo color: entrenamiento es sólido/oscuro, validación es punteado/claro.
    """
    plt.figure(figsize=(18, 8))
    
    # Label suffix for title
    suffix = ""
    if no_augmentation: suffix += " - Sin Aumento de Datos"
    else: suffix += ""
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(model_names))))
    
    def get_colors(m_name, idx):
        m_name_low = m_name.lower()
        if 'resnet18' in m_name_low:
            return 'darkblue', 'cornflowerblue'
        elif 'vgg16' in m_name_low:
            return 'darkorange', 'sandybrown'
        elif 'mobilenet' in m_name_low:
            return 'darkgreen', 'lightgreen'
        return colors[idx], colors[idx]
    
    # 1. Accuracy Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.xaxis.set_major_locator(MultipleLocator(3))
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    
    for i, (checkpoint_dir, model_name) in enumerate(zip(checkpoint_dirs, model_names)):
        history_path = os.path.join(checkpoint_dir, 'history.pth')
        if os.path.exists(history_path):
            history = torch.load(history_path, map_location='cpu', weights_only=True)
            epochs = range(len(history['val_acc']))
            train_color, val_color = get_colors(model_name, i)
            
            # Training Accuracy (Darker)
            plt.plot(epochs, history['train_acc'], label=f'{model_name} (Entr.)', 
                     color=train_color, linestyle='-', linewidth=2, alpha=1.0)
            # Validation Accuracy (Lighter, Dashed)
            plt.plot(epochs, history['val_acc'], label=f'{model_name} (Val)', 
                     color=val_color, linestyle='--', linewidth=2, alpha=1.0)
    
    plt.axvline(x=9, color='red', linestyle=':', alpha=0.8, label='Descongelación')
    
    title_acc = 'Comparación de Accuracy: Entrenamiento vs Validación'
    if suffix: title_acc += f' ({suffix.strip("- ").strip()})'
    
    plt.title(title_acc, fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    # plt.legend(fontsize=16, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 2. Loss Plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    
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
            plt.plot(epochs, history['val_loss'], label=f'{model_name} (Val)', 
                     color=val_color, linestyle='--', linewidth=2, alpha=1.0)
            
    plt.axvline(x=9, color='red', linestyle=':', alpha=0.8, label='Descongelación')
    
    title_loss = 'Comparación de Pérdida: Entrenamiento vs Validación'
    if suffix: title_loss += f' ({suffix.strip("- ").strip()})'
    
    plt.title(title_loss, fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=20)
    plt.ylabel('Pérdida', fontsize=20)
    # plt.legend(fontsize=16, ncol=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = "NA" if no_augmentation else "Aug"
    save_path = os.path.join(save_dir, f"models_comparison_full_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, f"models_comparison_full_{file_suffix}.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)
    
    # Extraer y guardar la leyenda por separado
    handles, labels = ax1.get_legend_handles_labels()
    fig_leg = plt.figure(figsize=(30, 5))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    legend = ax_leg.legend(handles, labels, loc='center', fontsize=40, ncol=4)
    fig_leg.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    
    legend_save_path = os.path.join(save_dir, f"models_comparison_full_legend_{file_suffix}.png")
    fig_leg.savefig(legend_save_path, dpi=300, bbox_inches=bbox.expanded(1.05, 1.05))
    legend_save_path_eps = os.path.join(save_dir, f"models_comparison_full_legend_{file_suffix}.eps")
    fig_leg.savefig(legend_save_path_eps, format='eps', dpi=300, bbox_inches=bbox.expanded(1.05, 1.05))
    plt.close(fig_leg)

    plt.show()
    return save_path

def plot_time_comparison(checkpoint_dirs, model_names, save_dir, pretrained:bool=True, non_rotated:bool=False, no_augmentation:bool=False):
    """
    Grafica una comparación de los tiempos de ejecución (tiempo total por época) para múltiples modelos.
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
    # Label suffix for title
    suffix = ""
    if no_augmentation: suffix += " - Sin Aumento de Datos"
    else: suffix += ""
    
    for checkpoint_dir, model_name in zip(checkpoint_dirs, model_names):
        times_path = os.path.join(checkpoint_dir, 'times_epochs.pth')
        if os.path.exists(times_path):
            data = load_time_epochs(times_path)
            if data:
                # Assuming epochs are integers and data is {epoch: {'total': ...}}
                epochs = sorted(data.keys())
                total_times = [data[e]['total'] for e in epochs]
                cumulative_time = sum(total_times)
                
                # Format time label (Total: XXm YYs)
                mins = int(cumulative_time // 60)
                secs = int(cumulative_time % 60)
                time_label = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                
                plt.plot(epochs, total_times, label=f'{model_name} (Total: {time_label})', marker='o', markersize=4)
                
    # Unfreezing line
    plt.axvline(x=9, color='red', linestyle='--', alpha=0.6, label='Descongelación')

    main_title = 'Comparación de Tiempo de Entrenamiento (Tiempo total por época)'
    if suffix:
        main_title += f' - {suffix.strip("- ").strip()}'
    plt.title(main_title, fontsize=20, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Tiempo (segundos)')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = "NA" if no_augmentation else "Aug"
    save_path = os.path.join(save_dir, f"time_comparison_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    save_path_eps = os.path.join(save_dir, f"time_comparison_{file_suffix}.eps")
    plt.savefig(save_path_eps, format='eps', dpi=300)
    plt.show()
    return save_path



