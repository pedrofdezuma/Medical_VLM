import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score, 
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# ================= CONFIGURATION =================
BASE_OUTPUT_DIR = "C:/Users/pedro/Desktop/Investigacion/classified_slices9"
# Asegúrate de que los modelos aquí coincidan con los que has ejecutado en el otro script
MODELS_TO_RUN = ["resnet18"] 
# =================================================

def plot_confusion_matrix(y_true, y_pred, title, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Vista Real')
    plt.xlabel('Vista Predicha')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm

def plot_and_metrics(y_true, y_pred, labels, task_name, out_dir):
    lb = LabelBinarizer()
    lb.fit(labels)
    y_true_bin = lb.transform(y_true)
    y_pred_bin = lb.transform(y_pred)
    
    n_classes = len(labels)
    
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    # --- ROC & AUC-ROC ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC {labels[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {task_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'roc_{task_name}.png'), bbox_inches='tight')
    plt.close()

    # --- PR & AUC-PR ---
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        pr_auc[i] = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])
        
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    pr_auc["micro"] = average_precision_score(y_true_bin, y_pred_bin, average="micro")
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall["micro"], precision["micro"],
             label=f'micro-average PR curve (AUC = {pr_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'PR {labels[i]} (AUC = {pr_auc[i]:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Exhaustividad)')
    plt.ylabel('Precision (Precisión)')
    plt.title(f'Curva Precision-Recall - {task_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f'pr_{task_name}.png'), bbox_inches='tight')
    plt.close()
    
    return roc_auc, pr_auc

def evaluate_model(model_name):
    csv_path = os.path.join(BASE_OUTPUT_DIR, model_name, "classification_results.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el CSV de resultados: {csv_path}")
        return

    print(f"\n{'='*50}")
    print(f" EVALUANDO MÉTRICAS PARA EL MODELO: {model_name} ")
    print(f"{'='*50}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Al leer el CSV: {e}")
        return
    
    # Validar las columnas
    if 'true_view' not in df.columns or 'predicted_view' not in df.columns:
        print("[ERROR] El CSV no contiene las columnas esperadas ('true_view', 'predicted_view')")
        return
        
    # Reemplazar posibles nulos y asegurar que sean strings
    df = df.fillna('unknown').astype(str)
    
    # Filtrar unknown si es necesario para métricas justas
    df_clean = df[df['true_view'] != 'unknown']
    unknown_views = len(df) - len(df_clean)
    
    print(f"Total de archivos en CSV: {len(df)}")
    print(f"Ignorados por vista 'unknown': {unknown_views}")
    print(f"Total a evaluar: {len(df_clean)}")
    
    if df_clean.empty:
         print("[WARNING] No hay datos válidos para evaluar.")
         return
         
    y_true = df_clean['true_view']
    y_pred = df_clean['predicted_view']
    
    print("\n--- Distribución de Vistas Reales ---")
    print(y_true.value_counts().to_string())
    print("\n--- Distribución de Vistas Predichas ---")
    print(y_pred.value_counts().to_string())
    
    # Cálculos de métricas
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Macro & Weighted metrics
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\nAccuracy (Exactitud estándar): {acc * 100:.2f}%")
    print(f"Balanced Accuracy (Exactitud balanceada): {bal_acc * 100:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    print(f"\nMacro Average    -> Precision: {prec_mac:.4f} | Recall: {rec_mac:.4f} | F1: {f1_mac:.4f}")
    print(f"Weighted Average -> Precision: {prec_wt:.4f} | Recall: {rec_wt:.4f} | F1: {f1_wt:.4f}\n")
    
    print("Reporte de Clasificación Detallado:")
    report_str = classification_report(y_true, y_pred, zero_division=0)
    print(report_str)
    
    # Matriz de Confusión y Curvas
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    out_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    cm_path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    
    cm = plot_confusion_matrix(y_true, y_pred, f"Matriz de Confusión - {model_name}", labels, cm_path)
    
    print("\n--- Accuracy por Clase ---")
    class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1)!=0)
    for label, acc in zip(labels, class_acc):
        print(f"  {label}: {acc * 100:.2f}%")
        
    roc_auc, pr_auc = plot_and_metrics(y_true, y_pred, labels, model_name, out_dir)
    
    print(f"Gráficos generados y guardados en: {out_dir}")
    
    # Guardar reporte en texto (Log estilo matrices)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_dir, f"metrics_report_{model_name}_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========================================\n")
        f.write(f"    METRICS - VIEW CLASSIFICATION ({model_name})\n")
        f.write("========================================\n\n")
        
        f.write(f"Total imágenes evaluadas: {len(df_clean)}\n")
        f.write(f"Accuracy (Exactitud estándar): {acc * 100:.2f}%\n")
        f.write(f"Balanced Accuracy: {bal_acc * 100:.2f}%\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
        
        f.write(f"Macro Average    -> Precision: {prec_mac:.4f} | Recall: {rec_mac:.4f} | F1: {f1_mac:.4f}\n")
        f.write(f"Weighted Average -> Precision: {prec_wt:.4f} | Recall: {rec_wt:.4f} | F1: {f1_wt:.4f}\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("Reporte de Clasificación Detallado:\n\n")
        f.write(report_str)
        f.write("\n" + "-" * 50 + "\n\n")
        
        f.write(f"Labels: {labels}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
        
        f.write("\n\nAccuracy por Clase:\n")
        class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1)!=0)
        for label, acc in zip(labels, class_acc):
            f.write(f"  {label}: {acc * 100:.2f}%\n")
            
        f.write("\nAUC-ROC (Micro-Average): {:.4f}\n".format(roc_auc["micro"]))
        f.write("AUC-PR (Micro-Average): {:.4f}\n".format(pr_auc["micro"]))

    print(f"Reporte de texto extendido guardado en: {report_path}")

def main():
    for model_name in MODELS_TO_RUN:
        evaluate_model(model_name)

if __name__ == "__main__":
    main()
