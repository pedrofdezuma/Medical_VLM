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

# Configuration
CSV_PATH = "D:/combined-test-4K-PT-NR-2/evaluation_results.csv"

def plot_confusion_matrix(y_true, y_pred, title, labels):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_and_metrics(y_true, y_pred, labels, task_name, log_dir):
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {task_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(log_dir, f'roc_{task_name}.png'), bbox_inches='tight')
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
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {task_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(log_dir, f'pr_{task_name}.png'), bbox_inches='tight')
    plt.close()
    
    return roc_auc, pr_auc

def analyze_advanced_results():
    try:
        df = pd.read_csv(CSV_PATH)
        
        # Replace any empty cells (NaN) with 'unknown' and force string type
        # This prevents the float vs str sorting error in sklearn
        df = df.fillna('unknown').astype(str)
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find the file at {CSV_PATH}")
        return

    print("="*50)
    print(" ADVANCED DATASET METRICS OVERVIEW ")
    print("="*50)
    print(f"Total files evaluated: {len(df)}")
    
    unknown_views = len(df[df['real_view'] == 'unknown'])
    unknown_seqs = len(df[df['real_sequence'] == 'unknown'])
    print(f"Ignored due to unknown real view: {unknown_views}")
    print(f"Ignored due to unknown real sequence: {unknown_seqs}")

    log_dir = 'logs_matrices'
    os.makedirs(log_dir, exist_ok=True)
    
    view_labels_out, view_cm, view_roc_auc, view_pr_auc = None, None, None, None
    seq_labels_out, seq_cm, seq_roc_auc, seq_pr_auc = None, None, None, None

    # -----------------------------------------
    # 0. DATA DISTRIBUTION (COUNTS)
    # -----------------------------------------
    print("\n" + "="*50)
    print(" 0. DATA DISTRIBUTION (COUNTS) ")
    print("="*50)

    # View counts
    print("\n--- Real Views Distribution ---")
    print(df['real_view'].value_counts().to_string())
    print("\n--- Predicted Views Distribution ---")
    print(df['predicted_view'].value_counts().to_string())

    # Sequence counts
    print("\n--- Real Sequences Distribution ---")
    print(df['real_sequence'].value_counts().to_string())
    print("\n--- Predicted Sequences Distribution ---")
    print(df['predicted_sequence'].value_counts().to_string())

    # View-Sequence Pair counts
    print("\n--- Real View-Sequence Pairs Distribution ---")
    real_pairs = df.groupby(['real_view', 'real_sequence']).size().reset_index(name='count')
    real_pairs = real_pairs.sort_values(by='count', ascending=False)
    print(real_pairs.to_string(index=False))

    print("\n--- Predicted View-Sequence Pairs Distribution ---")
    pred_pairs = df.groupby(['predicted_view', 'predicted_sequence']).size().reset_index(name='count')
    pred_pairs = pred_pairs.sort_values(by='count', ascending=False)
    print(pred_pairs.to_string(index=False))

    # -----------------------------------------
    # 1. VIEW CLASSIFIER ADVANCED METRICS
    # -----------------------------------------
    print("\n" + "="*50)
    print(" 1. VIEW CLASSIFIER PERFORMANCE ")
    print("="*50)
    
    # Filter out 'unknown' for fair metric calculation
    df_views = df[df['real_view'] != 'unknown']
    
    if not df_views.empty:
        y_true_v = df_views['real_view']
        y_pred_v = df_views['predicted_view']
        
        # Standard and Balanced Accuracy
        print(f"Standard Accuracy: {accuracy_score(y_true_v, y_pred_v) * 100:.2f}%")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true_v, y_pred_v) * 100:.2f}%")
        
        # Cohen's Kappa
        kappa_v = cohen_kappa_score(y_true_v, y_pred_v)
        print(f"Cohen's Kappa:     {kappa_v:.4f} (1.0 is perfect agreement)")
        
        # Macro & Weighted metrics
        prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true_v, y_pred_v, average='macro', zero_division=0)
        prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_true_v, y_pred_v, average='weighted', zero_division=0)
        
        print(f"\nMacro Average    -> Precision: {prec_mac:.4f} | Recall: {rec_mac:.4f} | F1: {f1_mac:.4f}")
        print(f"Weighted Average -> Precision: {prec_wt:.4f} | Recall: {rec_wt:.4f} | F1: {f1_wt:.4f}\n")
        
        print("Detailed Classification Report:")
        print(classification_report(y_true_v, y_pred_v, zero_division=0))
        
        view_labels_out = sorted(list(set(y_true_v.unique()) | set(y_pred_v.unique())))
        view_cm = confusion_matrix(y_true_v, y_pred_v, labels=view_labels_out)
        
        print("\nPer-class Accuracy (View):")
        view_class_acc = np.divide(view_cm.diagonal(), view_cm.sum(axis=1), out=np.zeros_like(view_cm.diagonal(), dtype=float), where=view_cm.sum(axis=1)!=0)
        for label, acc in zip(view_labels_out, view_class_acc):
            print(f"  {label}: {acc * 100:.2f}%")
        plot_confusion_matrix(y_true_v, y_pred_v, "Advanced View Confusion Matrix", view_labels_out)
        view_roc_auc, view_pr_auc = plot_and_metrics(y_true_v, y_pred_v, view_labels_out, 'view', log_dir)

    # -----------------------------------------
    # 2. SEQUENCE CLASSIFIER ADVANCED METRICS
    # -----------------------------------------
    print("\n" + "="*50)
    print(" 2. SEQUENCE CLASSIFIER PERFORMANCE ")
    print("="*50)
    
    # Filter out 'unknown'
    df_seqs = df[df['real_sequence'] != 'unknown']
    
    if not df_seqs.empty:
        y_true_s = df_seqs['real_sequence']
        y_pred_s = df_seqs['predicted_sequence']
        
        # Standard and Balanced Accuracy
        print(f"Standard Accuracy: {accuracy_score(y_true_s, y_pred_s) * 100:.2f}%")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true_s, y_pred_s) * 100:.2f}%")
        
        # Cohen's Kappa
        kappa_s = cohen_kappa_score(y_true_s, y_pred_s)
        print(f"Cohen's Kappa:     {kappa_s:.4f} (1.0 is perfect agreement)")
        
        # Macro & Weighted metrics
        prec_mac_s, rec_mac_s, f1_mac_s, _ = precision_recall_fscore_support(y_true_s, y_pred_s, average='macro', zero_division=0)
        prec_wt_s, rec_wt_s, f1_wt_s, _ = precision_recall_fscore_support(y_true_s, y_pred_s, average='weighted', zero_division=0)
        
        print(f"\nMacro Average    -> Precision: {prec_mac_s:.4f} | Recall: {rec_mac_s:.4f} | F1: {f1_mac_s:.4f}")
        print(f"Weighted Average -> Precision: {prec_wt_s:.4f} | Recall: {rec_wt_s:.4f} | F1: {f1_wt_s:.4f}\n")
        
        print("Detailed Classification Report:")
        print(classification_report(y_true_s, y_pred_s, zero_division=0))
        
        seq_labels_out = sorted(list(set(y_true_s.unique()) | set(y_pred_s.unique())))
        seq_cm = confusion_matrix(y_true_s, y_pred_s, labels=seq_labels_out)

        print("\nPer-class Accuracy (Sequence):")
        seq_class_acc = np.divide(seq_cm.diagonal(), seq_cm.sum(axis=1), out=np.zeros_like(seq_cm.diagonal(), dtype=float), where=seq_cm.sum(axis=1)!=0)
        for label, acc in zip(seq_labels_out, seq_class_acc):
            print(f"  {label}: {acc * 100:.2f}%")
        plot_confusion_matrix(y_true_s, y_pred_s, "Advanced Sequence Confusion Matrix", seq_labels_out)
        seq_roc_auc, seq_pr_auc = plot_and_metrics(y_true_s, y_pred_s, seq_labels_out, 'sequence', log_dir)

    # -----------------------------------------
    # 3. JOINT METRICS & ERROR CORRELATION
    # -----------------------------------------
    print("\n" + "="*50)
    print(" 3. JOINT PERFORMANCE & ERROR ANALYSIS ")
    print("="*50)
    
    # Analyze only fully known images
    df_joint = df[(df['real_view'] != 'unknown') & (df['real_sequence'] != 'unknown')].copy()
    
    if not df_joint.empty:
        # Define boolean columns for correctness
        df_joint['view_correct'] = df_joint['real_view'] == df_joint['predicted_view']
        df_joint['seq_correct'] = df_joint['real_sequence'] == df_joint['predicted_sequence']
        df_joint['both_correct'] = df_joint['view_correct'] & df_joint['seq_correct']
        
        joint_acc = df_joint['both_correct'].mean()
        print(f"Joint Accuracy (Both View & Seq correct): {joint_acc * 100:.2f}%")
        print(f"Evaluated on {len(df_joint)} fully known images.\n")
        
        # Error Correlation: Where is the sequence model failing?
        print("--- Sequence Error Distribution by Real View ---")
        seq_errors = df_joint[~df_joint['seq_correct']]
        if not seq_errors.empty:
            error_crosstab = pd.crosstab(seq_errors['real_view'], seq_errors['real_sequence'], margins=True)
            print("This table shows the counts of SEQUENCE MISCLASSIFICATIONS broken down by the REAL VIEW:")
            print(error_crosstab)
        else:
            print("No sequence errors found!")

    # -----------------------------------------
    # 4. SAVE TEXT METRICS TO LOG
    # -----------------------------------------
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f'matrices_{timestamp}.txt'
    log_file_path = os.path.join(log_dir, log_file_name)
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            if view_cm is not None:
                f.write("========================================\n")
                f.write("    METRICS - VIEW CLASSIFICATION\n")
                f.write("========================================\n\n")
                f.write(f"Labels: {view_labels_out}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(view_cm, separator=', '))
                
                f.write("\n\nPer-class Accuracy:\n")
                view_class_acc = np.divide(view_cm.diagonal(), view_cm.sum(axis=1), out=np.zeros_like(view_cm.diagonal(), dtype=float), where=view_cm.sum(axis=1)!=0)
                for label, acc in zip(view_labels_out, view_class_acc):
                    f.write(f"  {label}: {acc * 100:.2f}%\n")
                
                f.write("\nAUC-ROC (Micro-Average): {:.4f}\n".format(view_roc_auc["micro"]))
                f.write("AUC-PR (Micro-Average): {:.4f}\n".format(view_pr_auc["micro"]))
                f.write("\n\n\n----------------------------------------------------\n\n\n")
            
            if seq_cm is not None:
                f.write("=============================================\n")
                f.write("    METRICS - SEQUENCE CLASSIFICATION\n")
                f.write("=============================================\n\n")
                f.write(f"Labels: {seq_labels_out}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(seq_cm, separator=', '))
                
                f.write("\n\nPer-class Accuracy:\n")
                seq_class_acc = np.divide(seq_cm.diagonal(), seq_cm.sum(axis=1), out=np.zeros_like(seq_cm.diagonal(), dtype=float), where=seq_cm.sum(axis=1)!=0)
                for label, acc in zip(seq_labels_out, seq_class_acc):
                    f.write(f"  {label}: {acc * 100:.2f}%\n")
                
                f.write("\nAUC-ROC (Micro-Average): {:.4f}\n".format(seq_roc_auc["micro"]))
                f.write("AUC-PR (Micro-Average): {:.4f}\n".format(seq_pr_auc["micro"]))
        print(f"\nSuccessfully saved text metrics to '{log_file_path}'")
    except Exception as e:
        print(f"[ERROR] Failed to write text metrics log: {e}")

if __name__ == "__main__":
    analyze_advanced_results()