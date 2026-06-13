from dataset import get_mri_loaders
from utils import evaluate_classification, visualize_predictions
from model import get_mri_model
import torch
import argparse
# Save advanced metrics to txt file
from datetime import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, average_precision_score

MODEL_NAME="mobilenet_v2" # Change this to match the trained model (resnet18, vgg16, mobilenet_v2)
NUM_TRAIN="4k"


PRETRAINED=True
NON_ROTATED=True
MS=True

PICASSO = False # Toggle this for cluster inference

if PICASSO:
    BASE_DIR = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/classifiers/classifier_sequence"
    DATA_DIR = os.path.join(BASE_DIR, f"data/{NUM_TRAIN}")
    MODEL_PATH = os.path.join(BASE_DIR, f"checkpoints/{MODEL_NAME}-{NUM_TRAIN}")
    REGRESSION_DATA_DIR = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/2D_MRI_ms_ep_control_FLAIR_T1"
    REGRESSION_PKL = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/classifiers/subjects_split.pkl"
    BASE_OUTPUT_DIR = BASE_DIR
else:
    DATA_DIR = f'D:/classifiers/classifier_sequence/data/{NUM_TRAIN}'
    MODEL_PATH = f'D:/classifiers/classifier_sequence/checkpoints/{MODEL_NAME}-{NUM_TRAIN}'
    REGRESSION_DATA_DIR = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
    REGRESSION_PKL = r"C:\Users\pedro\Proyectos\Medical_VLM\classifiers\subjects_split.pkl"
    BASE_OUTPUT_DIR = None

if NON_ROTATED:
    DATA_DIR += '-NR'

if PRETRAINED:
    MODEL_PATH+='-PT'
if NON_ROTATED:
    MODEL_PATH+='-NR'
if MS:
    MODEL_PATH+='_ms'   
MODEL_PATH+= '/best_mri_classifier.pth'
DEVICE = "cuda"

# --- CONFIGURATION FOR REGRESSION DATASET INFERENCE ---
USE_REGRESSION_DATASET = True

SUBSETS = {
    'ms': 'MRIms_kde',
    'e': 'MRIe_kde',
    'control': 'MRIcontrol_kde',
    'full': '', # Empty string to use base directory
    'medtrinityMS': '' # Use base directory for standard dataset
}

def run_test():
    parser = argparse.ArgumentParser(description="Inference for MRI Sequence Classifier")
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test', 'all'], default='test',
                        help="Split to evaluate: train, val, test, or all (default: test)")
    parser.add_argument("--subset", type=str, choices=['ms', 'e', 'control', 'full', 'medtrinityMS', 'all_subsets'], default='medtrinityMS',
                        help="Subset to evaluate: ms, e, control, full, medtrinityMS (everything together), or all_subsets (process each individually)")
    args = parser.parse_args()
  
    # Determine which subsets to run
    if not USE_REGRESSION_DATASET:
        subsets_to_run = ['medtrinityMS']
    elif args.subset == 'all_subsets':
        subsets_to_run = ['ms', 'e', 'control']
    else:
        subsets_to_run = [args.subset]

    for subset_key in subsets_to_run:
        all_to_test = (subset_key in ['e', 'control'])
        if USE_REGRESSION_DATASET:
            current_data_dir = os.path.join(REGRESSION_DATA_DIR, SUBSETS[subset_key])
            print(f"\n{'='*60}")
            print(f" PROCESSING SUBSET: {subset_key.upper()} ({current_data_dir})")
            if all_to_test:
                print(" [INFO] Merging all data into TEST split for this subset.")
            print(f"{'='*60}")
            loaders, sizes, classes = get_mri_loaders(current_data_dir, batch_size=32, no_augmentation=True, 
                                                    use_regression_split=True, pkl_path=REGRESSION_PKL,
                                                    all_to_test=all_to_test)
        else:
            print(f"\n{'='*60}")
            print(f" PROCESSING STANDARD DATASET ({DATA_DIR})")
            if all_to_test:
                print(" [INFO] Merging all data into TEST split for this subset.")
            print(f"{'='*60}")
            loaders, sizes, classes = get_mri_loaders(DATA_DIR, no_augmentation=True, all_to_test=all_to_test)
        
        all_splits = ['train', 'val', 'test']

        # Determine which splits to run
        if args.split == 'all':
            splits_to_run = all_splits
        else:
            if args.split in all_splits:
                splits_to_run = [args.split]
            else:
                print(f"Error: Split '{args.split}' is not available for subset '{subset_key}'. skipping.")
                continue

        # Load model and weights
        model = get_mri_model(model_name=MODEL_NAME, num_classes=len(classes))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()

        for split in splits_to_run:
            if split not in loaders:
                print(f"Skipping split {split.upper()} for subset {subset_key.upper()} because it is empty.")
                continue

            print(f"\nEvaluating the {split.upper()} split of subset {subset_key.upper()}...")
            
            # Update output dir to include subset
            if BASE_OUTPUT_DIR:
                subset_output_dir = os.path.join(BASE_OUTPUT_DIR, subset_key)
            else:
                subset_output_dir = os.path.join("classifiers/classifier_sequence", subset_key)

            cm,report,acc,image_save_path,report_save_path,all_labels,all_preds,misclassified = evaluate_classification(model=model, 
                                                dataloader=loaders[split],
                                                classes=classes, 
                                                model_name=MODEL_NAME,
                                                num_train=NUM_TRAIN,
                                                pretrained=PRETRAINED,
                                                device=DEVICE,
                                                non_rotated=NON_ROTATED,
                                                split=split,
                                                plot_cm=(split == 'test' or args.split != 'all'),
                                                base_output_dir=subset_output_dir)
            
            if split=="test" :
                print(f"Confusion matrix saved in {image_save_path}") 
                
            print(f"Classification report saved in {report_save_path}")
            print(f"Overall Accuracy for {subset_key.upper()} - {split.upper()}: {acc:.4f}")

            # Print misclassified examples
            if misclassified:
                print(f"\n--- Misclassified Examples ({split.upper()}) [Max 5] ---")
                for i, item in enumerate(misclassified[:5]):
                    print(f"{i+1}. File: {item['filename']} | Real: {item['real']} | Pred: {item['pred']}")
                print("-" * 40)

            # Save additional metrics
            lb = LabelBinarizer()
            lb.fit(range(len(classes)))
            y_true_bin = lb.transform(all_labels)
            y_pred_bin = lb.transform(all_preds)
            
            if len(classes) == 2:
                y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
                y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            pr_auc_micro = average_precision_score(y_true_bin, y_pred_bin, average="micro")

            log_dir = os.path.join(subset_output_dir, "figures", f"{MODEL_NAME}-{NUM_TRAIN}")
            if PRETRAINED: log_dir += '-PT'
            if NON_ROTATED: log_dir += '-NR'
                
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_dir, f'matrices_{split}_{timestamp}.txt')
            
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write("=============================================\n")
                f.write(f"    METRICS - SEQUENCE CLASSIFICATION ({MODEL_NAME}) - SUBSET: {subset_key.upper()} - SPLIT: {split.upper()}\n")
                f.write("=============================================\n\n")
                f.write(f"Classes: {classes}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(cm, separator=', '))
                f.write(f"\n\nOverall Accuracy: {acc:.4f}\n")
                f.write("AUC-ROC (Micro-Average): {:.4f}\n".format(roc_auc_micro))
                f.write("AUC-PR (Micro-Average): {:.4f}\n".format(pr_auc_micro))
                f.write("\n\nClassification Report:\n")
                f.write(str(report))
                
            print(f"Metrics saved to '{log_file_path}'")

if __name__ == '__main__':
    run_test()
