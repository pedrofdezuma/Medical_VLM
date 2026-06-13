import os
import torch
from utils import plot_models_comparison, plot_time_comparison, plot_models_comparison_full

def main():
    # Base directory for checkpoints
    base_dir = "D:/classifiers/regression_slice/checkpoints"
    
    models = [
        {"name": "ResNet18", "folder": "resnet18-98k"},
        {"name": "ResNet50", "folder": "resnet50-98k"},
        {"name": "EfficientNetV2-S", "folder": "efficientnet_v2_s-98k"}
    ]
    
    checkpoint_dirs = [os.path.join(base_dir, m["folder"]) for m in models]
    model_names = [m["name"] for m in models]
    
    # Directory to save comparison figures
    # Seguimos la estructura: classifiers/regression_slice/figures/comparison
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(project_root, "classifiers", "regression_slice", "figures", "comparison")
    
    print("Generating models performance comparison (Validation only)...")
    mae_loss_path = plot_models_comparison(checkpoint_dirs, model_names, save_dir)
    print(f"MAE and Loss comparison saved to: {mae_loss_path}")

    print("\nGenerating models performance comparison (Train vs Validation)...")
    full_path = plot_models_comparison_full(checkpoint_dirs, model_names, save_dir)
    print(f"Full comparison saved to: {full_path}")
    
    print("\nGenerating training time comparison...")
    time_path = plot_time_comparison(checkpoint_dirs, model_names, save_dir)
    print(f"Time comparison saved to: {time_path}")

if __name__ == "__main__":
    main()
