import os
import torch
from utils import plot_models_comparison, plot_time_comparison, plot_models_comparison_full

def main():
    # Base directory for checkpoints
    base_dir = "D:/classifiers/classifier_view/checkpoints"
    
    # Model configurations
    PRETRAINED = True
    NON_ROTATED = True
    NO_AUGMENTATION = False
    MS=True
    
    models = [
        {"name": "ResNet18", "folder": "resnet18-4k-PT-EN"},
        {"name": "VGG16", "folder": "vgg16-4k-PT-EN"},
        {"name": "MobileNetV2", "folder": "mobilenet_v2-4k-PT-EN"}
    ]
    
    checkpoint_dirs = [os.path.join(base_dir, (m["folder"]+"_ms" if MS else m["folder"])) for m in models]
    model_names = [m["name"] for m in models]
    
    # Directory to save comparison figures
    save_dir = "classifiers/classifier_view/figures/comparison"
    
    print("Generando comparación de rendimiento de modelos (solo Validación)...")
    acc_loss_path = plot_models_comparison(checkpoint_dirs, model_names, save_dir, pretrained=PRETRAINED, non_rotated=NON_ROTATED, no_augmentation=NO_AUGMENTATION)
    print(f"Comparación de Precisión y Pérdida guardada en: {acc_loss_path}")

    print("\nGenerando comparación de rendimiento de modelos (Entrenamiento vs Validación)...")
    full_path = plot_models_comparison_full(checkpoint_dirs, model_names, save_dir, pretrained=PRETRAINED, non_rotated=NON_ROTATED, no_augmentation=NO_AUGMENTATION)
    print(f"Comparación completa guardada en: {full_path}")
    
    print("\nGenerando comparación de tiempo de entrenamiento...")
    time_path = plot_time_comparison(checkpoint_dirs, model_names, save_dir, pretrained=PRETRAINED, non_rotated=NON_ROTATED, no_augmentation=NO_AUGMENTATION)
    print(f"Comparación de tiempo guardada en: {time_path}")

if __name__ == "__main__":
    main()
