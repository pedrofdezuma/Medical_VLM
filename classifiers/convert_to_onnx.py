import torch
import torch.nn as nn
from torchvision import models
import argparse
import os

def get_mri_model(model_name: str, num_classes: int, task: str = "classifier"):
    """
    Recreates the model architecture as defined in the project's model.py files.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Model {model_name} not supported")

    if task == "regression":
        if model_name in ["resnet18", "resnet50"]:
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1),
                nn.Tanh()
            )
        elif model_name == "vgg16":
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, 1),
                nn.Tanh()
            )
        elif model_name == "mobilenet_v2":
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, 1),
                nn.Tanh()
            )
    else: # classifier
        if model_name in ["resnet18", "resnet50"]:
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == "vgg16":
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name == "mobilenet_v2":
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, num_classes)
            )
    
    return model

def convert_to_onnx(model_name, num_classes, checkpoint_path, output_path, task="classifier"):
    # 1. Instantiate the model
    print(f"Creating model: {model_name} for {task} task...")
    model = get_mri_model(model_name, num_classes, task)
    
    # 2. Load the weights
    print(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # In some cases, the state_dict might be wrapped in another dict (e.g., if saved with optimizer)
    # The project's 'best_mri_classifier.pth' seems to be just the state_dict based on common patterns,
    # but let's be safe.
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Create dummy input
    # Project uses 224x224 images
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch MRI Classifier checkpoints to ONNX.")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16", "mobilenet_v2"], help="Model architecture")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (4 for view, 5 for sequence, not used for regression)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path for the output .onnx file")
    parser.add_argument("--task", type=str, default="classifier", choices=["classifier", "regression"], help="Task type")
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model, args.num_classes, args.checkpoint, args.output, args.task)
