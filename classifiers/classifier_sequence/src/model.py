import torch
import torch.nn as nn
from torchvision import models

def get_mri_model(model_name:str="resnet18", pretrained:bool = False, num_classes=5):
    
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model.model_name=model_name # Add this attribute for plotting
    
    if pretrained:
        weights_path=f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/classifiers/model_weights/{model.model_name}-weights.pth"
        #weights_path = f"D:/classifiers/model_weights/{model.model_name}-weights.pth"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    
    # Freeze all layers except the last one
    # Block layers that already know how to detect edges and shapes.
    # They will be unfrozen during training after several epochs (Two-stage training)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad=False
            
    # Configure the final layer depending on the model
    if model_name == "resnet18":
        num_ftrs=model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "vgg16":
        num_ftrs = model.classifier[6].in_features
        # Ensure only the new layers require grad if pretrained
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


if __name__=='__main__':
    model=get_mri_model()
    print(model)
