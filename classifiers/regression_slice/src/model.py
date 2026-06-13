import torch
import torch.nn as nn
from torchvision import models
import os

def get_regression_model(model_name="resnet18", pretrained=True, picasso=True):
    """
    Fábrica de modelos de regresión.
    Carga pesos locales para evitar errores de conexión en servidores como Picasso.
    """
    
    # 1. Instanciar el modelo base sin pesos
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Modelo {model_name} no soportado")

    # 2. Cargar pesos preentrenados localmente (Lógica de tus otros clasificadores)
    if pretrained:
        if picasso:
            # Ruta en Picasso
            weights_path = f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/classifiers/model_weights/{model_name}-weights.pth"
        else:
            # Ruta local Windows
            weights_path = f"D:/classifiers/model_weights/{model_name}-weights.pth"
            
        if os.path.exists(weights_path):
            print(f" >>> Cargando pesos preentrenados locales desde: {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        else:
            print(f" AVISO: No se encontró el archivo de pesos en {weights_path}. Se inicializará con pesos aleatorios.")

    # 3. Configurar la cabeza de regresión (salida única con Tanh)
    if model_name in ["resnet18", "resnet50"]:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Tanh()
        )
    elif model_name == "efficientnet_v2_s":
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1),
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
        
    model.model_name = model_name
    return model

if __name__ == '__main__':
    # Prueba rápida
    m = get_regression_model("resnet18", picasso=False)
    print(f"Modelo {m.model_name} configurado correctamente.")
