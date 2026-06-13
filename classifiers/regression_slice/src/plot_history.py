import torch
import os
import argparse
from utils import plot_learning_curves

def main():
    parser = argparse.ArgumentParser(description="Graficar curvas de aprendizaje de regresión")
    parser.add_argument("--history", type=str, 
                        default=r"D:\classifiers\regression_slice\checkpoints\efficientnet_v2_s-98k\history.pth",
                        help="Ruta al archivo history.pth")
    parser.add_argument("--model", type=str, default="efficientnet_v2_s", help="Nombre del modelo")
    parser.add_argument("--train_k", type=str, default="98k", help="Tamaño del dataset (ej: 98k)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.history):
        print(f"Error: No se encontró el archivo {args.history}")
        return

    print(f"Cargando historial desde: {args.history}")
    history = torch.load(args.history, map_location='cpu')
    
    # El directorio de guardado será: ./classifiers/regression_slice/figures/modelo-90k
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(
        project_root, 
        "classifiers", 
        "regression_slice", 
        "figures", 
        f"{args.model}-{args.train_k}"
    )
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generando gráficas con formato profesional en: {save_dir}")
    plot_learning_curves(history, args.model, args.train_k, save_dir)
    print(f"¡Hecho! Las gráficas se han guardado en: {save_dir}")

if __name__ == "__main__":
    main()
