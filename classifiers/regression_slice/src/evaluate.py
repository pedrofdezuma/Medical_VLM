import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import get_regression_loaders
from model import get_regression_model
import os

def evaluate_model(data_dir, model_path, model_name="resnet18", device='cuda'):
    """
    Evalúa el rendimiento del modelo de regresión utilizando MAE, MSE y R^2.
    Genera una gráfica de dispersión para visualizar la calidad de las predicciones.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")

    # 1. Preparar Dataloader y Modelo
    # El loader de 'val' apunta por defecto a 'imagesTs' según dataset.py
    loaders = get_regression_loaders(data_dir, batch_size=16)
    val_loader = loaders['val']
    
    # Instanciar arquitectura y cargar pesos
    model = get_regression_model(model_name, pretrained=False)
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo de pesos en {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Procesando {len(val_loader.dataset)} imágenes de prueba...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # 2. Cálculo de métricas estadísticas
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print("\n" + "="*40)
    print("       REPORTE DE RENDIMIENTO")
    print("="*40)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"R^2 Score (Determ.):       {r2:.4f}")
    print("="*40)

    # 3. Generación de Gráficas de Diagnóstico
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(9, 9))
    plt.scatter(all_labels, all_preds, alpha=0.3, color='#2c7bb6', label='Predicciones')
    
    # Línea de referencia y=x (predicción perfecta)
    plt.plot([-1, 1], [-1, 1], color='#d7191c', linestyle='--', lw=2, label='Ideal (Error 0)')
    
    plt.xlabel('Posición Real del Corte (Normalizada [-1, 1])')
    plt.ylabel('Posición Predicha por el Modelo')
    plt.title(f'Evaluación de Regresión: {model_name}\nR² = {r2:.4f} | MAE = {mae:.4f}')
    plt.legend(['Predicciones', 'Ideal (Error 0)'])
    plt.grid(True, linestyle=':', alpha=0.7)
    
    output_plot = 'results/evaluation_scatter.png'
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"\nVisualización guardada en: {output_plot}")

if __name__ == '__main__':
    # Configuración de rutas
    DATA_PATH = r"D:\classifiers\regression_slice\data\2D_MRI_ms_ep_control_FLAIR_T1"
    MODEL_WEIGHTS = "checkpoints/best_regression_model.pth"
    
    evaluate_model(DATA_PATH, MODEL_WEIGHTS, model_name="resnet18")
