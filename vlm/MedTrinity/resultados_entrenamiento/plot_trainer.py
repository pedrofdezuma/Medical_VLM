# %% [markdown]
# # Visualización de las métricas obtenidas en el entrenamiento

# %%
import json
import argparse #para obterner los argumentos desde consola
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %% [markdown]
# Función para cargar el JSON y transformar la columna *`log_history`* a DataFrame de Pandas

# %%
def load_and_process_data(filepath):
    """
    Carga el JSON y convierte el log_history en un DataFrame de Pandas.
    Separa métricas de entrenamiento y evaluación.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extraemos el historial
        if 'log_history' not in data:
            raise ValueError("El archivo JSON no contiene 'log_history'.")
            
        df = pd.DataFrame(data['log_history'])
        
        # Pandas es eficiente manejando datos faltantes (NaN).
        # Los logs de HuggingFace suelen tener filas separadas para train y eval.
        
        return df
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return None

# %% [markdown]
# Función para la creación de gráficas.

# %%
def plot_training_results(df: pd.DataFrame, output_dir: str, model_alias:str,grafLoss:bool =True,grafLR:bool=True,grafAcc:bool=True)->None:
    """
    Genera gráficas eficientemente usando Seaborn y guarda los resultados.
    
    Args:

        - df (pd.DataFrame):Datos de 'log_history' del JSON
        - output_dir (str): Ruta del directorio donde se guardan las gráficas
        - model_alias (str): alias con el que se conoce al modelo entrenado 
        - grafLoss (bool): Whether to plot the loss curves. Default to True.
        - grafLR (bool): Whether to plot the learning rate evolution along steps. Default to True.
        - grafAcc (bool): Whether to plot extra training and evaluation metrics. Default to True.
   
    """
    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14}) # Aumentar tamaño de fuente general
    
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. PREPARACIÓN DE DATOS
    train_df = df[df['loss'].notna()].copy()
    eval_df = df[df['eval_loss'].notna()].copy()
    
    # ---------------------------------------------------------
    # GRAFICA 1: Loss (Pérdida) - Train vs Eval
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not train_df.empty:
        sns.lineplot(data=train_df, x='step', y='loss', label='Entrenamiento', color='blue', alpha=0.6)
    
    if not eval_df.empty:
        sns.lineplot(data=eval_df, x='step', y='eval_loss', label='Validación', color='red', marker='o')
        
    plt.title('Pérdida (Entropía Cruzada)', fontsize=24, fontweight='bold', pad=25)
    plt.xlabel('Paso (Step)', fontsize=18)
    plt.ylabel('Entropía Cruzada', fontsize=18)
    plt.legend(fontsize=16)
    
    # Ajustar eje X cada 100 pasos
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_alias}_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{model_alias}_loss.eps"), format='eps', bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {model_alias}_loss.png / .eps")

    # ---------------------------------------------------------
    # GRAFICA 2: Learning Rate y Epoch
    # ---------------------------------------------------------
    if 'learning_rate' in train_df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:green'
        ax1.set_xlabel('Paso (Step)', fontsize=18)
        ax1.set_ylabel('Tasa de Aprendizaje (LR)', color=color, fontsize=18)
        sns.lineplot(data=train_df, x='step', y='learning_rate', ax=ax1, color=color)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        
        # Ajustar eje X cada 100 pasos
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))

        # Segundo eje para ver el progreso de épocas
        ax2 = ax1.twinx()  
        color = 'tab:gray'
        ax2.set_ylabel('Época', color=color, fontsize=18)
        sns.lineplot(data=df, x='step', y='epoch', ax=ax2, color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

        plt.title('Evolución LR y Épocas', fontsize=24, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_alias}_lr.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{model_alias}_lr.eps"), format='eps', bbox_inches='tight')
        plt.close()
        print(f"-> Gráfica guardada: {model_alias}_lr.png / .eps")

    # ---------------------------------------------------------
    # GRAFICA 3: Métricas adicionales (Accuracy, Entropy, etc.)
    # ---------------------------------------------------------
    potential_metrics = [col for col in df.columns if 'accuracy' in col or 'entropy' in col]
    
    base_metrics = set()
    for m in potential_metrics:
        if m.startswith('eval_'):
            base_metrics.add(m.replace('eval_', ''))
        else:
            base_metrics.add(m)

    for metric in base_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        # Plot Train
        if metric in train_df.columns:
            sns.lineplot(data=train_df, x='step', y=metric, label='Entrenamiento', alpha=0.6)
            has_data = True
            
        # Plot Eval
        eval_metric = f"eval_{metric}"
        if eval_metric in eval_df.columns:
            sns.lineplot(data=eval_df, x='step', y=eval_metric, label='Validación', marker='o')
            has_data = True
            
        if has_data:
            # Traducir nombre de métrica para el título si es posible
            metric_title = metric.replace('_', ' ').capitalize()
            plt.title(f'Métrica: {metric_title}', fontsize=24, fontweight='bold', pad=25)
            plt.xlabel('Paso (Step)', fontsize=18)
            plt.ylabel(metric_title, fontsize=18)
            plt.legend(fontsize=16)

            # Ajustar eje X cada 100 pasos
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.tick_params(axis='both', labelsize=14)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_alias}_{metric}.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f"{model_alias}_{metric}.eps"), format='eps', bbox_inches='tight')
            plt.close()
            print(f"-> Gráfica guardada: {model_alias}_{metric}.png / .eps")

    # ---------------------------------------------------------
    # GRAFICA 4: Pérdida por ÉPOCAS
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    if not train_df.empty:
        sns.lineplot(data=train_df, x='epoch', y='loss', label='Entrenamiento', color='blue', alpha=0.6)
    if not eval_df.empty:
        sns.lineplot(data=eval_df, x='epoch', y='eval_loss', label='Validación', color='red', marker='o')
        
    plt.title('Pérdida (Entropía Cruzada) por Época', fontsize=24, fontweight='bold', pad=25)
    plt.xlabel('Época', fontsize=18)
    plt.ylabel('Entropía Cruzada', fontsize=18)
    plt.legend(fontsize=16)
    
    # Ajustar eje X cada 0.2 épocas
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis='both', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_alias}_loss_epoch.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{model_alias}_loss_epoch.eps"), format='eps', bbox_inches='tight')
    plt.close()
    print(f"-> Gráfica guardada: {model_alias}_loss_epoch.png / .eps")

    # ---------------------------------------------------------
    # GRAFICA 5: Mean Token Accuracy por ÉPOCAS
    # ---------------------------------------------------------
    # Buscamos si existe la métrica específica
    acc_metric = 'mean_token_accuracy'
    if acc_metric in train_df.columns or f'eval_{acc_metric}' in eval_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if acc_metric in train_df.columns:
            sns.lineplot(data=train_df, x='epoch', y=acc_metric, label='Entrenamiento', alpha=0.6)
        
        eval_acc = f"eval_{acc_metric}"
        if eval_acc in eval_df.columns:
            sns.lineplot(data=eval_df, x='epoch', y=eval_acc, label='Validación', marker='o')
            
        plt.title('Precisión de Tokens por Época', fontsize=24, fontweight='bold', pad=25)
        plt.xlabel('Época', fontsize=18)
        plt.ylabel('Precisión (Mean Token Accuracy)', fontsize=18)
        plt.legend(fontsize=16)
        
        # Ajustar eje X cada 0.2 épocas
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_alias}_accuracy_epoch.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{model_alias}_accuracy_epoch.eps"), format='eps', bbox_inches='tight')
        plt.close()
        print(f"-> Gráfica guardada: {model_alias}_accuracy_epoch.png / .eps")


# %% [markdown]
# Función main. Está diseñada para que se pueda llamar desde consolo al método. Es posible pasarle tantos JSONs como se quiera.

# %%
def main():
    
    # Inicializamos el analizador de argumentos para interactuar con la terminal
    parser = argparse.ArgumentParser(description="Procesar logs de entrenamiento (JSON) y generar gráficas.")
    
    # Argumento posicional: permite pasar uno o más archivos (gracias a nargs='+')
    # Ejemplo: python script.py file1.json file2.json
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='Ruta al archivo(s) .json') #importante el nargs
    
    # Argumento opcional: permite definir la ruta de salida
    parser.add_argument('--out', type=str, default=None, help='Directorio de salida para las gráficas')

    # Parseamos los argumentos introducidos por el usuario
    args = parser.parse_args()

    # Procesamiento por lotes: iteramos sobre cada archivo proporcionado
    for file_path in args.files:
        # Validación de seguridad: verificamos que el archivo realmente exista en el disco
        if os.path.exists(file_path):
            print(f"--- Processing: {file_path} ---")
            
            # Carga de datos mediante la función auxiliar
            df = load_and_process_data(file_path)
            
            if df is not None:
                filename_base = os.path.splitext(os.path.basename(file_path))[0]
                
                # Si no se especifica --out, crear carpeta 'plots' en el mismo directorio que el JSON
                output_dir = args.out
                if output_dir is None:
                    output_dir = os.path.join(os.path.dirname(file_path), 'plots')
                
                # Ejecución de la lógica de graficado
                plot_training_results(df, output_dir, filename_base)
        else:
            # Notificación de error amable si una de las rutas es incorrecta
            print(f"Error: File not found at {file_path}")



# %%
if __name__ == "__main__":
    main()





