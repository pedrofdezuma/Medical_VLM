# # Análisis cualitativo del modelo entrenado con respecto al modelo base

#Imports
import os
import sys
import json  # Import necesario para leer metadata y guardar respuestas

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image
import gc
from tqdm import tqdm

# ## Carga del modelo base de MedGemma

#base_model_id = "google/medgemma-4b-it" # string referencial
# Ajusta esto a tu ruta real en Picasso si difiere
cache_dir_base = "D:\modelos\MedGemma\medgemma1.5" 
base_model_id = "google/medgemma-1.5-4b-it" # string referencial

from transformers import BitsAndBytesConfig

def cargar_modelo_base(model_path:str):
    """
    Devuelve el modelo MedGemma y su processor base
    """
    model_kwargs = dict(
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16, # Recomendado bfloat16 para Ampere (A100) en Picasso
    )
    
    #No se usará en un principio
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True # ESTO ES LA CLAVE
    )

    print("\tCargando el modelo base")
    #En Picasso nunca será online
    #sin quantificacion
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, **model_kwargs)
    print("\tModel base cargado")

    print("\tCarga del processor base")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print("\tProcessor base cargado")

    return model, processor

# %%
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# %%
# Cargar modelo base
base_model, processor = cargar_modelo_base(cache_dir_base)

# ## Carga del modelo obtenido con funetunning. Al modelo base le cargamos los nuevos pesos

# %%
def cargar_modelo_con_adapter(base_model, path_checkpoint:str):
    print("\tCargando el adaptador (LoRA) sobre el modelo base")
    # PEFT fusiona o acopla los pesos sin duplicar la memoria del modelo base
    model_con_ft = PeftModel.from_pretrained(base_model, path_checkpoint) 
    print("\tModel ft cargado (Adapter attached)")
    return model_con_ft

# %%
path_checkpoint = "/mnt/home/users/tic_163_colab/pedrofdez/resultados_FT_Medgemma/medgemma-FT-MedTrinity25M_full_10k/final_model"   #ruta a donde están guardado el checkpoint que vamos a usar

# Ahora 'model' contiene tanto el base como el adaptador
model = cargar_modelo_con_adapter(base_model=base_model, path_checkpoint=path_checkpoint)

# Ponemos el modelo en modo evaluación
model.eval()

# %% [markdown]
# ## Diseño del Prompt

# %% [markdown]
# Usamos el mismo que el usado en el entrenamiento

# %%
# Nota: Ajustamos ligeramente el prompt para asegurarnos de que el processor lo entienda bien
# El processor de Gemma suele manejar la imagen internamente, el texto basta con la pregunta.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # Placeholder para la imagen
            {"type": "text", "text": "Examine this axial brain MRI image. Identify any abnormalities, their location, and their characteristics, only if they are present."}
        ]
    }
]

# Pre-calculamos el template de texto (sin la imagen física aún)
prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# %% [markdown]
# ## Carga la imagen a inferir

# %%
SAMPLES_DIR = "/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/dataset/inferences/axial_mri_samples"

# Creacion del jsonl de respuesta.
answers_file = os.path.join(SAMPLES_DIR, "answers.jsonl")

# Donde se encuentra el caption de cada imagen
metadata_path = os.path.join(SAMPLES_DIR, "metadata.json")

# 1. Cargar metadatos primero
print("Cargando metadatos...")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata_content = json.load(f)

# 2. Filtrar solo las imágenes que existen en la carpeta (y que estén en metadata)
all_files = os.listdir(SAMPLES_DIR)
valid_images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f in metadata_content]

print(f"Se procesarán {len(valid_images)} imágenes.")

# Abrimos el archivo en modo escritura ("w"). Usa "a" si quieres añadir sin borrar lo anterior.
# Abrimos el archivo en modo escritura ("w"). 
with open(answers_file, "w", encoding="UTF-8") as f_out:
    
    for img_filename in tqdm(valid_images, total=len(valid_images), desc="Generando respuestas"):
        img_path_full = os.path.join(SAMPLES_DIR, img_filename)
        
        # Cargar ground truth y metadatos
        caption_real = metadata_content[img_filename]["caption"]
        source_data = metadata_content[img_filename].get("source", "unknown")

        try:
            raw_image = Image.open(img_path_full).convert("RGB")
        except Exception as e:
            print(f"Error cargando imagen {img_filename}: {e}")
            continue
        
        # Preparamos inputs y medimos longitud del prompt
        inputs = processor(text=prompt_text, images=raw_image, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[-1] 

        # ------ MODELO BASE ---------------------
        gc.collect()
        torch.cuda.empty_cache()
        
        respuesta_base = ""
        with model.disable_adapter():
            with torch.no_grad():
                base_output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    use_cache=True
                )
            
            # Recorte y decodificación limpia
            nuevo_contenido_base = base_output_ids[0][input_len:]
            respuesta_base = processor.decode(nuevo_contenido_base, skip_special_tokens=True).strip()

        print(f"Respuesta base generada para: {img_filename}")

        # ---------- MODELO FT -----------------------
        gc.collect()
        torch.cuda.empty_cache()
        
        respuesta_ft = ""
        with torch.no_grad():
            ft_output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                use_cache=True
            )
        
        # Recorte y decodificación limpia
        nuevo_contenido_ft = ft_output_ids[0][input_len:]
        respuesta_ft = processor.decode(nuevo_contenido_ft, skip_special_tokens=True).strip()
        
        print(f"Respuesta ft generada para: {img_filename}")

        # Guardar en el JSONL
        registro = {
            "file_name": img_filename,
            "source": source_data,
            "caption": caption_real,
            "model_base_response": respuesta_base,
            "model_ft_response": respuesta_ft
        }
        
        json.dump(registro, f_out, ensure_ascii=False)
        f_out.write("\n")
        f_out.flush() # Guardado inmediato
        
        print(f"Registro completado para {img_filename}\n")

print(f"Inferencia completada. Resultados en: {answers_file}")