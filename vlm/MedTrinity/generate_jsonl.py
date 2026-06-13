import json
import os

# --- RUTAS EN PICASSO ---

BASE_PATH = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/25M_full_55k"

directorios = {
    "train": os.path.join(BASE_PATH, "train"),
    "val": os.path.join(BASE_PATH, "val")
}

# Prompt que se utilizará para entrenar el modelo
#PROMPT_BASE = "Examine this brain MRI scan. Indicate its  Identify any abnormalities, their location, and their characteristics, only if they are present." 
PROMPT_BASE ="""You are provided with this brain MRI image froma medical dataset.
              Your task is to answer the following questions based on the image and green bounding box, and condense your answers into caption-styledtext.
              ###question1
              Give me a detailed description of the image, including type of the image,organs in the image, approximate location of these organs and relavant locations of these organs and any medical devices(if present) visible in the image as detailedly as possible.
              Note when answering question1:
              1. It is irrelevant for this question whether the image depicts a lesion or not. The answer should be the same.
              2. Your answer should not contain anything about the green bounding box like the contour itself and its outline.
              3. Do not explain or emphasize your analysis.
              ###question2
              Identify and describe any lesions visible in the image, if present. These lesions may be indicated by green bounding boxes; in such cases, prioritize describing the specific regions within these boxes. For each lesion, specify its exact anatomical location and its position relative to reference structures. Describe what is unusual in those regions, providing details on signal intensity, texture, size, and other morphological features.
              Note when answering question2:
              1. If one or more green bounding boxes are present, they indicate regions of interest; prioritize identifying and describing the pathological findings within every provided bounding box.
              2. If no lesion is visible in the image or within the bounding boxes, explicitly state that the image shows no visible lesion.
              3. Do not mention the box's physical attributes (e.g., "green line", "contour", or "square outline").
              4. Do not use phrase "green bounding box" in your response, use "region of interest" as a substitution.
              5. If multiple lesions or bounding boxes exist, ensure each is described individually to maintain a comprehensive clinical description.
              6. Do not say anything that is not needed in your analysis, like introduction of the disease and medical equipments.
              7. Do not explain or emphasize your analysis.
              ###question3
              Specify the relationship between the identified lesion(s) and other anatomical regions, explaining the underlying cause of this interaction and its clinical probability.
              Note when answering question3:
              1. Answer this question only if a lesion is present in the image. If no lesion is visible, provide no response for this section.
              2. The response must be highly condensed, limited to a maximum of 2 lines.
              3. Do not emphasize your analysis.
              ###Consolidated Clinical Narrative
              Synthesize the findings from the three previous questions into a single, fluid paragraph. Avoid a "Question-Answer" format or bullet points. The final output must be a cohesive and slightly condensed descriptive summary that preserves all essential medical details and observations.
    
              """

def create_jsonl(tipo_dataset, ruta_directorio):
    archivo_metadata = os.path.join(ruta_directorio, "metadata.json")
    archivo_salida = f"{tipo_dataset}_dataset.jsonl"
    
    if not os.path.exists(archivo_metadata):
        print(f"ERROR: No se encontró {archivo_metadata}")
        return

    print(f"Generando {archivo_salida}...")
    
    with open(archivo_metadata, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    registros_procesados = []
    
    for entrada in datos: #cada linea del jsonl
        nombre_archivo = entrada.get("file_name")
        caption = entrada.get("caption")
        ruta_imagen = os.path.join(ruta_directorio, nombre_archivo)
        
        if os.path.exists(ruta_imagen):
            # El token <image> DEBE ir al principio del prefix
            registro = {
                "image": ruta_imagen,
                "prefix": f"<image> {PROMPT_BASE}",  #prefix es imagen+ prompt(el mismo para todos)
                "suffix": caption #suffix es el caption, lo que debe responder el modelo
            }
            registros_procesados.append(registro)

    with open(archivo_salida, 'w', encoding='utf-8') as f:
        for reg in registros_procesados:
            f.write(json.dumps(reg) + "\n")
            
    print(f" Creado con éxito. {len(registros_procesados)} muestras en {archivo_salida}")

# Creacion de los jsonl
create_jsonl("train", directorios["train"])
create_jsonl("val", directorios["val"])