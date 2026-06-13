# %%
dataset_version="25M_demo"

# %% [markdown]
# Cargamos los conjuntos de entrenamiento y validación

# %%
from datasets import load_from_disk, DatasetDict
import os



BASE_PATH=f"/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/dataset/{dataset_version}"
train_ds = load_from_disk(os.path.join(BASE_PATH, "train"))
validation_ds=load_from_disk(os.path.join(BASE_PATH, "validation"))

# %% [markdown]
# Creamos el formato de conversación con el que entrena MedGemma. Creamos una nueva columna con la conversacion con la que sera entrenado el modelo para cada par imagen-texto

# %%
def formatting_message(example):
    example["message"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", #la imagen que se le pasa
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text":example["caption"],
                },
            ],
        },
    ]
    return example



# %%
#PROMPT="Describe this brain MRI scan in detail "
#train_ds_small= train_ds.select(range(10))
#formatted_train_small = train_ds_small.map(formatting_message)
# formatted_train_small["message"]

# %%

PROMPT="Describe this brain MRI scan in detail "
formatted_train = train_ds.map(formatting_message)
formatted_validation=validation_ds.map(formatting_message)




# %% [markdown]
# Cargamos el modelo medGemma

# %%
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

model_id = "/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/modelo"

# Check if GPU supports bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")


model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = AutoModelForImageTextToText.from_pretrained(model_id,local_files_only=True, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id,local_files_only=True)

# Use right padding to avoid issues during training
processor.tokenizer.padding_side = "right"

# %% [markdown]
# Usamos Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method

# %%
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear", #Busca automáticamente todas las capas lineales del modelo y les pone un adaptador Lora
    task_type="CAUSAL_LM", #Tipico de LLMs normales. Predice la siguiente palabra basada en las anteriores
    modules_to_save=[ #estos modulos tambien van a ser entrenados (creo que se podrían quitar)
        "lm_head", # ultima capa que predice la siguiente palabra
        "embed_tokens", # pasa el texto a tokens
    ],
)



# %% [markdown]
# Definimos un collator para tratar con imagenes y texto en el entrenamiento
# Toma un batch de examples imagenes, texto juntos y los separa para que solo aprenda el texto.
#Tambien tokeniza el texto y procesa la imagen 
# Devuelve los labels que el modelo debera aprender. Los labels es la entrada (inputs_id) pero quitandole la imagen, el padding...

# %%
def collate_fn(examples: list[dict[str, any]]):
    texts = []
    images = []
    for example in examples: #examples es una lista de tamaño del batch de diccionarios
        images.append([example["image"]])
        texts.append(
            processor.apply_chat_template(
                example["message"], add_generation_prompt=False, tokenize=False
            ).strip()
        )

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, with the padding and image tokens masked in
    # the loss computation
    labels = batch["input_ids"].clone() #Es la entrada completa. Token de la imagen, la pregunta, padding...

    # Mask image tokens
    # Identifica el número exacto (ID) que representa a la "Imagen" dentro del vocabulario del modelo
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]

    # Mask tokens that are not used in the loss computation

    #-100 significa que se ignore para calcular la perdida
    labels[labels == processor.tokenizer.pad_token_id] = -100  #Mask the padding
    labels[labels == image_token_id] = -100 #Mask the image
    labels[labels == 262144] = -100 #262144 es un numero específico de la arquitectura Gemma/PaliGemma. Mask it.

    batch["labels"] = labels
    return batch #BATCH CONTIENE inputs_ids, labels y pixel values

# %% [markdown]
#
#
# <table style="width:100%; border-collapse: collapse; border: 1px solid #ddd;">
#     <thead>
#         <tr style="background-color: #f2f2f2;">
#             <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 15%;">Concepto</th>
#             <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 35%;">Contenido (Simplificado)</th>
#             <th style="border: 1px solid #ddd; padding: 10px; text-align: left; width: 50%;">Función</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td style="border: 1px solid #ddd; padding: 10px;"><strong><code>input_ids</code></strong></td>
#             <td style="border: 1px solid #ddd; padding: 10px; font-family: monospace;"><code>[&lt;IMAGEN&gt;, Es, un, glioma]</code></td>
#             <td style="border: 1px solid #ddd; padding: 10px;">Es la <strong>secuencia completa</strong> (imagen + texto) que el modelo <strong>LEE</strong> y utiliza como contexto.</td>
#         </tr>
#         <tr>
#             <td style="border: 1px solid #ddd; padding: 10px;"><strong><code>labels</code></strong></td>
#             <td style="border: 1px solid #ddd; padding: 10px; font-family: monospace;"><code>[-100, Es, un, glioma]</code></td>
#             <td style="border: 1px solid #ddd; padding: 10px;">Es la <strong>respuesta correcta</strong> que el modelo <strong>DEBE GENERAR</strong>. El valor <code>-100</code> indica a PyTorch que ignore ese token y no lo evalúe.</td>
#         </tr>
#     </tbody>
# </table>

# %% [markdown]
# Argumentos de entrenamiento

# %%
from trl import SFTConfig

args = SFTConfig(
    output_dir=f"/mnt/home/users/tic_163_colab/pedrofdez/resultados_FT_Medgemma/medgemma-FT-MedTrinity{dataset_version}",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=0.1,
    save_strategy="steps",
    save_steps=200,              # Guarda cada 200 pasos (aprox cada 3-4 horas)
    save_total_limit=2,          #Guarda los dos ultimos checkpoints
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to="none",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns = False, #obligas a mantener todas las columnas (incluida la imagen) para que lleguen al collator
    label_names=["labels"],
)

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=formatted_train,
    eval_dataset=formatted_validation.shuffle().select(range(50)),
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# %% [markdown]
# ## Entrenamiento del modelo

from transformers.trainer_utils import get_last_checkpoint
# Lógica para detectar si existe un checkpoint previo
last_checkpoint = None
if os.path.isdir(args.output_dir):
    last_checkpoint = get_last_checkpoint(args.output_dir)

if last_checkpoint:
    print(f" Se ha detectado un checkpoint previo. Reanudando desde: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("🚀 No se encontraron checkpoints previos. Iniciando entrenamiento desde cero.")
    trainer.train()

# %%
trainer.save_model()
processor.save_pretrained(args.output_dir) # Es buena práctica guardar también el processor