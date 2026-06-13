#  Medical VLM: Modelos de visión-lenguaje y Classificadores para imágenes de resonancia magnética cerebrales

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-blue.svg)

**Medical VLM** es un repositorio integral diseñado para el procesamiento, clasificación y análisis avanzado de imágenes de resonancia magnética (MRI). El proyecto combina enfoques tradicionales de Deep Learning (CNNs para clasificación y regresión) con el modelos de lenguaje visual (VLM) de vanguardia **MedGemma**, mejorado utilizando el dataset **MedTrinity**.

---

##  Tabla de Contenidos

1. [Estructura del Proyecto](#-estructura-del-proyecto)
2. [Requisitos e Instalación](#-requisitos-e-instalación)
3. [Módulo 1: Clasificadores y Regresión (Deep Learning Tradicional)](#-módulo-1-clasificadores-y-regresión-deep-learning-tradicional)
   - [Clasificador de Secuencia](#1-clasificador-de-secuencia)
   - [Clasificador de Vista (Plano)](#2-clasificador-de-vista)
   - [Regresión de Slices](#3-regresión-de-slices)
4. [Módulo 2: Vision-Language Models (VLM)](#-módulo-2-vision-language-models-vlm)
   - [Fine-Tuning de MedGemma](#1-fine-tuning-de-medgemma)
   - [Inferencia](#2-inferencia)
   - [Evaluación Avanzada (LLM-as-a-judge & BERTScore)](#3-evaluación-avanzada)
5. [Consideraciones para el HPC (Picasso)](#-consideraciones-para-el-hpc-picasso)

---

##  Modelos Pre-entrenados y Pesos (Hugging Face)

Los modelos resultantes y sus pesos han sido publicados en el Hugging Face Hub para facilitar su acceso y reutilización:

- **[MedGemma-FT-Neuroimaging](https://huggingface.co/pedrofdez/MedGemma-FT-Neuroimaging)**: Modelo de lenguaje visual MedGemma con ajuste fino (*fine-tuning*) especializado en neuroimagen y descripciones clínicas MRI.
- **[Classifiers View and Sequence (MobileNetV2)](https://huggingface.co/pedrofdez/classifiers_view_and_sequence)**: Pesos pre-entrenados para la arquitectura MobileNetV2, correspondientes a los clasificadores de plano anatómico (*view*) y de secuencia MRI (*sequence*).

---

## Estructura del Proyecto

El repositorio está dividido en dos grandes bloques:

```text
Medical_VLM/
│
├── classifiers/                    # Modelos CNN tradicionales (ResNet, VGG, MobileNet)
│   ├── classifier_sequence/        # Clasificación del tipo de secuencia MRI (FLAIR, T1, T2, etc.)
│   ├── classifier_view/            # Clasificación del plano de la imagen (Axial, Coronal, Sagital)
│   └── regression_slice/           # Predicción/Regresión de la posición del slice
│
├── vlm/                            # Modelos Fundacionales Visión-Lenguaje
│   └── MedTrinity/                 # Scripts para MedGemma y dataset MedTrinity
│       ├── data/                   # Listas de shards e información del dataset
│       ├── inferences/             # Scripts para generar descripciones con el VLM
│       └── resultados_entrenamiento/# Monitoreo y gráficas del entrenamiento
│
├── .gitignore                      # Reglas de git (ignora datos pesados, .pyc, etc.)
└── *.pkl                           # Archivos de partición y configuración de datasets
```

---

## Requisitos e Instalación

1. **Entorno Python**: Se recomienda Python 3.10 o superior.
2. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd Medical_VLM
   ```
3. **Instalar dependencias**:
   Se incluye un archivo `requirements.txt` con las versiones exactas utilizadas en el entorno de desarrollo. Para instalar todo, ejecuta:
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: Es posible que necesites instalar una versión específica de PyTorch con soporte para tu versión de CUDA desde la [página oficial de PyTorch](https://pytorch.org/).*

---

## Módulo 1: Clasificadores y Regresión (Deep Learning Tradicional)

Este módulo utiliza modelos preentrenados (VGG16, ResNet18, MobileNet_V2, EfficientNet) adaptados para tareas específicas de MRI.

### Funcionalidades Comunes
- **Reanudación automática:** Si el entrenamiento se interrumpe, al volver a ejecutar el script `main.py` se cargarán los pesos, el optimizador y el historial previos de manera automática.
- **Configuración:** Los hiperparámetros (Batch Size, Learning Rate, Epochs) se configuran directamente al inicio de los archivos `src/main.py` de cada subcarpeta.

### 1. Clasificador de Secuencia
Clasifica la imagen MRI según su secuencia (FLAIR, T1-w, T1-ce, T2-w).
- **Ruta:** `classifiers/classifier_sequence/src/`
- **Cómo ejecutar el entrenamiento:**
  ```bash
  python classifiers/classifier_sequence/src/main.py
  ```
- **Evaluación e Inferencia:** Puedes usar `eval_results.py` o `inference.py` para probar el modelo guardado en la carpeta `checkpoints/`.

### 2. Clasificador de Vista
Clasifica el plano anatómico de la resonancia (Axial, Coronal, Sagital).
- **Ruta:** `classifiers/classifier_view/src/`
- **Cómo ejecutar el entrenamiento:**
  ```bash
  python classifiers/classifier_view/src/main.py
  ```

### 3. Regresión de Slices
Predice la posición relativa del "slice" dentro del volumen 3D original.
- **Ruta:** `classifiers/regression_slice/src/`
- **Cómo ejecutar el entrenamiento:**
  ```bash
  python classifiers/regression_slice/src/main.py
  ```

---

##  Módulo 2: Modelos de visión-lenguaje (VLM)

Este módulo se centra en adaptar (Fine-Tuning) y evaluar **MedGemma** utilizando técnicas eficientes como **LoRA** (Low-Rank Adaptation) sobre el dataset **MedTrinity**.

### 1. Fine-Tuning de MedGemma
El script principal prepara los prompts multimodales (Imagen + Texto), enmascara los tokens correctos para la función de pérdida y entrena el modelo en formato bfloat16.

- **Ruta principal:** `vlm/MedTrinity/finetuning.py` (o la versión `finetunig10k.py` para pruebas rápidas).
- **Configuración Destacada en `finetuning.py`:**
  - `dataset_version`: Define la variante del dataset a usar (ej. `25M_demo`).
  - `model_id`: Ruta local del modelo base a cargar.
  - `args (SFTConfig)`: Controla el checkpointing, steps de guardado y parámetros de LoRA.
- **Cómo ejecutar:**
  ```bash
  python vlm/MedTrinity/finetuning.py
  ```
  *Nota: Soporta reanudación automática si detecta un checkpoint en el directorio de salida.*

### 2. Inferencia
Una vez entrenado el modelo (o usando uno base), se pueden generar descripciones clínicas de nuevas resonancias.
- **Ruta:** `vlm/MedTrinity/inferences/inferences_medgemma.py`
- **Cómo ejecutar:**
  ```bash
  python vlm/MedTrinity/inferences/inferences_medgemma.py
  ```

### 3. Evaluación Avanzada
Para medir la calidad de los textos médicos generados por el VLM, no basta con métricas clásicas, por lo que se incluyen:
- **BERTScore:** Evalúa la similitud semántica.
  ```bash
  python vlm/MedTrinity/evaluate_bertscore.py
  ```
- **LLM-as-a-Judge:** Utiliza un LLM superior (ej. GPT-4 o Claude) para juzgar clínicamente la respuesta generada vs el Ground Truth.
  ```bash
  python vlm/MedTrinity/llm-as-a-judge.py
  ```

---

## 🖥️ Consideraciones para el HPC (Picasso)

Gran parte de las rutas dentro de los scripts apuntan a un sistema de alto rendimiento (Ej: `/mnt/home/users/tic_163_colab/pedrofdez/fscratch/picasso/`). 

Si vas a ejecutar este código en **local** o en otra máquina:
1. Busca la variable `DATA_DIR` o `BASE_PATH` en los archivos `main.py` o `finetuning.py`.
2. Comenta la ruta de Picasso y descomenta (o añade) tu ruta local. (Ej: `DATA_DIR = 'D:/classifiers/...'`).
3. Asegúrate de ignorar los archivos de salida generados por el gestor de colas Slurm (configurado en `.gitignore` como `*.out`, `*.err`, `slurm-*.out`).


