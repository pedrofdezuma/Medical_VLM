import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, LoraConfig
import os
import sys
import logging
import gc
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

try:
    # 1. Configuration for Picasso
    # Based on your adapter_config.json, the base model is here:
    base_model_path = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/medgemma1.5"
    
    # Path to your adapter on Picasso 
    adapter_path = "/mnt/home/users/tic_163_colab/pedrofdez/resultados_FT_Medgemma/medgemma-FT-MedTrinity25M_full_55k/final_model"
    
    # Path where to save the merged model
    save_path = "/mnt/home/users/tic_163_colab/pedrofdez/resultados_FT_Medgemma/medgemma-FT-MedTrinity25M_full_55k/merge_model"

    print(f"Loading base model from {base_model_path}...")
    # On Picasso we have plenty of RAM, so we can load normally
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    cleanup()

    print(f"Loading LoRA adapters from {adapter_path}...")
    # Loading the config to see modules_to_save
    config = LoraConfig.from_pretrained(adapter_path)
    modules_to_save = config.modules_to_save
    
    # On Picasso, we can try loading EVERYTHING at once if we have >32GB RAM
    # But for safety, we'll keep the manual bypass if you want, or just load it directly.
    # Let's try loading it DIRECTLY first, as Picasso nodes usually have 128GB+.
    
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        is_trainable=False,
        low_cpu_mem_usage=True
    )

    print("Merging weights...")
    merged_model = model.merge_and_unload()
    del model
    cleanup()

    print(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path, safe_serialization=True)

    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(save_path)

    print(f"Done! Merged model saved successfully in {save_path}")

except Exception as e:
    print(f"\nAN ERROR OCCURRED:\n{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
