import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, LoraConfig, PeftConfig
import os
import sys
import logging
import gc
import json
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

try:
    # 1. Configuration
    base_model_path = r"D:\modelos\MedGemma\medgemma1.5"
    adapter_path = r"D:\modelos\MedTrinity25M_full\MedTrinity25M_full_55k\final_model" 
    save_path = r"D:\modelos\MedTrinity25M_full\MedTrinity25M_full_55k\merge_model"

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    cleanup()

    # Load adapter config and handle modules_to_save manually to avoid OOM
    print(f"Loading and modifying adapter config from {adapter_path}...")
    config = LoraConfig.from_pretrained(adapter_path)
    
    modules_to_save = config.modules_to_save
    # Disable modules_to_save in the config so PEFT doesn't deepcopy them
    config.modules_to_save = None

    print(f"Loading LoRA adapters (bypassing modules_to_save: {modules_to_save})...")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        config=config,
        is_trainable=False,
        low_cpu_mem_usage=True
    )

    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    del model
    cleanup()

    if modules_to_save:
        print(f"Manually loading weights for modules_to_save: {modules_to_save}")
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_weights_path):
            adapter_weights = load_file(adapter_weights_path)
            
            state_dict = merged_model.state_dict()
            applied_count = 0
            
            for key, value in adapter_weights.items():
                # Map keys from PEFT format to base model format
                clean_key = key.replace("base_model.model.", "")
                if "modules_to_save.default" in clean_key:
                    clean_key = clean_key.replace(".modules_to_save.default", "")
                
                if clean_key in state_dict:
                    state_dict[clean_key].copy_(value.to(state_dict[clean_key].dtype))
                    applied_count += 1
                    print(f"  Applied weight for: {clean_key}")
            
            print(f"Applied {applied_count} manual module weights.")
            del adapter_weights
        else:
            print(f"Warning: {adapter_weights_path} not found. Could not apply modules_to_save weights.")
        
        cleanup()

    print(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path, safe_serialization=True)

    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, local_files_only=True)
    processor.save_pretrained(save_path)

    print(f"Done! Merged model saved successfully in {save_path}")

except Exception as e:
    print(f"\nAN ERROR OCCURRED:\n{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
