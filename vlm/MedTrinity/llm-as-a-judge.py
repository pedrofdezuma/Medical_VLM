# -*- coding: utf-8 -*-
import os
import json
import re
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------
# 1. HARDCODED CONFIGURATION
# ---------------------------------------------


PICASSO = True 
INFERENCE_FOLDER="shard_57-50"

if PICASSO:
    MODEL_ID = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/prometheus/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5"
    DEFAULT_CACHE_DIR = None
    DEFAULT_INPUT_FILE = rf"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/inferences/{INFERENCE_FOLDER}/MedTrinity25M_full_55k/answers.jsonl"
    DEFAULT_OUTPUT_FILE = rf"/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/dataset/inferences/{INFERENCE_FOLDER}/MedTrinity25M_full_55k/llm_evaluation_pointwise.jsonl"
else:
    MODEL_ID = "/mnt2/fscratch/users/tic_163_colab/pedrofdez/picasso/prometheus/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5"
    DEFAULT_CACHE_DIR = r"D:\modelos"
    DEFAULT_INPUT_FILE  = rf"D:\inferences\{INFERENCE_FOLDER}\MedTrinity25M_full_55k\answers.jsonl"
    DEFAULT_OUTPUT_FILE = rf"D:\inferences\{INFERENCE_FOLDER}\MedTrinity25M_full_55k\llm_evaluation_pointwise.jsonl"


# ---------------------------------------------
# 2. RUBRICS (POINTWISE 1-5)
# ---------------------------------------------

RUBRIC_POINTWISE = """[System: Expert Neuroradiologist Judge]
Grade the AI-generated MRI report on a scale of 1.0 to 5.0 based on the Reference (REF).

SCORING BENCHMARKS (Use the full range):
- 1.0: Severe clinical failure, hallucinations, or dangerous misinformation.
- 2.0: Major anatomical errors (e.g., wrong lobe/side) or extremely poor description.
- 3.0: Factually correct but MEDICORE. Contains heavy redundancies, generic language, or lacks professional depth.
- 4.0: HIGH QUALITY. Professional terminology, accurate localization, and clear structure. No significant redundancies.
- 5.0: PERFECT. Indistinguishable from the REF in every clinical and stylistic aspect.


GRANULARITY RULE:
Use decimals (e.g., 3.2, 3.8, 4.5) to indicate if a report is slightly better or worse than the benchmarks. 
Avoid using only integers if you see small differences.

SAFETY & QUALITY FLAGS (Mandatory):
Evaluate the following. Answer "Yes" ONLY if the failure is present:
- CRITICAL_HALLUCINATION: Yes if there is a SEVERE error (inventing/missing a primary lesion).
- ANATOMICAL_MISMATCH: Yes if hemispheres (left/ right) or lobes are wrongly identified.
- REDUNDANCY_LEVEL: Yes if the text is highly repetitive or stuck in a loop.
- CLINICAL_INCOHERENCE: Yes if the report contradicts itself or lacks internal logic.

For each flag, you MUST choose ONLY "Yes" or "No". 
- Use "Yes" if the error is present.
- Use "No" if the error is absent.
DO NOT use "N/A", "Unknown", or any other value. If you are unsure, default to "No".

[Output Format]
FINAL_GRADE: [score]
CRITICAL_HALLUCINATION: [Yes/No]
ANATOMICAL_MISMATCH: [Yes/No]
REDUNDANCY_LEVEL: [Yes/No]
CLINICAL_INCOHERENCE: [Yes/No]
Feedback: [Brief analysis...]
"""

# ---------------------------------------------
# 3. PROMETHEUS INFERENCE HELPER
# ---------------------------------------------

def load_prometheus(model_id, cache_dir=None):
    if cache_dir:
        print(f"Directory set to: {cache_dir}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir,
        local_files_only=PICASSO 
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=PICASSO
    )
    return tokenizer, model

def call_prometheus(tokenizer, model, prompt, max_tokens=768):
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.3  
        )
    
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded.strip()

def extract_score(text):
    if not text: return None
    # Buscamos FINAL_GRADE, Final_Grade, score, grade, etc.
    # Soportamos: FINAL_GRADE: 3.5, [score] 2.0, final grade is 3.0
    patterns = [
        r"FINAL_GRADE[:\s]*([1-5](?:\.\d+)?)",
        r"\[?score\]?[:\s]*([1-5](?:\.\d+)?)",
        r"grade is[:\s]*([1-5](?:\.\d+)?)",
        r"score is[:\s]*([1-5](?:\.\d+)?)"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match: return float(match.group(1))
    
    # Fallback: Ãºltimo nÃºmero 1-5 que no sea parte de un rango
    numbers = re.findall(r"\b([1-5](?:\.\d+)?)\b", text)
    if numbers:
        for n in reversed(numbers):
            if float(n) == 5.0 and "[1-5]" in text: continue
            return float(n)
    return None

def extract_flags(text):
    # Diccionario de búsqueda: Clave JSON -> [Etiqueta a buscar, Palabras clave en texto]
    flags_map = {
        "critical_hallucination": ["CRITICAL_HALLUCINATION", "hallucination", "invented"],
        "anatomical_mismatch": ["ANATOMICAL_MISMATCH", "anatomical error", "location error"],
        "redundancy_level": ["REDUNDANCY_LEVEL", "redundancy", "repetition", "repeated"],
        "clinical_incoherence": ["CLINICAL_INCOHERENCE", "incoherence", "contradict", "inconsistent"],
    }
    
    results = {}
    text_upper = text.upper()

    for key, keywords in flags_map.items():
        # 1. Intento por etiqueta formal (Soporta N/A, corchetes y variaciones)
        # Este regex busca la etiqueta y captura lo que haya después (YES, NO, N/A, X...)
        pattern = fr"{keywords[0]}[\s\]]*[:\-]*\s*(?:\[Yes/No\]\s*)?([A-Z/]+)"
        match = re.search(pattern, text_upper)
        
        if match:
            val = match.group(1).strip()
            if val == "YES":
                results[key] = True
            elif val == "NO":
                results[key] = False
            else:
                # Si es N/A o cualquier otra cosa, buscamos en el texto para desempatar
                results[key] = any(kw.upper() in text_upper for kw in keywords[1:])
        else:
            # 2. Si no hay etiqueta, buscamos si el feedback menciona el problema
            # Ejemplo: si el texto dice "major anatomical error", marcamos True
            found_in_text = any(kw.upper() in text_upper for kw in keywords[1:])
            
            # Verificamos que no sea una negación (ej: "no hallucinations")
            if found_in_text:
                if re.search(fr"no\s+{keywords[1]}", text, re.IGNORECASE):
                    results[key] = False
                else:
                    results[key] = True
            else:
                results[key] = False # Por defecto, si no se menciona, asumimos que está bien

    return results

# ---------------------------------------------
# 4. EVALUATION FUNCTION (POINTWISE)
# ---------------------------------------------

def evaluate_pointwise(tokenizer, model, gt, response):
    prompt = f"{RUBRIC_POINTWISE}\n\nREF:\n{gt}\n\nREPORT TO EVALUATE:\n{response}"
    raw_output = call_prometheus(tokenizer, model, prompt)
    
    score = extract_score(raw_output)
    flags_dict = extract_flags(raw_output)
    
    score_norm = round((score - 1.0) / 4.0, 4) if score is not None else None

    # Estructura plana
    result = {
        "score_1_to_5": score,
        "score_normalized": score_norm
    }
    
    # AÃ±adimos las flags directamente al diccionario principal
    if flags_dict:
        result.update(flags_dict)
    else:
        # Valores por defecto si falla todo
        result.update({k: None for k in ["critical_hallucination", "anatomical_mismatch", "redundancy_level", "clinical_incoherence"]})
    
    result["raw"] = raw_output
    return result

# ---------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    print(f"Initializing {MODEL_ID}...", flush=True)
    tokenizer, model = load_prometheus(MODEL_ID, cache_dir=args.cache_dir)

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    print(f"Evaluating {len(cases)} cases...", flush=True)
    
    # Initialize statistics
    stats = {
        "count": 0,
        "base_score_sum": 0.0,
        "ft_score_sum": 0.0,
        "hallucinations_base": 0,
        "hallucinations_ft": 0
    }

    with open(output_path, "a", encoding="utf-8") as f_out:
        pbar = tqdm(cases, desc="Evaluating")
        for case in pbar:
            file_name = case.get("file_name", "Unknown")
            
            gt = case.get("caption") or ""
            base_r = case.get("model_base_response") or ""
            ft_r = case.get("model_ft_response") or ""

            eval_base = evaluate_pointwise(tokenizer, model, gt, base_r)
            eval_ft = evaluate_pointwise(tokenizer, model, gt, ft_r)

            # Update statistics
            stats["count"] += 1
            if eval_base["score_1_to_5"] is not None:
                stats["base_score_sum"] += eval_base["score_1_to_5"]
            if eval_ft["score_1_to_5"] is not None:
                stats["ft_score_sum"] += eval_ft["score_1_to_5"]
            
            if eval_base.get("critical_hallucination"): stats["hallucinations_base"] += 1
            if eval_ft.get("critical_hallucination"): stats["hallucinations_ft"] += 1

            # Update progress bar description with current averages
            avg_base = stats["base_score_sum"] / stats["count"]
            avg_ft = stats["ft_score_sum"] / stats["count"]
            h_rate_ft = (stats["hallucinations_ft"] / stats["count"]) * 100
            
            pbar.set_postfix({
                "avg_base": f"{avg_base:.2f}",
                "avg_ft": f"{avg_ft:.2f}",
                "h_rate_ft": f"{h_rate_ft:.1f}%"
            })

            res = {
                "case_id": file_name,
                "evaluation": {
                    "base_model": eval_base,
                    "ft_model": eval_ft
                }
            }

            json.dump(res, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush() 

if __name__ == "__main__":
    main()
