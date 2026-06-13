import torch
import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

random.seed(42)

def load_models():
    # Loading medical-specific models for evaluation
    print("Loading medical-specific models...")
    
    # PubMed-trained model for BERTScore
    bert_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Cross-Encoder for NLI to detect contradictions
    nli_model_name = "cross-encoder/nli-deberta-v3-base"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    
    # Move NLI model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nli_model.to(device)
    
    return bert_model, nli_tokenizer, nli_model, device

def evaluate_pair(reference, candidate, bert_model, nli_tokenizer, nli_model, device):
    # --- Calculate Semantic Similarity (BERTScore) ---
    # Using PubMedBERT to ensure clinical terminology is well-captured
    P, R, F1 = score([candidate], [reference], model_type=bert_model, num_layers=9, lang="en", verbose=False)
    raw_f1 = F1.item()

    # --- Check for Clinical Contradictions (NLI) ---
    # Label 0 corresponds to contradiction in DeBERTa-v3-base NLI
    inputs = nli_tokenizer(reference, candidate, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        nli_logits = nli_model(**inputs).logits
        nli_probs = torch.softmax(nli_logits, dim=1)
        contradiction_prob = nli_probs[0][0].item()

    # --- Check Lateral Consistency ---
    # Detects if 'left' and 'right' are swapped between reference and candidate
    lateral_penalty = 0
    keywords = ["left", "right"]
    ref_lat = [w for w in keywords if w in reference.lower()]
    cand_lat = [w for w in keywords if w in candidate.lower()]
    
    if ref_lat and cand_lat and ref_lat[0] != cand_lat[0]:
        lateral_penalty = 0.2  # 20% penalty for laterality mismatch

    # --- Final Weighted Score Calculation ---
    # Current threshold: 0.5 for contradiction. Penalty: 0.5
    penalty = 0.2 if contradiction_prob > 0.5 else 0.0
    final_score = max(0, raw_f1 - penalty - lateral_penalty)

    return {
        "BERTScore_F1": round(raw_f1, 4),
        "Contradiction_Prob": round(contradiction_prob, 4),
        "Lateral_Error": lateral_penalty > 0,
        "Clinical_Score": round(final_score, 4)
    }

def main():
    # Setup Argument Parser with hardcoded default path
    model="MedTrinity25M_full_55k"
    default=rf"D:\inferences\shard_57\{model}"
    parser = argparse.ArgumentParser(description="Analyze Medical VLM answers from JSONL file.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=default, 
        help=f"Directory containing answers.jsonl (Default: {default})"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=f"{default}/evaluation_results.csv", 
        help="Name of the output CSV file"
    )
    args = parser.parse_args()

    # Define file paths
    input_path = os.path.join(args.input_dir, "answers.jsonl")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # Load JSONL data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    random.shuffle(data)

    print(f"Loaded {len(data)} samples from {input_path}")

    # Initialize models and device
    bert_model, nli_tokenizer, nli_model, device = load_models()

    all_results = []
    
    # Initialize statistics
    stats = {
        "count": 0,
        "base_score_sum": 0.0,
        "ft_score_sum": 0.0
    }

    # Process each entry in the dataset
    try:
        pbar = tqdm(data, desc="Evaluating")
        for entry in pbar:
            file_name = entry.get("file_name", "N/A")
            source = entry.get("source", "N/A")
            reference = entry.get("caption", "")
            base_pred = entry.get("model_base_response", "")
            ft_pred = entry.get("model_ft_response", "")

            # Compute metrics for both models
            base_metrics = evaluate_pair(reference, base_pred, bert_model, nli_tokenizer, nli_model, device)
            ft_metrics = evaluate_pair(reference, ft_pred, bert_model, nli_tokenizer, nli_model, device)

            # Update statistics
            stats["count"] += 1
            stats["base_score_sum"] += base_metrics["Clinical_Score"]
            stats["ft_score_sum"] += ft_metrics["Clinical_Score"]

            # Update progress bar
            avg_base = stats["base_score_sum"] / stats["count"]
            avg_ft = stats["ft_score_sum"] / stats["count"]
            pbar.set_postfix({
                "avg_base": f"{avg_base:.3f}",
                "avg_ft": f"{avg_ft:.3f}",
                "delta": f"{avg_ft - avg_base:+.3f}"
            })

            # Consolidate results for CSV export
            result_row = {
                "file_name": file_name,
                "source": source,
                "base_BERTScore": base_metrics["BERTScore_F1"],
                "base_Contradiction": base_metrics["Contradiction_Prob"],
                "base_Lateral_Error": base_metrics["Lateral_Error"],
                "base_Clinical_Score": base_metrics["Clinical_Score"],
                "ft_BERTScore": ft_metrics["BERTScore_F1"],
                "ft_Contradiction": ft_metrics["Contradiction_Prob"],
                "ft_Lateral_Error": ft_metrics["Lateral_Error"],
                "ft_Clinical_Score": ft_metrics["Clinical_Score"],
            }
            all_results.append(result_row)
            
    finally:
        # Save results to a DataFrame and export to CSV
        df = pd.DataFrame(all_results)
        output_path = os.path.join(args.input_dir, args.output_csv)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}. {len(all_results)} saved in total")

        # Display final performance summary
        print("\n--- Summary Statistics ---")
        metrics = [
            "BERTScore", "Contradiction", "Lateral_Error", "Clinical_Score"
        ]
        
        summary_data = []
        for metric in metrics:
            base_val = df[f"base_{metric}"].mean()
            ft_val = df[f"ft_{metric}"].mean()
            summary_data.append({
                "Metric": metric,
                "Base": base_val,
                "FT": ft_val,
                "Delta": ft_val - base_val
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        summary_path = os.path.join(args.input_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary statistics saved to: {summary_path}")

if __name__ == "__main__":
    main()