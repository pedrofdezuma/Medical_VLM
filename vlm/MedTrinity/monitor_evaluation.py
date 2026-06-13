import json
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_pointwise(file_path):
    results = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                continue
    
    if not results:
        print("No results found yet.")
        return

    data = []
    for r in results:
        case_id = r.get("case_id")
        eval_data = r.get("evaluation", {})
        base = eval_data.get("base_model", {})
        ft = eval_data.get("ft_model", {})
        
        data.append({
            "case_id": case_id,
            "base_score": base.get("score_1_to_5"),
            "ft_score": ft.get("score_1_to_5"),
            "base_hallucination": base.get("critical_hallucination", False),
            "ft_hallucination": ft.get("critical_hallucination", False),
            "base_redundancy": base.get("redundancy_level", False),
            "ft_redundancy": ft.get("redundancy_level", False)
        })
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("      POINTWISE EVALUATION PROGRESS SUMMARY")
    print("="*50)
    print(f"Total cases evaluated: {len(df)}")
    print("-" * 50)
    print(f"Average Score (1-5):")
    print(f"  Base Model: {df['base_score'].mean():.3f}")
    print(f"  FT Model:   {df['ft_score'].mean():.3f}")
    print(f"  Improvement: {df['ft_score'].mean() - df['base_score'].mean():+.3f}")
    print("-" * 50)
    print(f"Failure Rates (%):")
    print(f"  Hallucinations: Base {df['base_hallucination'].mean()*100:.1f}% | FT {df['ft_hallucination'].mean()*100:.1f}%")
    print(f"  Redundancy:     Base {df['base_redundancy'].mean()*100:.1f}% | FT {df['ft_redundancy'].mean()*100:.1f}%")
    print("="*50 + "\n")

    return df

def analyze_pairwise(file_path):
    results = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                continue
    
    if not results:
        print("No results found yet.")
        return

    data = []
    for r in results:
        p = r.get("pairwise", {})
        data.append({
            "case_id": r.get("case_id"),
            "score": p.get("score"),
            "consistent": p.get("consistent", False)
        })
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("      PAIRWISE EVALUATION PROGRESS SUMMARY")
    print("="*50)
    print(f"Total cases evaluated: {len(df)}")
    print("-" * 50)
    print(f"Average Relative Score (-1 to 1): {df['score'].mean():.4f}")
    print(f"  (Positive favors FT Model, Negative favors Base)")
    print(f"Consistency Rate: {df['consistent'].mean()*100:.1f}%")
    print("-" * 50)
    
    win_rate_ft = (df['score'] > 0).mean() * 100
    win_rate_base = (df['score'] < 0).mean() * 100
    tie_rate = (df['score'] == 0).mean() * 100
    
    print(f"Win Rates:")
    print(f"  FT Model Wins:   {win_rate_ft:.1f}%")
    print(f"  Base Model Wins: {win_rate_base:.1f}%")
    print(f"  Ties:            {tie_rate:.1f}%")
    print("="*50 + "\n")

    return df

def main():
    parser = argparse.ArgumentParser(description="Monitor evaluation progress from jsonl files.")
    parser.add_argument("file", help="Path to the .jsonl evaluation file")
    parser.add_argument("--loop", action="store_true", help="Continuously monitor the file")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds for loop mode")
    
    args = parser.parse_args()
    
    while True:
        # Detect type of evaluation
        with open(args.file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                print("File is empty.")
                if not args.loop: break
                time.sleep(args.interval)
                continue
            
            try:
                sample = json.loads(first_line)
                if "evaluation" in sample:
                    analyze_pointwise(args.file)
                elif "pairwise" in sample:
                    analyze_pairwise(args.file)
                else:
                    print("Unknown evaluation format.")
            except:
                print("Error parsing JSON.")
        
        if not args.loop:
            break
        
        print(f"Next update in {args.interval}s... (Ctrl+C to stop)")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
