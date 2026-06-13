"""
BERTScore evaluation (PubMedBERT) to compare:
  - Base model vs. Ground Truth
  - Fine-tuned model vs. Ground Truth
  - Fine-tuned model vs. Base model

Requirements:
    pip install bert-score transformers pandas
"""

from bert_score import score
import pandas as pd

# ─────────────────────────────────────────────
# 1. EXAMPLE TEXTS — replace with your own
# ─────────────────────────────────────────────

import torch
import transformers
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Ruta de PyTorch: {torch.__file__}")


ground_truth = """
Brain MRI study with axial FLAIR sequences. Multiple hyperintense lesions are identified
in the periventricular and subcortical white matter, predominantly bilateral and asymmetric,
compatible with moderate leukoaraiosis. No space-occupying lesions, active bleeding, or
pathological enhancement after contrast are observed. Midline is centered. Ventricular
system is normal in size for the patient's age. Conclusion: findings suggestive of moderate
small vessel disease.
"""

response_base = """
The brain MRI shows some bright areas in the white matter. There is no evidence of tumors
or bleeding. The brain appears generally normal. Midline structures are centered.
The ventricles are not dilated.
"""

response_finetuned = """
Brain MRI with FLAIR sequences. Hyperintense foci are observed in the periventricular white
matter with bilateral distribution, compatible with moderate leukoaraiosis. No mass-effect
lesions or intracranial bleeding are identified. Midline structures show no deviation.
Ventricles are preserved in morphology and size. Diagnostic impression: chronic ischemic
changes from small vessel disease, moderate grade.
"""

# ─────────────────────────────────────────────
# 2. MODEL — PubMedBERT
# ─────────────────────────────────────────────

MODEL  = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# Valid alternatives:
# "allenai/scibert_scivocab_uncased"
# "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ─────────────────────────────────────────────
# 3. BERTSCORE CALCULATION
# ─────────────────────────────────────────────

def compute_bertscore(candidates: list[str], references: list[str], label: str) -> dict:
    """Computes BERTScore and returns a dict with Precision, Recall, F1."""
    print(f"\n⏳ Computing BERTScore: {label}...")
    
    # Use PubMedBERT (ensure you use the version that supports safetensors if possible)
    # or the one you currently have, but specifying num_layers is MANDATORY.
    P, R, F1 = score(
        cands=candidates,
        refs=references,
        model_type=MODEL,
        num_layers=9,                # Fixes the KeyError
        lang="en",
        verbose=False,
        rescale_with_baseline=False, # Custom models don't have baseline files
    )
    return {
        "Comparison": label,
        "Precision":  round(P.mean().item(), 4),
        "Recall":     round(R.mean().item(), 4),
        "F1":         round(F1.mean().item(), 4),
    }


results = []

# Base model vs. Ground Truth
results.append(compute_bertscore(
    candidates=[response_base],
    references=[ground_truth],
    label="Base → Ground Truth"
))

# Fine-tuned model vs. Ground Truth
results.append(compute_bertscore(
    candidates=[response_finetuned],
    references=[ground_truth],
    label="Fine-tuned → Ground Truth"
))

# Fine-tuned vs. Base (relative improvement)
results.append(compute_bertscore(
    candidates=[response_finetuned],
    references=[response_base],
    label="Fine-tuned → Base"
))

# ─────────────────────────────────────────────
# 4. DISPLAY RESULTS
# ─────────────────────────────────────────────

df = pd.DataFrame(results)

print("\n" + "="*60)
print("         BERTSCORE RESULTS (PubMedBERT)")
print("="*60)
print(df.to_string(index=False))
print("="*60)

# Fine-tuned improvement over base model
f1_base      = df[df["Comparison"] == "Base → Ground Truth"]["F1"].values[0]
f1_finetuned = df[df["Comparison"] == "Fine-tuned → Ground Truth"]["F1"].values[0]
improvement  = round((f1_finetuned - f1_base) * 100, 2)

print(f"\n📊 Fine-tuned F1 improvement over base: {improvement:+.2f} pp")

if improvement > 0:
    print("✅ Fine-tuning improves report generation quality.")
elif improvement == 0:
    print("➖ No appreciable difference between models.")
else:
    print("⚠️  Fine-tuning reduces quality compared to the base model.")

# ─────────────────────────────────────────────
# 5. EXPORT TO CSV (optional)
# ─────────────────────────────────────────────

df.to_csv("bertscore_results.csv", index=False)
print("\n💾 Results saved to: bertscore_results.csv")

