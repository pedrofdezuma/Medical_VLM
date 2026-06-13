from huggingface_hub import snapshot_download

# Use the exact ID of MedGemma
repo_id = "google/medgemma-1.5-4b-it" # or medgemma-7b

# THE IMPORTANT PART: Your specific path in Picasso or local
local_dir = "D:/modelos/MedGemma/medgemma1.5"

print(f"Downloading model to: {local_dir}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Essential to get REAL files, not links
    token=True                    # Uses your HF_TOKEN
)

print("Done! All files are now in your directory.")