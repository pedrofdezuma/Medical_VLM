from huggingface_hub import HfApi

# Initialize the Hugging Face API client
api = HfApi()

# TODO: Replace with your actual username and the NEW repository name
REPO_ID = r"pedrofdez/classifiers_view_and_sequence"

print("Starting the upload of the 6 neural network weights...")

# Upload the entire folder containing the 6 .pth files
api.upload_folder(
    folder_path=r"D:\classifiers\best_models",  # Path to the folder with the 6 files
    repo_id=REPO_ID,
    repo_type="model"
)

# Function to clear string literals with backslashes if needed:
# folder_path = r"C:\ruta\de\tu\carpeta\mis_redes"

print("All 2 model weights have been uploaded successfully!")