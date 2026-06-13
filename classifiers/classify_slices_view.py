import os
import shutil
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ================= CONFIGURATION =================
# We will iterate over these models
#MODELS_TO_RUN = ["resnet18", "vgg16", "mobilenet_v2"]
MODELS_TO_RUN=["resnet18"]

BASE_CHECKPOINT_DIR = "D:/classifiers/classifier_view/checkpoints"
INPUT_DIR = "C:/Users/pedro/Desktop/Investigacion/slices_esclerosis"
BASE_OUTPUT_DIR = "C:/Users/pedro/Desktop/Investigacion/classified_slices9"

VIEWS_TO_PROCESS = ['axial', 'coronal', 'sagital']
NUM_PATIENTS_PER_VIEW = 20
CLASSES = ['axial', 'coronal', 'sagittal', 'non_brain_mri']

# Image transformations (same as in validation/test)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

USE_TTA = False # Set to True to use predict_view_with_tta, False to use simple inference
ROTATION=False
# =================================================

def predict_view_with_tta(model, image, preprocess, device, class_names=CLASSES, verbose=False):
    """
    Performs inference on an image and its 4 rotations (0, 90, 180, 270).
    Returns the prediction with the highest confidence score.
    """
    model.eval()
    if isinstance(image, str):
      img_orig = Image.open(image).convert('RGB')
    else:
      img_orig = image

    rotations = [0, 90, 180, 270]
    results = []
    
    with torch.no_grad():
        for angle in rotations:
            # 1. Rotate the PIL image
            # expand=True ensures the image isn't cropped during rotation
            img_rotated = img_orig.rotate(angle, expand=True)
            
            # 2. Preprocess and move to device
            input_tensor = preprocess(img_rotated).unsqueeze(0).to(device)
            
            # 3. Get Logits and Probabilities
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
            # 4. Get the top prediction for this specific rotation
            conf, pred_idx = torch.max(probs, 1)
            label = class_names[pred_idx.item()]
            
            results.append({
                'angle': angle,
                'label': label,
                'confidence': conf.item(),
                'logits': logits.cpu().numpy()
            })

    # 5. Find the winner (the one with the maximum confidence)
    best_result = max(results, key=lambda x: x['confidence'])
    return best_result

def predict_view_simple(model, image, preprocess, device, class_names=CLASSES):
    """
    Performs inference directly on the input image without Test-Time Augmentation.
    """
    model.eval()
    if isinstance(image, str):
      img_orig = Image.open(image).convert('RGB')
    else:
      img_orig = image

    with torch.no_grad():
        # Preprocess and move to device
        input_tensor = preprocess(img_orig).unsqueeze(0).to(device)
        
        # Get Logits and Probabilities
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        
        # Get the top prediction
        conf, pred_idx = torch.max(probs, 1)
        label = class_names[pred_idx.item()]
        
        return {
            'label': label,
            'confidence': conf.item(),
            'logits': logits.cpu().numpy()
        }

def load_mri_model(model_name, checkpoint_path, device):
    """Loads the specified architecture and its weights."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, len(CLASSES))
        )
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, len(CLASSES))
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, len(CLASSES))
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def get_middle_20_percent_slices(slice_paths):
    """Sorts slices numerically and returns only the middle 20%."""
    def extract_number(path):
        filename = os.path.basename(path) # e.g.: slice_005.jpg
        name = os.path.splitext(filename)[0] # e.g.: slice_005
        try:
            return int(name.split('_')[-1]) # Extracts the 5
        except ValueError:
            return 0
            
    # Sort ensuring slice_2 comes before slice_10
    sorted_paths = sorted(slice_paths, key=extract_number)
    
    total = len(sorted_paths)
    if total == 0:
        return []
        
    start_idx = int(0.40 * total)
    end_idx = total - int(0.40 * total)
    
    return sorted_paths[start_idx:end_idx]

def process_model(model_name, device):
    print(f"\n{'='*50}")
    print(f"STARTING PROCESS FOR MODEL: {model_name}")
    print(f"{'='*50}")
    
    # Define paths for this specific model
    checkpoint_path = os.path.join(BASE_CHECKPOINT_DIR, f"{model_name}-4k-PT-NR", "best_mri_classifier.pth")
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    csv_output_path = os.path.join(output_dir, "classification_results.csv")
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        print(f"Skipping model {model_name}...")
        return
        
    print(f"Loading the {model_name} model...")
    try:
        model = load_mri_model(model_name, checkpoint_path, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        return
    
    # Clear the output directory if it exists to remove previous images
    if os.path.exists(output_dir):
        print(f"Borrando imágenes previas en {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)
        
    # Create output directories specific to the model
    os.makedirs(output_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)
        
    # Open CSV file to log the results
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['filename', 'true_view', 'predicted_view'])
    
        for view in VIEWS_TO_PROCESS:
            view_dir = os.path.join(INPUT_DIR, view)
            if not os.path.isdir(view_dir):
                print(f"[WARNING] View directory '{view_dir}' not found. Skipping...")
                continue
                
            print(f"\n--- Processing view directory: {view} with {model_name} ---")
            
            # Get patient folders and ensure they are sorted/selected predictably
            patient_folders = [d for d in os.listdir(view_dir) if os.path.isdir(os.path.join(view_dir, d)) and d.startswith('patient_')]
            
            # Sort folders by patient number
            patient_folders.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
            
            # Take only the first NUM_PATIENTS_PER_VIEW
            patient_folders = patient_folders[:NUM_PATIENTS_PER_VIEW]
            
            for patient in patient_folders:
                patient_dir = os.path.join(view_dir, patient)
                print(f"  Processing patient: {patient}...")
                
                # Look for T1 directory
                t1_dir = os.path.join(patient_dir, 'T1')
                if not os.path.isdir(t1_dir):
                    print(f"    [!] T1 not found in {patient_dir}")
                    continue
                    
                # Look for normalizado directory
                norm_dir = os.path.join(t1_dir, 'normalizado')
                if not os.path.isdir(norm_dir):
                    print(f"    [!] 'normalizado' not found in {t1_dir}")
                    continue
                    
                # Extract valid images
                valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                all_slices = [os.path.join(norm_dir, f) for f in os.listdir(norm_dir) 
                              if f.lower().endswith(valid_exts) and 'slice_' in f.lower()]
                
                if not all_slices:
                    print(f"    [!] No valid slices found in {norm_dir}")
                    continue
                    
                # Filter only the central 20%
                central_slices = get_middle_20_percent_slices(all_slices)
                
                for slice_path in central_slices:
                    try:
                        img = Image.open(slice_path).convert('RGB')

                        if ROTATION:
                            if view in ['sagital', 'coronal','axial']:
                                img = img.transpose(Image.Transpose.ROTATE_90)

                        if USE_TTA:
                            # Use the Test-Time Augmentation function, passing the PIL image object
                            best_result = predict_view_with_tta(model, img, TRANSFORM, device, class_names=CLASSES)
                        else:
                            # Use simple inference directly on the input image
                            best_result = predict_view_simple(model, img, TRANSFORM, device, class_names=CLASSES)
                            
                        predicted_class_name = best_result['label']

                        # Define destination for the image
                        filename = os.path.basename(slice_path)
                        new_filename = f"{view}_{patient}_{filename}"
                        dest_path = os.path.join(output_dir, predicted_class_name, new_filename)

                        # Save the (potentially rotated) image
                        img.save(dest_path)

                        # Log to CSV
                        csv_writer.writerow([new_filename, view, predicted_class_name])
                        
                    except Exception as e:
                        print(f"    [ERROR] Failed to process {slice_path}: {e}")

    print(f"\n[OK] Finished model {model_name}. Images at: {output_dir}")
    print(f"Results saved in CSV at: {csv_output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for model_name in MODELS_TO_RUN:
        process_model(model_name, device)
        
    print("\n" + "="*50)
    print("ALL MODELS HAVE BEEN PROCESSED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()