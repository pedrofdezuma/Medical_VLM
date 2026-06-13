import os
import csv
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIR = "D:/combined-test-4k-PT-NR-2"
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation_results.csv")
VIEW_MAPPING_CSV = "D:/mri_view_mapping.csv"

# Dictionary to standardize sequence names
abbreviations = {
    "t1-w": {"t1", "t1n", "t1w"},
    "t1-ce": {"t1c", "t1ce", "t1gd"},
    "t2-w": {"t2w", "t2"},
    "flair": {"t2f", "flair", "t2flair"}
}

def load_view_mapping(filepath: str) -> dict:
    """
    Reads the mapping CSV (delimiter ';') and creates a nested dictionary.
    Format: {'brats-goat': {'x': 'sagittal', 'y': 'coronal', 'z': 'axial'}, ...}
    """
    mapping = {}
    if not os.path.exists(filepath):
        print(f"[WARNING] Mapping file '{filepath}' not found.")
        return mapping
        
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            dataset_name = row['dataset'].strip().lower()
            mapping[dataset_name] = {
                'x': row['x'].strip().lower() if 'x' in row and row['x'] else 'n/a',
                'y': row['y'].strip().lower() if 'y' in row and row['y'] else 'n/a',
                'z': row['z'].strip().lower() if 'z' in row and row['z'] else 'n/a'
            }
    return mapping

def get_sequence(filename: str) -> str:
    """
    Extracts the raw MRI sequence using regex, then maps it 
    to a standardized category using the abbreviations dictionary.
    """
    raw_sequence = None
    
    # Try to match the MR pattern (e.g., mr_adc--ISLES2022...)
    match_mr = re.search(r'^mr_([a-z0-9]+)--', filename)
    if match_mr:
        raw_sequence = match_mr.group(1)
    else:
        # Try to match the BraTS pattern (e.g., brats-goat-00007_t1c_slice...)
        match_brats = re.search(r'^[a-z0-9-]+_([a-z0-9]+)_', filename)
        if match_brats:
            raw_sequence = match_brats.group(1)

    # Standardize the sequence using the dictionary
    if raw_sequence:
        raw_sequence = raw_sequence.lower()
        for standard_key, variants in abbreviations.items():
            if raw_sequence in variants:
                return standard_key
                
    return "others"

def get_view_indicator(filename: str) -> str:
    """
    Extracts a single letter surrounded by delimiters.
    Matches cases like '_y_', '--y_', or '_y.png'
    """
    match = re.search(r'(?:_|--)([a-z])(?:_|\.)', filename.lower())
    if match:
        return match.group(1)
    return None

def get_dataset(filename):
    """
    Extracts the dataset origin from a filename, ignoring sequences or slice IDs.
    All comments are in English as requested.
    """
    fn_lower = filename.lower()

    # Case 1: MedTrinity/ISLES format (e.g., mr_t1c--braintumour--...)
    # The dataset name is always located between the first and second double dash
    if  "--" in fn_lower:
        parts = fn_lower.split("--")
        if len(parts) > 1:
            return parts[1] # Returns 'braintumour', 'isles2016', etc.

    # Case 2: Standard BraTS prefix format (e.g., brats-goat-001_...)
    # We look for the 'brats-word' pattern at the start of the string
    brats_match = re.match(r"^(brats-[a-z]+)", fn_lower)
    if brats_match:
        return brats_match.group(1) # Returns 'brats-goat', 'brats-gli', etc.

    return None

def generate_evaluation_csv():
    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] Directory '{BASE_DIR}' not found.")
        return

    # Load the mapping dictionary into memory once
    view_map = load_view_mapping(VIEW_MAPPING_CSV)

    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'predicted_view', 'real_view', 'predicted_sequence', 'real_sequence'])
        
        file_count = 0
        
        for root, dirs, files in os.walk(BASE_DIR):
            for filename in files:
                if filename == "evaluation_results.csv" or filename.startswith('.'):
                    continue
                
                # Extract predictions from the folder structure
                relative_path = os.path.relpath(root, BASE_DIR)
                path_parts = relative_path.split(os.sep)
                
                # We expect exactly 2 levels of folders: view / sequence
                if len(path_parts) >= 2:
                    predicted_view = path_parts[0]
                    predicted_sequence = path_parts[1]
                else:
                    continue 

                # Extract real labels
                real_sequence = get_sequence(filename)
                indicator = get_view_indicator(filename)
                ds=get_dataset(filename)
                

                real_view = "non_brain_mri"
                
                # If we found the dataset and the letter, lookup the true view
                if ds and indicator:
                    try:
                        mapped_view = view_map[ds].get(indicator, "non_brain_mri")
                    except Exception as e:
                        mapped_view="non_brain_mri"
                        
                    if mapped_view.lower() != "n/a":
                        real_view = mapped_view

                writer.writerow([filename, predicted_view, real_view, predicted_sequence, real_sequence])
                file_count += 1
                
    print(f"\n--- Process Completed ---")
    print(f"Total files processed: {file_count}")
    print(f"CSV saved at: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_evaluation_csv()
