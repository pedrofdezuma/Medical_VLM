import os
import pickle
import pathlib
from tqdm import tqdm

def list_shard_files(root_dir, output_dir):
    """
    Iterate through shard_X folders in root_dir, collect image filenames, 
    and save them into individual .pkl files.
    """
    root_path = pathlib.Path(root_dir)
    output_path = pathlib.Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all directories matching shard_*
    shard_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('shard_')])
    
    if not shard_dirs:
        print(f"No shard folders found in {root_dir}")
        return

    # Image extensions to filter
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    
    print(f"Found {len(shard_dirs)} shards. Processing...")
    
    for shard_dir in tqdm(shard_dirs, desc="Processing shards"):
        # Collect all image filenames in the shard
        # filenames = [f.name for f in shard_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        # Actually, user might want the full relative path or just the filename. 
        # "all the filenames" usually means f.name, but let's stick to what was asked.
        filenames = [f.name for f in shard_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not filenames:
            print(f"No images found in {shard_dir.name}, skipping.")
            continue
            
        # Save to pickle
        pkl_filename = f"{shard_dir.name}_files.pkl"
        pkl_path = output_path / pkl_filename
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(filenames, f)
            
    print(f"Finished processing. Shard file lists saved to {output_dir}")

if __name__ == "__main__":
    # Base directory for the axial-flair dataset
    ROOT_DATASET = r"D:\axial-flair-dataset"
    
    # Output directory for the pickle files (relative to this script)
    # We'll place it in vlm/MedTrinity/data/shard_lists
    script_dir = pathlib.Path(__file__).parent
    OUTPUT_DIR = script_dir / "data" / "shard_lists"
    
    list_shard_files(ROOT_DATASET, OUTPUT_DIR)
