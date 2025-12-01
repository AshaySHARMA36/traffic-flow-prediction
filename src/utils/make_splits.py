import os
import json
import pandas as pd
from pathlib import Path

def create_splits(metadata_path, output_dir):
    """
    Generates train/val/test split indices using Temporal Splitting.
    """
    df = pd.read_csv(metadata_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    
    for source in df['dataset_source'].unique():
        # Get all files for this source
        subset = df[df['dataset_source'] == source].sort_values('filename')
        files = subset['filename'].tolist()
        
        n = len(files)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        splits[source] = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }
        
        print(f"\n🔹 {source} Split:")
        print(f"   Train: {len(splits[source]['train'])} files")
        print(f"   Val:   {len(splits[source]['val'])} files")
        print(f"   Test:  {len(splits[source]['test'])} files")

    # Save to JSON for the Data Loader to use later
    save_path = output_path / "data_splits.json"
    with open(save_path, 'w') as f:
        json.dump(splits, f, indent=4)
    
    print(f"\n✅ Splits saved to: {save_path}")

if __name__ == "__main__":
    METADATA_FILE = r"data/processed/dataset_metadata.csv"
    OUTPUT_DIR = r"configs"
    
    if os.path.exists(METADATA_FILE):
        create_splits(METADATA_FILE, OUTPUT_DIR)
    else:
        print("❌ Metadata file not found. Run organize_datasets.py first.")