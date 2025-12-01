import os
import shutil
import cv2
import pandas as pd
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm
import datetime

def get_video_metadata(file_path):
    """Extracts resolution, FPS, duration, and frame count from video files."""
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        "resolution": f"{width}x{height}",
        "fps": fps,
        "duration_sec": duration,
        "frame_count": frame_count,
        "valid": True
    }

def get_h5_metadata(file_path):
    """Extracts shape and keys from H5 files."""
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            # Assuming the first key holds the data
            data_shape = f[keys[0]].shape
            return {
                "resolution": f"{data_shape[-3]}x{data_shape[-2]}", # Spatial dim
                "fps": 0, # Not applicable
                "duration_sec": 0,
                "frame_count": data_shape[0], # Time dim
                "valid": True
            }
    except Exception:
        return None

def organize_datasets(cityflow_root, traffic4cast_root, output_dir):
    # Setup directories
    processed_dir = Path(output_dir)
    videos_out = processed_dir / "videos"
    maps_out = processed_dir / "traffic_maps"
    
    videos_out.mkdir(parents=True, exist_ok=True)
    maps_out.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    print(f"🚀 Starting Data Organization...")
    print(f"📂 Output Directory: {processed_dir}")

    # --- PROCESS CITYFLOW ---
    if cityflow_root and os.path.exists(cityflow_root):
        print("\n🔵 Processing CityFlow (renaming vdo.avi -> cityflow_XXX.avi)...")
        cf_path = Path(cityflow_root)
        # Recursive search for vdo.avi based on your structure
        cf_files = list(cf_path.rglob("vdo.avi"))
        
        for i, file_path in enumerate(tqdm(cf_files, desc="CityFlow Files")):
            # Generate new name
            new_filename = f"cityflow_{i+1:03d}.avi"
            target_path = videos_out / new_filename
            
            # Copy file
            shutil.copy2(file_path, target_path)
            
            # Extract Metadata
            meta = get_video_metadata(target_path)
            file_stats = os.stat(target_path)
            
            record = {
                "filename": new_filename,
                "original_path": str(file_path),
                "dataset_source": "CityFlow",
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "status": "OK" if meta else "CORRUPT"
            }
            if meta:
                record.update(meta)
            else:
                record.update({"valid": False})
                
            metadata.append(record)
    else:
        print(f"⚠️ CityFlow path not found: {cityflow_root}")

    # --- PROCESS TRAFFIC4CAST ---
    if traffic4cast_root and os.path.exists(traffic4cast_root):
        print("\n🟢 Processing Traffic4Cast (renaming -> traffic4cast_XXX.h5)...")
        t4c_path = Path(traffic4cast_root)
        t4c_files = list(t4c_path.glob("*.h5"))
        
        for i, file_path in enumerate(tqdm(t4c_files, desc="Traffic4Cast Files")):
            # Generate new name
            new_filename = f"traffic4cast_{i+1:03d}.h5"
            target_path = maps_out / new_filename
            
            # Copy file
            shutil.copy2(file_path, target_path)
            
            # Extract Metadata
            meta = get_h5_metadata(target_path)
            file_stats = os.stat(target_path)
            
            record = {
                "filename": new_filename,
                "original_path": str(file_path),
                "dataset_source": "Traffic4Cast",
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "status": "OK" if meta else "CORRUPT"
            }
            if meta:
                record.update(meta)
            else:
                record.update({"valid": False})
                
            metadata.append(record)
    else:
        print(f"⚠️ Traffic4Cast path not found: {traffic4cast_root}")

    # --- SAVE METADATA CSV ---
    df = pd.DataFrame(metadata)
    csv_path = processed_dir / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Organization Complete!")
    print(f"📄 Metadata saved to: {csv_path}")
    print(f"📊 Summary:\n{df['dataset_source'].value_counts()}")

if __name__ == "__main__":
    # Hardcoded paths based on your previous messages
    # You can edit these if paths change
    CITYFLOW_PATH = r"D:\traffic-flow-prediction\AICity22_Track1_MTMC_Tracking\train"
    TRAFFIC4CAST_PATH = r"D:\traffic-flow-prediction\london"
    OUTPUT_PATH = r"data\processed"
    
    organize_datasets(CITYFLOW_PATH, TRAFFIC4CAST_PATH, OUTPUT_PATH)