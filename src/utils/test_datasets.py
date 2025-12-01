import os
import pandas as pd
import json
import cv2
import h5py
import numpy as np
from pathlib import Path

def run_day2_verification():
    print("🧪 Starting Day 2 Verification...\n")
    
    # 1. Check Directory Structure
    paths = [
        "data/processed/videos",
        "data/processed/traffic_maps",
        "configs"
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"✅ Found directory: {p}")
        else:
            print(f"❌ MISSING directory: {p}")
            return

    # 2. Verify Metadata CSV
    meta_path = "data/processed/dataset_metadata.csv"
    if not os.path.exists(meta_path):
        print("❌ MISSING Metadata CSV")
        return
        
    df = pd.read_csv(meta_path)
    print(f"✅ Metadata loaded: {len(df)} records found.")
    
    # 3. Verify Splits JSON
    split_path = "configs/data_splits.json"
    if not os.path.exists(split_path):
        print("❌ MISSING Splits JSON")
        return
        
    with open(split_path, 'r') as f:
        splits = json.load(f)
    print(f"✅ Splits loaded: Found keys {list(splits.keys())}")

    # 4. Test Video Loading (CityFlow)
    cf_sample = df[df['dataset_source'] == 'CityFlow'].iloc[0]
    vid_path = Path("data/processed/videos") / cf_sample['filename']
    
    cap = cv2.VideoCapture(str(vid_path))
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame.shape == (1080, 1920, 3):
            print(f"✅ Video Read Test: Success ({cf_sample['filename']} is 1080p)")
        else:
            print(f"⚠️ Video Read Warning: Read successful but unexpected shape {frame.shape}")
    else:
        print(f"❌ Failed to open video: {vid_path}")
    cap.release()

    # 5. Test Traffic Map Loading (Traffic4Cast)
    t4c_sample = df[df['dataset_source'] == 'Traffic4Cast'].iloc[0]
    h5_path = Path("data/processed/traffic_maps") / t4c_sample['filename']
    
    try:
        with h5py.File(h5_path, 'r') as f:
            key = list(f.keys())[0]
            data = f[key]
            if data.shape == (288, 495, 436, 8):
                print(f"✅ H5 Read Test: Success ({t4c_sample['filename']} is 8-channel)")
            else:
                print(f"❌ H5 Read Error: Unexpected shape {data.shape}")
    except Exception as e:
        print(f"❌ Failed to read H5: {e}")

    print("\n" + "="*30)
    print("   DAY 2 COMPLETE ✓   ")
    print("="*30)

if __name__ == "__main__":
    run_day2_verification()