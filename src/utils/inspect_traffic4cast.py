import h5py
import numpy as np
import os

def inspect_h5(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"\n🔍 Inspecting Traffic4Cast File: {os.path.basename(file_path)}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Print all root keys in the file
            keys = list(f.keys())
            print(f"   📂 Keys found: {keys}")
            
            # Usually the main data is under a key like 'array' or 'data'
            # We grab the first key available
            main_key = keys[0]
            data = f[main_key]
            
            print(f"   📐 Tensor Shape: {data.shape}")
            print(f"   🔢 Data Type: {data.dtype}")
            
            # shape is usually (Time, Height, Width, Channels) or (Time, Channels, H, W)
            # Let's verify the range of values (e.g., 0-255 or 0-1)
            sample = data[0]
            print(f"   📊 Value Range: min={np.min(sample):.2f}, max={np.max(sample):.2f}")
            
            if data.shape[-1] == 8:
                print("   ✅ Confirmed: 8-Channel Traffic Movie (Volume/Speed)")
            else:
                print(f"   ℹ️ Channels: {data.shape[-1]}")

    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")

if __name__ == "__main__":
    # REPLACE THIS PATH with the actual path to one of your London .h5 files
    # Example: "D:/traffic_prediction/london/2019-07-02_london_8ch.h5"
    FILE_PATH = r"D:\traffic-flow-prediction\london\2019-07-02_london_8ch.h5" 
    inspect_h5(FILE_PATH)