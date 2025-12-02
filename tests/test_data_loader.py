import time
import torch
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.data.data_loader import create_dataloaders

def test_dataloader_speed():
    print("🚀 Testing DataLoader Performance...")
    
    # Config
    METADATA = "data/processed/dataset_metadata.csv"
    ROOT_DIR = "data/processed/videos_224"
    BATCH_SIZE = 2 # Small batch for testing
    
    if not os.path.exists(METADATA):
        print("❌ Metadata not found. Run Day 2 setup first.")
        return

    # Create Loaders
    train_loader, _ = create_dataloaders(METADATA, ROOT_DIR, batch_size=BATCH_SIZE, num_workers=0) # Use 0 workers for debugging stability
    
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Dataset Size: {len(train_loader.dataset)}")
    
    start_time = time.time()
    num_batches = 0
    total_frames = 0
    
    # Iterate through 5 batches to warm up and measure
    try:
        for batch_idx, (frames, labels) in enumerate(tqdm(train_loader, desc="Loading Batches")):
            if batch_idx >= 5: break
            
            # Simulate GPU transfer
            # frames = frames.cuda() 
            
            num_batches += 1
            total_frames += frames.size(0) * frames.size(1) # B * T
            
            # Verify shapes
            if batch_idx == 0:
                print(f"\n✅ Batch Shape: {frames.shape}") # Should be [B, 16, 3, 224, 224]
                print(f"✅ Label Shape: {labels.shape}")
                
    except Exception as e:
        print(f"\n❌ Error during loading: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time
    
    if duration > 0:
        fps = total_frames / duration
        print(f"\n📊 Performance Report:")
        print(f"   Time taken: {duration:.2f}s")
        print(f"   Throughput: {fps:.2f} frames/sec")
        
        if fps < 10:
            print("⚠️ WARNING: Loading is slow. Check disk I/O or reduce video resolution.")
        else:
            print("✅ Speed looks good for training.")

if __name__ == "__main__":
    test_dataloader_speed()