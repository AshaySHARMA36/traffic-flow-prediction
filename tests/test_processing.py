import torch
import cv2
import time
import os
import sys
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is in python path
sys.path.append(os.getcwd())

from src.data.dataset import TrafficVideoDataset
from src.utils.augmentations import VideoAugmentor
from src.utils.video_processor import VideoProcessor

def save_batch_visual(frames, save_path="experiments/eda_results/batch_preview.png"):
    """
    Saves a visualization of a training batch.
    frames: (B, T, C, H, W)
    """
    # Take first sample in batch, first 5 frames
    sample = frames[0] # (T, C, H, W)
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        # Convert Tensor (C, H, W) -> Numpy (H, W, C)
        img = sample[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"Frame {i}")
    
    plt.suptitle(f"Training Batch Sample (Shape: {tuple(frames.shape)})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_integration_test():
    print("🔄 Starting Preprocessing Integration Test...")
    
    # 1. Load Config
    config_path = "configs/preprocessing_config.yaml"
    if not os.path.exists(config_path):
        print(f"⚠️ Config not found at {config_path}. Using defaults.")
        config = {'dataloader': {'batch_size': 4, 'num_workers': 0}}
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # 2. Setup Components
    metadata_path = "data/processed/dataset_metadata.csv"
    root_dir = "data/processed/videos_224" # Use the optimized data
    
    # 3. Initialize Augmentor
    augmentor = VideoAugmentor(target_size=(224, 224))
    
    # 4. Initialize Dataset
    dataset = TrafficVideoDataset(
        metadata_file=metadata_path,
        split='train',
        root_dir=root_dir,
        sequence_length=16,
        transform=augmentor
    )
    
    print(f"✅ Dataset Initialized. Size: {len(dataset)}")
    
    # 5. Initialize DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        shuffle=True
    )
    
    # 6. Run Batch Loading Loop
    start_time = time.time()
    batches_to_test = 5
    
    print(f"\n🚀 Processing {batches_to_test} batches...")
    
    for i, (frames, labels) in enumerate(loader):
        if i >= batches_to_test: break
        
        # Verify Shapes
        B, T, C, H, W = frames.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == 224 and W == 224, f"Expected 224x224, got {H}x{W}"
        assert T == 16, f"Expected 16 frames, got {T}"
        
        print(f"   Batch {i+1}: Shape {tuple(frames.shape)} | Label {tuple(labels.shape)}")
        
        if i == 0:
            save_batch_visual(frames)
            print("   📸 Saved batch visualization to experiments/eda_results/batch_preview.png")

    end_time = time.time()
    duration = end_time - start_time
    fps = (batches_to_test * config['dataloader']['batch_size'] * 16) / duration
    
    print("\n📊 Performance Stats:")
    print(f"   Total Time: {duration:.2f}s")
    print(f"   Throughput: {fps:.2f} frames/sec")
    
    if fps > 100:
        print("✅ SPEED: Excellent (>100 FPS)")
    elif fps > 50:
        print("✅ SPEED: Good (>50 FPS)")
    else:
        print("⚠️ SPEED: Warning (<50 FPS). Check bottleneck.")

    print("\n✅ Day 3 Integration Test PASSED.")

if __name__ == "__main__":
    run_integration_test()