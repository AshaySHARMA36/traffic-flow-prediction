import cv2
import numpy as np
import albumentations as A
import random
from typing import List, Dict, Optional, Tuple

class VideoAugmentor:
    """
    Applies consistent spatial and temporal augmentations to video sequences.
    Ensures that random spatial transforms (rotation, crop) are identical across all frames in a clip.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.height, self.width = target_size
        
        # Define the pool of spatial augmentations
        # We use ReplayCompose to apply the same random parameters to all frames
        self.spatial_transform = A.ReplayCompose([
            # Geometry (Conservative)
            A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5), # Valid for most roads, but be careful with lane-specific logic
            A.RandomResizedCrop(
                height=self.height, width=self.width, 
                scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5
            ),
            
            # Color/Lighting (Aggressive for Day/Night simulation)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.7),
            
            # Noise/Quality
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.3),
        ])

    def add_rain_effect(self, frames: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Simulates rain using simple white streaks overlay."""
        if random.random() > intensity: return frames
        
        seq_len, h, w, c = frames.shape
        augmented = []
        
        # Generate rain layer once or per frame? 
        # Per frame looks like static noise. We need 'moving' rain.
        # Simple approximation: Random distinct lines per frame.
        for frame in frames:
            layer = frame.copy()
            num_drops = int(500 * intensity)
            
            for _ in range(num_drops):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                # Draw small slant line
                cv2.line(layer, (x, y), (x+1, y+3), (200, 200, 200), 1)
                
            # Blend
            augmented.append(cv2.addWeighted(frame, 0.8, layer, 0.2, 0))
            
        return np.stack(augmented)

    def simulate_night(self, frames: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Darkens the image and adds ISO noise to simulate night."""
        if random.random() > p: return frames
        
        # 1. Darken (Gamma Correction)
        gamma = random.uniform(2.5, 3.5) # High gamma = Darker
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        dark_frames = np.array([cv2.LUT(f, table) for f in frames])
        
        # 2. Add high-ISO Grain
        noise = np.random.normal(0, 15, dark_frames.shape).astype(np.uint8)
        noisy_frames = cv2.add(dark_frames, noise)
        
        return noisy_frames

    def temporal_augment(self, frames: np.ndarray) -> np.ndarray:
        """Applies temporal shifts (Reverse, Frame Drop)."""
        seq_len = len(frames)
        
        # 1. Reverse Sequence (p=0.3)
        # Valid because traffic flows both ways usually
        if random.random() < 0.3:
            frames = frames[::-1]

        # 2. Time Warp / Frame Drop (p=0.3)
        # Simulates lower FPS camera or lag
        if random.random() < 0.3:
            # Drop every 2nd frame, then duplicate to fill gap
            new_frames = []
            for i in range(0, seq_len, 2):
                new_frames.append(frames[i])
                if i+1 < seq_len:
                    new_frames.append(frames[i]) # Duplicate/Freeze frame
            frames = np.array(new_frames[:seq_len])
            
        return frames

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Main entry point.
        Input: (Seq_Len, C, H, W) float32 or uint8
        Output: (Seq_Len, C, H, W)
        """
        # Convert to (Seq_Len, H, W, C) uint8 for OpenCV/Albumentations
        is_float = frames.dtype == np.float32
        if is_float:
            frames = (frames * 255).astype(np.uint8)
        
        # PyTorch (C, H, W) -> OpenCV (H, W, C)
        frames = np.transpose(frames, (0, 2, 3, 1))

        # 1. Apply Temporal Augs
        frames = self.temporal_augment(frames)
        
        # 2. Apply Special Effects (Weather/Time)
        frames = self.add_rain_effect(frames, intensity=0.3)
        frames = self.simulate_night(frames, p=0.3)

        # 3. Apply Consistent Spatial Augs using ReplayCompose
        # Apply transforms to first frame to get parameters
        aug_data = self.spatial_transform(image=frames[0])
        replay_params = aug_data['replay']
        
        augmented_frames = [aug_data['image']]
        
        # Replay exact same transforms on rest of frames
        for i in range(1, len(frames)):
            res = A.ReplayCompose.replay(replay_params, image=frames[i])
            augmented_frames.append(res['image'])
            
        frames = np.stack(augmented_frames)

        # Convert back to PyTorch format
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        if is_float:
            frames = frames.astype(np.float32) / 255.0
            
        return frames

if __name__ == "__main__":
    # --- Visualization / Test Block ---
    import matplotlib.pyplot as plt
    
    # Create a dummy sequence (White line moving across black background)
    seq = np.zeros((16, 3, 224, 224), dtype=np.float32)
    for i in range(16):
        seq[i, :, 100:120, 10*i:10*i+50] = 1.0 # Moving white rectangle
        
    augmentor = VideoAugmentor()
    
    # Apply augmentation
    aug_seq = augmentor(seq.copy())
    
    # Visualize
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    indices = [0, 4, 8, 12, 15]
    
    for i, idx in enumerate(indices):
        # Original
        axs[0, i].imshow(seq[idx].transpose(1, 2, 0))
        axs[0, i].set_title(f"Orig Frame {idx}")
        axs[0, i].axis('off')
        
        # Augmented
        axs[1, i].imshow(aug_seq[idx].transpose(1, 2, 0))
        axs[1, i].set_title(f"Aug Frame {idx}")
        axs[1, i].axis('off')
        
    plt.suptitle("Augmentation Validation (Top: Original, Bottom: Augmented)")
    plt.tight_layout()
    plt.savefig("experiments/eda_results/augmentation_test.png")
    print("✅ Augmentation test saved to experiments/eda_results/augmentation_test.png")