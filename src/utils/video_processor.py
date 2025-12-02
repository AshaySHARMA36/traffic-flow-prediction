import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(
        self, 
        target_fps: int = 10, 
        target_resolution: Tuple[int, int] = (224, 224),
        sequence_length: int = 16,
        stride: int = 8,
        use_imagenet_stats: bool = True
    ):
        """
        Robust Video Processing Pipeline for Traffic Flow Prediction.

        Args:
            target_fps: Frame rate to sample at (e.g., 10 FPS).
            target_resolution: (height, width) for model input.
            sequence_length: Number of frames per input sequence.
            stride: Overlap stride for sequence generation.
            use_imagenet_stats: If True, normalizes with ImageNet mean/std.
        """
        self.target_fps = target_fps
        self.target_res = target_resolution
        self.seq_len = sequence_length
        self.stride = stride
        self.use_imagenet = use_imagenet_stats
        
        # ImageNet constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes frame to target resolution while maintaining aspect ratio (Letterboxing).
        Adds black padding to fill the remaining area.
        """
        h, w = frame.shape[:2]
        target_h, target_w = self.target_res
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas of target size (black background)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts to float32, scales to [0, 1], and optionally applies ImageNet normalization.
        """
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        if self.use_imagenet:
            frame = (frame - self.mean) / self.std
            
        return frame

    def extract_frames(
        self, 
        video_path: Union[str, Path], 
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Reads video and samples frames at self.target_fps.
        
        Returns:
            np.ndarray: Shape (Num_Frames, H, W, 3)
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return np.array([])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video codec: {video_path}")
            return np.array([])

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling interval (e.g., if video is 30fps and target is 10fps, take every 3rd frame)
        if original_fps <= 0:
            logger.warning(f"Invalid FPS detected for {video_path}. Assuming 30 FPS.")
            original_fps = 30.0
            
        sample_interval = max(1, int(round(original_fps / self.target_fps)))
        
        frames = []
        frame_idx = 0
        processed_count = 0
        
        # Use a context manager logic for cleaner resource handling
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should keep this frame
                if frame_idx % sample_interval == 0:
                    # 1. Resize (Letterbox)
                    resized = self.resize_frame(frame)
                    # 2. Normalize
                    normalized = self.normalize_frame(resized)
                    frames.append(normalized)
                    processed_count += 1
                    
                    if max_frames and processed_count >= max_frames:
                        break
                
                frame_idx += 1
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
        finally:
            cap.release()

        if len(frames) == 0:
            logger.warning(f"No frames extracted from {video_path}")
            return np.array([])
            
        return np.stack(frames) # Returns (T, H, W, C)

    def extract_sequences(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Extracts sliding window sequences from the video.
        
        Returns:
            List of arrays, where each array is (Seq_Len, C, H, W) for PyTorch.
            Note: PyTorch expects Channel-First (C, H, W).
        """
        # 1. Get all frames (T, H, W, C)
        frames = self.extract_frames(video_path)
        if len(frames) < self.seq_len:
            logger.warning(f"Video too short ({len(frames)} frames) for sequence length {self.seq_len}")
            return []

        sequences = []
        num_frames = len(frames)
        
        # 2. Sliding Window
        for i in range(0, num_frames - self.seq_len + 1, self.stride):
            seq = frames[i : i + self.seq_len] # Shape (Seq_Len, H, W, C)
            
            # 3. Permute to PyTorch format (Seq_Len, C, H, W)
            # Transpose (H, W, C) -> (C, H, W)
            seq = np.transpose(seq, (0, 3, 1, 2))
            
            sequences.append(seq)
            
        return sequences

if __name__ == "__main__":
    # Test the class
    import time
    
    # Create dummy video path (Change this to a real path to test!)
    dummy_path = "data/processed/videos/cityflow_001.avi" 
    
    if Path(dummy_path).exists():
        print(f"Testing VideoProcessor on {dummy_path}...")
        
        vp = VideoProcessor(target_fps=10, target_resolution=(224, 224), sequence_length=16, stride=8)
        
        start = time.time()
        # Test sequences extraction
        seqs = vp.extract_sequences(dummy_path)
        end = time.time()
        
        if seqs:
            print(f"✅ Success! Extracted {len(seqs)} sequences.")
            print(f"   Single Sequence Shape: {seqs[0].shape} (Seq, C, H, W)")
            print(f"   Value Range: [{seqs[0].min():.3f}, {seqs[0].max():.3f}]")
            print(f"   Processing Time: {end - start:.2f}s")
        else:
            print("⚠️ Extracted 0 sequences (Video might be too short or empty).")
    else:
        print(f"❌ Test file not found: {dummy_path}")
        print("Please run this on an actual video file from 'data/processed/videos/'.")