import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Union, Tuple
from pathlib import Path

class FrameSampler:
    """
    Implements various strategies to sample frames from video sequences
    for Spatio-Temporal modeling.
    """
    
    @staticmethod
    def _read_video_metadata(video_path: str) -> Tuple[cv2.VideoCapture, int, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap, total_frames, fps

    def uniform_sampling(self, video_path: str, num_frames: int) -> List[int]:
        """
        Selects N frames evenly spaced across the entire video.
        
        Args:
            video_path: Path to video file.
            num_frames: Desired number of frames.
            
        Returns:
            List of frame indices.
        """
        cap, total_frames, _ = self._read_video_metadata(video_path)
        cap.release()
        
        if total_frames <= 0: return []
        
        # Calculate evenly spaced indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return sorted(list(set(indices))) # Unique and sorted

    def adaptive_sampling(self, video_path: str, num_frames: int, motion_threshold: float = 5.0) -> List[int]:
        """
        Samples frames based on motion intensity. High motion = higher sampling probability.
        Uses simple frame differencing as a proxy for Optical Flow speed.
        
        Args:
            motion_threshold: Minimum pixel diff intensity to consider 'motion'.
        """
        cap, total_frames, _ = self._read_video_metadata(video_path)
        
        motion_scores = []
        prev_frame = None
        
        # Pass 1: Calculate motion score for every frame (downsampled for speed)
        # We assume frame 0 has score 0
        motion_scores.append(0.0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Convert to gray and resize for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            if prev_frame is not None:
                # Calculate absolute difference
                diff = cv2.absdiff(prev_frame, gray)
                score = np.mean(diff)
                motion_scores.append(score)
            
            prev_frame = gray
            
        cap.release()
        
        # Normalize motion scores to probability distribution
        motion_scores = np.array(motion_scores)
        
        # Add small epsilon to allow sampling low-motion frames (don't ignore them entirely)
        probabilities = (motion_scores + 1e-5) / np.sum(motion_scores + 1e-5)
        
        # Weighted random sampling without replacement (preserves temporal order later)
        indices = np.random.choice(len(motion_scores), num_frames, replace=False, p=probabilities)
        return sorted(indices)

    def sliding_window_sampling(self, video_path: str, window_size: int, stride: int) -> List[List[int]]:
        """
        Generates overlapping windows for training sequences.
        
        Returns:
            List of Lists, where each inner list is a sequence of frame indices.
            e.g. [[0,1,2], [2,3,4], ...]
        """
        cap, total_frames, _ = self._read_video_metadata(video_path)
        cap.release()
        
        sequences = []
        for i in range(0, total_frames - window_size + 1, stride):
            seq_indices = list(range(i, i + window_size))
            sequences.append(seq_indices)
            
        return sequences

    def keyframe_sampling(self, video_path: str, num_frames: int) -> List[int]:
        """
        Extracts frames where significant scene changes occur.
        Returns exactly 'num_frames' by selecting the top-N changes.
        """
        cap, total_frames, _ = self._read_video_metadata(video_path)
        
        diff_scores = []
        prev_hist = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Use Histogram Comparison (more robust to lighting than raw diff)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                # Correlation: 1.0 = identical, 0.0 = different
                score = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diff_scores.append((frame_idx, score))
            else:
                diff_scores.append((frame_idx, 0.0))
                
            prev_hist = hist
            frame_idx += 1
            
        cap.release()
        
        # Sort by score (descending) to find biggest changes
        diff_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N indices
        top_indices = [x[0] for x in diff_scores[:num_frames]]
        
        # Always include first and last frame for context
        if 0 not in top_indices: top_indices[-1] = 0
        
        return sorted(top_indices)

def compare_strategies(video_path: str, target_frames: int = 16):
    """
    Visualizes how different strategies pick frames on a timeline.
    """
    sampler = FrameSampler()
    
    # 1. Run all methods
    start = time.time()
    uniform_idxs = sampler.uniform_sampling(video_path, target_frames)
    t_uniform = time.time() - start
    
    start = time.time()
    adaptive_idxs = sampler.adaptive_sampling(video_path, target_frames)
    t_adaptive = time.time() - start
    
    start = time.time()
    # For sliding window, we just visualize the first sequence
    sliding_seqs = sampler.sliding_window_sampling(video_path, window_size=target_frames, stride=target_frames//2)
    sliding_idxs = sliding_seqs[0] if sliding_seqs else []
    t_sliding = time.time() - start
    
    start = time.time()
    keyframe_idxs = sampler.keyframe_sampling(video_path, target_frames)
    t_key = time.time() - start

    # 2. Visualize
    _, total, _ = sampler._read_video_metadata(video_path)
    cap = cv2.VideoCapture(video_path) # Keep open to grab frames
    
    print(f"\n📊 Strategy Comparison (Video: {Path(video_path).name})")
    print(f"Total Frames in Video: {total}")
    print(f"{'Strategy':<20} | {'Time (s)':<10} | {'Indices (First 5)'}")
    print("-" * 60)
    print(f"{'Uniform':<20} | {t_uniform:<10.4f} | {uniform_idxs[:5]}")
    print(f"{'Adaptive (Motion)':<20} | {t_adaptive:<10.4f} | {adaptive_idxs[:5]}")
    print(f"{'Sliding Window':<20} | {t_sliding:<10.4f} | {sliding_idxs[:5]}")
    print(f"{'Keyframe (Scene)':<20} | {t_key:<10.4f} | {keyframe_idxs[:5]}")

    # Plot Timeline
    plt.figure(figsize=(15, 6))
    y_vals = [4, 3, 2, 1]
    labels = ['Uniform', 'Adaptive', 'Sliding Window (Seq 0)', 'Keyframe']
    colors = ['blue', 'red', 'green', 'purple']
    
    all_indices = [uniform_idxs, adaptive_idxs, sliding_idxs, keyframe_idxs]
    
    for i, idxs in enumerate(all_indices):
        plt.scatter(idxs, [y_vals[i]] * len(idxs), alpha=0.6, s=100, c=colors[i], label=labels[i])
        plt.hlines(y_vals[i], 0, total, colors='gray', alpha=0.2)

    plt.title(f"Frame Selection Strategy Comparison ({target_frames} frames)")
    plt.xlabel("Frame Index")
    plt.yticks(y_vals, labels)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_path = "experiments/eda_results/sampling_comparison.png"
    plt.savefig(save_path)
    print(f"\n✅ Visualization saved to: {save_path}")

if __name__ == "__main__":
    # Test with one of your CityFlow videos
    VIDEO_PATH = "data/processed/videos/cityflow_001.avi"
    if Path(VIDEO_PATH).exists():
        compare_strategies(VIDEO_PATH, target_frames=20)
    else:
        print("Please run this script after Day 2 data processing.")