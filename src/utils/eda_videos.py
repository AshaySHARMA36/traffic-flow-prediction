import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

class VideoEDA:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect video files (CityFlow uses .avi, we also look for .mp4)
        self.video_files = list(self.data_dir.glob("*.avi")) + list(self.data_dir.glob("*.mp4"))
        print(f"found {len(self.video_files)} video files in {self.data_dir}")

    def analyze_single_video(self, video_path):
        """Extracts technical and semantic metrics from a single video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        # Technical Metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        # Content Analysis (Sample 3 frames: Start, Middle, End)
        frames_to_sample = [0, frame_count // 2, max(0, frame_count - 10)]
        sampled_frames = []
        brightness_vals = []
        motion_vals = []
        
        # Simple Motion Detector (Background Subtraction)
        back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        # Iterate through a subset of frames for speed (every 10th frame)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frames_to_sample:
                sampled_frames.append(frame)
            
            # Analyze every 20th frame to save time
            if frame_idx % 20 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Brightness (Mean Pixel Intensity)
                brightness_vals.append(np.mean(gray))
                
                # 2. Motion (Foreground Mask)
                fg_mask = back_sub.apply(frame)
                motion_score = np.count_nonzero(fg_mask) / (width * height)
                motion_vals.append(motion_score)
            
            frame_idx += 1
            
        cap.release()
        
        # Aggregated Metrics
        avg_brightness = np.mean(brightness_vals) if brightness_vals else 0
        avg_motion = np.mean(motion_vals) if motion_vals else 0
        
        return {
            "filename": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "size_mb": file_size_mb,
            "avg_brightness": avg_brightness,
            "avg_motion": avg_motion,
            "sampled_frames": sampled_frames
        }

    def generate_report(self):
        """Main execution loop."""
        results = []
        
        print(f"📊 Analyzing {len(self.video_files)} videos...")
        for vid in tqdm(self.video_files):
            data = self.analyze_single_video(vid)
            if data:
                results.append(data)
                
        if not results:
            print("❌ No valid video data found.")
            return

        df = pd.DataFrame(results)
        
        # --- Categorization Logic ---
        # Brightness < 80 roughly indicates Night (0-255 scale)
        df['Time_of_Day'] = df['avg_brightness'].apply(lambda x: 'Night' if x < 80 else 'Day')
        
        # Motion > 0.05 (5% of pixels moving) roughly indicates High Traffic
        df['Traffic_Level'] = df['avg_motion'].apply(lambda x: 'High' if x > 0.05 else ('Medium' if x > 0.02 else 'Low'))

        # --- Generate Visuals ---
        self.plot_distributions(df)
        self.save_sample_collage(results[:3]) # Only collage first 3 videos to save space
        
        # --- Save Summary CSV ---
        csv_path = self.output_dir / "eda_summary.csv"
        df.drop(columns=['sampled_frames']).to_csv(csv_path, index=False)
        
        # --- Print Console Summary ---
        self.print_console_summary(df)

    def plot_distributions(self, df):
        """Saves histograms of video stats."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Duration Dist
        axs[0, 0].hist(df['duration'], bins=10, color='skyblue', edgecolor='black')
        axs[0, 0].set_title('Video Duration Distribution (sec)')
        
        # Brightness Dist
        axs[0, 1].hist(df['avg_brightness'], bins=10, color='orange', edgecolor='black')
        axs[0, 1].set_title('Brightness Distribution (0=Black, 255=White)')
        
        # Motion/Traffic Dist
        axs[1, 0].hist(df['avg_motion'], bins=10, color='green', edgecolor='black')
        axs[1, 0].set_title('Motion Score Distribution (Traffic Density)')
        
        # File Size
        axs[1, 1].hist(df['size_mb'], bins=10, color='purple', edgecolor='black')
        axs[1, 1].set_title('File Size Distribution (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_plots.png")
        plt.close()

    def save_sample_collage(self, results):
        """Creates a visual collage of Start/Middle/End frames."""
        # Create a grid: Rows = Videos, Cols = 3 (Start, Mid, End)
        rows = len(results)
        cols = 3
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1: axs = [axs] # Handle single video case

        for r, res in enumerate(results):
            frames = res['sampled_frames']
            for c, frame in enumerate(frames):
                if frame is None: continue
                # Convert BGR to RGB for Matplotlib
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                ax = axs[r][c] if rows > 1 else axs[c]
                ax.imshow(rgb_frame)
                ax.axis('off')
                if c == 0: ax.set_title(f"{res['filename']}\nStart")
                if c == 1: ax.set_title("Middle")
                if c == 2: ax.set_title("End")
                
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_frames.png")
        plt.close()

    def print_console_summary(self, df):
        """Prints a summary to console using to_string instead of markdown."""
        print("\n" + "="*40)
        print("📋 EDA SUMMARY REPORT")
        print("="*40)
        
        print(f"\n**Total Videos:** {len(df)}")
        print(f"**Total Dataset Size:** {df['size_mb'].sum():.2f} MB")
        print(f"**Avg Duration:** {df['duration'].mean():.2f} seconds")
        
        print("\n### 🚦 Traffic Conditions")
        # FIXED: using to_string() instead of to_markdown()
        print(df['Traffic_Level'].value_counts().to_string())
        
        print("\n### ☀️ Lighting Conditions")
        print(df['Time_of_Day'].value_counts().to_string())
        
        print(f"\n### 💾 Outputs Saved")
        print(f"- Plots: {self.output_dir}/distribution_plots.png")
        print(f"- Samples: {self.output_dir}/sample_frames.png")
        print(f"- Data: {self.output_dir}/eda_summary.csv")

if __name__ == "__main__":
    # Path configuration
    VIDEO_DIR = r"data/processed/videos"
    OUTPUT_DIR = r"experiments/eda_results"
    
    eda = VideoEDA(VIDEO_DIR, OUTPUT_DIR)
    eda.generate_report()