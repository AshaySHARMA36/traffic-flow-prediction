import cv2
import os
from pathlib import Path

def inspect_cityflow_structure(root_dir):
    print(f"\n🔍 Scanning CityFlow Dataset at: {root_dir}")
    root_path = Path(root_dir)
    
    # We now know the structure is train/Sxx/cxxx/vdo.avi
    # We search specifically for this pattern
    video_files = list(root_path.rglob("vdo.avi"))
    
    if not video_files:
        print("❌ No 'vdo.avi' files found. Check your path.")
        return

    print(f"✅ Found {len(video_files)} video clips total.")
    
    # Let's inspect the first 3 videos to confirm they are readable
    print("\n--- detailed inspection of first 3 clips ---")
    
    for i, vid_path in enumerate(video_files[:3]):
        # Get parent folders to identify Scenario and Camera
        camera_id = vid_path.parent.name  # e.g., c001
        scenario_id = vid_path.parent.parent.name # e.g., S01
        
        print(f"\n🎥 Video {i+1}: [{scenario_id} - {camera_id}]")
        print(f"   Path: {vid_path}")
        
        cap = cv2.VideoCapture(str(vid_path))
        
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"   ✅ Status: Readable")
            print(f"   📐 Resolution: {width}x{height}")
            print(f"   ⏱️ FPS: {fps:.2f}")
            print(f"   🎞️ Frames: {frame_count} ({duration:.1f}s)")
        else:
            print(f"   ❌ Status: Corrupt/Unreadable")
            
        cap.release()

if __name__ == "__main__":
    # POINT THIS to your specific folder path from the structure.txt
    # Based on your upload, it is:
    DATASET_ROOT = r"D:\traffic-flow-prediction\AICity22_Track1_MTMC_Tracking\train"
    
    if os.path.exists(DATASET_ROOT):
        inspect_cityflow_structure(DATASET_ROOT)
    else:
        print(f"❌ Error: The path '{DATASET_ROOT}' does not exist on this machine.")