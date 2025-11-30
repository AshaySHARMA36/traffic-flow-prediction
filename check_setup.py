import sys
import torch
import cv2
import transformers
import numpy as np

def verify_environment():
    print(f"python: {sys.version.split()[0]}")
    
    # 1. Check PyTorch & CUDA
    print(f"\n[1] Checking PyTorch...")
    print(f"    - Torch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"    - CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"    - GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"    - CUDA Version: {torch.version.cuda}")
    else:
        print("    ! WARNING: CUDA is NOT available. Training will be slow on CPU.")

    # 2. Check Computer Vision
    print(f"\n[2] Checking Computer Vision...")
    print(f"    - OpenCV Version: {cv2.__version__}")
    
    # 3. Check Transformer Libs
    print(f"\n[3] Checking Transformers...")
    print(f"    - Transformers Version: {transformers.__version__}")
    
    # 4. Simple Tensor Test
    try:
        x = torch.tensor([1.0, 2.0]).cuda() if cuda_available else torch.tensor([1.0, 2.0])
        print(f"\n[4] Tensor Test: PASSED (Tensor created on {x.device})")
    except Exception as e:
        print(f"\n[4] Tensor Test: FAILED ({e})")

    print("\n✅ Environment Setup Complete." if cuda_available else "\n⚠️ Environment Setup Complete (CPU Only).")

if __name__ == "__main__":
    verify_environment()