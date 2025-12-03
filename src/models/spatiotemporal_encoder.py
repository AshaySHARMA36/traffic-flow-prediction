import torch
import torch.nn as nn
import sys
import os
import time

# --- Path Hack for direct execution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------

from src.models.temporal_encoder import TemporalConvNet
from src.models.spatial_encoder import VideoSpatialEncoder

class SpatioTemporalEncoder(nn.Module):
    """
    Complete spatio-temporal encoder combining spatial (CNN) and temporal (TCN) processing.
    """
    
    def __init__(self, 
                 spatial_encoder, 
                 tcn_hidden_dims=[64, 128, 128, 256], 
                 tcn_kernel_size=3, 
                 dropout=0.2):
        super(SpatioTemporalEncoder, self).__init__()
        
        self.spatial_encoder = spatial_encoder
        
        # Get spatial feature dimension automatically
        if hasattr(spatial_encoder, 'output_dim'):
            spatial_dim = spatial_encoder.output_dim
        else:
            spatial_dim = 512 # Fallback
            
        self.spatial_dim = spatial_dim
        
        # Temporal Convolution Network
        self.tcn = TemporalConvNet(
            input_dim=spatial_dim,
            hidden_dims=tcn_hidden_dims,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )
        
        self.output_dim = tcn_hidden_dims[-1]

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video sequences
        Returns:
            temporal_features: (B, T, temporal_dim)
        """
        B, T, C, H, W = x.shape
        
        # 1. Spatial Encoding
        # Reshape to (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)
        
        # Extract features: (B*T, spatial_dim)
        spatial_features = self.spatial_encoder(x_reshaped)
        
        # Restore dimensions: (B, T, spatial_dim)
        spatial_features = spatial_features.view(B, T, self.spatial_dim)
        
        # 2. Temporal Encoding
        temporal_features = self.tcn(spatial_features)
        
        return temporal_features

# --- Comprehensive Test ---
if __name__ == "__main__":
    def test_spatiotemporal_encoder():
        print("🧪 Testing Spatio-Temporal Encoder Pipeline...")
        print("=" * 60)
        
        # 1. Setup Components
        print("   Initializing Spatial Encoder (ResNet18)...")
        spatial_enc = VideoSpatialEncoder(output_dim=512, freeze_backbone=True)
        
        # Initialize ST Encoder
        st_encoder = SpatioTemporalEncoder(
            spatial_encoder=spatial_enc,
            tcn_hidden_dims=[512, 512, 256], 
            tcn_kernel_size=3
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st_encoder = st_encoder.to(device)
        
        # 2. Dummy Input (Video Batch)
        x = torch.randn(2, 16, 3, 224, 224).to(device)
        
        # 3. Forward Pass
        print(f"   Input Shape: {x.shape}")
        start = time.time()
        out = st_encoder(x)
        end = time.time()
        
        print(f"   Output Shape: {out.shape}")
        print(f"   Inference Time: {end - start:.4f}s")
        
        # 4. Assertions
        assert out.shape == (2, 16, 256), f"Shape mismatch! Got {out.shape}"
        
        print("\n✅ All Spatio-Temporal Tests PASSED!")

    test_spatiotemporal_encoder()