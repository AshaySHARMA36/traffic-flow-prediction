import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class VideoSpatialEncoder(nn.Module):
    """
    Processes RGB Video Frames (3 channels).
    Uses a Pre-trained ResNet-18 backbone.
    """
    def __init__(self, output_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        
        # Load Pre-trained ResNet18
        # We use standard ImageNet weights
        weights = ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        
        # Remove the final Fully Connected (FC) classification layer
        # We want the 512-dim feature vector from the layer before it
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optional: Freeze weights to save training time/memory
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.output_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (Batch, Channels=3, Height, Width)
        Output: (Batch, 512)
        """
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x) # -> [B, 512, 1, 1]
        return features.squeeze(-1).squeeze(-1) # -> [B, 512]


class HeatmapSpatialEncoder(nn.Module):
    """
    Processes Traffic Heatmaps (8 channels).
    Uses a custom lightweight CNN.
    """
    def __init__(self, input_channels: int = 8, output_dim: int = 512):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 8 -> 64
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling to handle any input resolution
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project to target dimension (e.g., 512 to match ResNet)
        self.projection = nn.Linear(256, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (Batch, Channels=8, Height, Width)
        Output: (Batch, 512)
        """
        x = self.features(x)       # -> [B, 256, 1, 1]
        x = x.flatten(1)           # -> [B, 256]
        x = self.projection(x)     # -> [B, 512]
        return x

if __name__ == "__main__":
    # --- Quick Unit Test ---
    print("🧪 Testing Spatial Encoders...")
    
    # 1. Test Video Encoder (CityFlow)
    dummy_video_batch = torch.randn(4, 3, 224, 224) 
    vid_encoder = VideoSpatialEncoder()
    vid_out = vid_encoder(dummy_video_batch)
    
    print(f"\n🎥 Video Encoder:")
    print(f"   Input: {dummy_video_batch.shape}")
    print(f"   Output: {vid_out.shape}")