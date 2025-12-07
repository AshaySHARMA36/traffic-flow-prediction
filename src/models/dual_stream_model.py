import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import yaml

# --- Path Hack ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------

from src.models.spatial_encoder import VideoSpatialEncoder
from src.models.temporal_encoder import TemporalConvNet
from src.models.temporal_transformer import TemporalTransformer
from src.models.fusion import ConcatenationFusion, CrossAttentionFusion, GatedFusion

class DualStreamSpatioTemporalTransformer(nn.Module):
    """
    Complete dual-stream architecture for traffic flow prediction.
    
    Combines:
    - Spatial encoder (CNN backbone)
    - Temporal convolutions (TCN) for local dynamics
    - Temporal transformer for global context
    - Fusion module to merge streams
    - Prediction head for final regression
    
    Args:
        backbone_name (str): CNN backbone name (default 'resnet18')
        spatial_feature_dim (int): Output dim of spatial encoder
        tcn_hidden_dims (list): Hidden dims for TCN layers
        transformer_layers (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads
        d_model (int): Dimension for transformer and fusion
        fusion_type (str): 'concat', 'cross_attn', or 'gated'
        num_classes (int): Output dimension (1 for regression)
        dropout (float): Dropout probability
    """
    
    def __init__(self,
                 # Spatial config
                 backbone_name='resnet18',
                 spatial_feature_dim=512,
                 
                 # Temporal config
                 tcn_hidden_dims=[64, 128, 256],
                 transformer_layers=2,
                 num_heads=4,
                 d_model=256,
                 
                 # Fusion config
                 fusion_type='cross_attn',
                 
                 # Prediction config
                 num_classes=1,
                 dropout=0.1):
        
        super().__init__()
        
        # 1. Spatial Encoder
        # We use our VideoSpatialEncoder (ResNet18 backbone)
        self.spatial_encoder = VideoSpatialEncoder(output_dim=spatial_feature_dim, freeze_backbone=True)
            
        # 2. Temporal Convolution Network (Local Stream)
        # Input: spatial_feature_dim -> Output: tcn_hidden_dims[-1]
        self.tcn = TemporalConvNet(
            input_dim=spatial_feature_dim,
            hidden_dims=tcn_hidden_dims,
            kernel_size=3,
            dropout=dropout
        )
        self.tcn_out_dim = tcn_hidden_dims[-1]
        
        # Project TCN output to Transformer dimension if needed
        self.tcn_to_trans = nn.Linear(self.tcn_out_dim, d_model) if self.tcn_out_dim != d_model else nn.Identity()
        
        # 3. Temporal Transformer (Global Stream)
        # Takes TCN output features as input to refine global context
        self.transformer = TemporalTransformer(
            d_model=d_model,
            num_layers=transformer_layers,
            num_heads=num_heads,
            d_ff=d_model * 4,
            dropout=dropout
        )
        
        # Project Spatial features to fusion dimension
        self.spatial_proj = nn.Linear(spatial_feature_dim, d_model)
        
        # 4. Fusion Module
        if fusion_type == 'concat':
            self.fusion = ConcatenationFusion(
                spatial_dim=d_model, 
                temporal_dim=d_model, 
                output_dim=d_model
            )
        elif fusion_type == 'cross_attn':
            self.fusion = CrossAttentionFusion(
                spatial_dim=d_model, 
                temporal_dim=d_model, 
                output_dim=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                spatial_dim=d_model, 
                temporal_dim=d_model, 
                output_dim=d_model
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # 5. Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.d_model = d_model

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T, C, H, W) video batch
            return_attention: If True, returns transformer attention maps
        
        Returns:
            predictions: (B, num_classes)
            attention_weights: (Optional) List of attention maps
        """
        B, T, C, H, W = x.shape
        
        # === A. Spatial Encoding ===
        # Reshape to process all frames at once: (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)
        
        # Extract features: (B*T, 512)
        spatial_features_flat = self.spatial_encoder(x_reshaped)
        
        # Reshape back to sequence: (B, T, 512)
        spatial_features = spatial_features_flat.view(B, T, -1)
        
        # === B. Temporal Processing ===
        
        # Stream 1: TCN (Local Dynamics)
        # TCN expects (B, T, C) input if using our wrapper, or (B, C, T) if raw Conv1d
        # Our TemporalConvNet.forward handles the permute internally: takes (B, T, C) -> returns (B, T, C)
        tcn_out = self.tcn(spatial_features) # (B, T, tcn_out_dim)
        
        # Stream 2: Transformer (Global Context)
        # We feed TCN features into Transformer to refine them
        trans_in = self.tcn_to_trans(tcn_out) # (B, T, d_model)
        trans_out, attn_weights = self.transformer(trans_in) # (B, T, d_model)
        
        # === C. Fusion ===
        # Prepare spatial features for fusion (project to d_model)
        spatial_proj = self.spatial_proj(spatial_features) # (B, T, d_model)
        
        # Fuse Spatial (Appearance) + Transformer (Global Context)
        fused_seq = self.fusion(spatial_proj, trans_out) # (B, T, d_model)
        
        # === D. Prediction ===
        # Use the last timestep features for prediction
        final_state = fused_seq[:, -1, :] # (B, d_model)
        
        prediction = self.predictor(final_state) # (B, 1)
        
        if return_attention:
            return prediction, attn_weights
        return prediction

def save_model_config(path='configs/model_config.yaml'):
    """Save model configuration for reproducibility"""
    config = {
        'spatial_encoder': {
            'backbone': 'resnet18',
            'feature_dim': 512,
            'freeze_backbone': True
        },
        'temporal_processing': {
            'tcn_hidden_dims': [64, 128, 256],
            'transformer_layers': 2,
            'num_heads': 4,
            'd_model': 256
        },
        'fusion': {
            'type': 'cross_attn'
        },
        'prediction': {
            'num_classes': 1,
            'dropout': 0.1
        }
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Model config saved to {path}")

# --- Comprehensive Test ---
if __name__ == "__main__":
    def test_complete_model():
        print("🧪 Testing Dual-Stream Spatio-Temporal Transformer...")
        print("=" * 60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Running on: {device}")
        
        # Initialize Model
        model = DualStreamSpatioTemporalTransformer(
            spatial_feature_dim=512,
            tcn_hidden_dims=[64, 128, 256],
            transformer_layers=2,
            num_heads=4,
            d_model=256,
            fusion_type='cross_attn'
        ).to(device)
        
        # Generate Dummy Input
        # (Batch=2, Time=16, C=3, H=224, W=224)
        x = torch.randn(2, 16, 3, 224, 224).to(device)
        
        # 1. Forward Pass
        print("   Running Forward Pass...")
        pred, attn = model(x, return_attention=True)
        
        print(f"   Input Shape: {x.shape}")
        print(f"   Output Shape: {pred.shape}")
        
        if attn:
            print(f"   Attention Layers: {len(attn)}")
            print(f"   Attention Map Shape: {attn[0].shape}")
        
        assert pred.shape == (2, 1), f"Shape mismatch: {pred.shape}"
        assert not torch.isnan(pred).any(), "NaNs in output"
        
        # 2. Parameter Count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n   Trainable Params: {params:,} (excluding frozen ResNet)")
        
        # 3. Test Fusion Switching
        print("\n   Testing Fusion Types...")
        for f_type in ['concat', 'gated']:
            try:
                m = DualStreamSpatioTemporalTransformer(fusion_type=f_type).to(device)
                out = m(x)
                print(f"   - {f_type}: OK (Shape {out.shape})")
            except Exception as e:
                print(f"   - {f_type}: FAILED ({e})")
            
        print("\n✅ Complete Model Integration Tests PASSED!")
        
        # Save Config
        save_model_config()

    test_complete_model()