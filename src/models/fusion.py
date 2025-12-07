import torch
import torch.nn as nn
import sys
import os

# --- Path Hack ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------

from src.models.multi_head_attention import MultiHeadAttention

class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion.
    Concatenates spatial and temporal features, then projects them.
    Low computational cost.
    """
    
    def __init__(self, spatial_dim, temporal_dim, output_dim):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: (B, T, spatial_dim)
            temporal_features: (B, T, temporal_dim)
        Returns:
            fused: (B, T, output_dim)
        """
        # Concatenate along feature dimension
        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        fused = self.projection(combined)
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion.
    Allows spatial features to 'query' the temporal context, and vice versa.
    High computational cost, high expressivity.
    """
    
    def __init__(self, spatial_dim, temporal_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Project both to a common dimension for attention
        self.common_dim = output_dim
        self.spatial_proj = nn.Linear(spatial_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        
        # Cross-Attention 1: Spatial queries Temporal
        self.spatial_to_temporal = MultiHeadAttention(
            d_model=output_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Cross-Attention 2: Temporal queries Spatial
        self.temporal_to_spatial = MultiHeadAttention(
            d_model=output_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, spatial_features, temporal_features):
        # 1. Project to same dimension
        spatial = self.spatial_proj(spatial_features)   # (B, T, D)
        temporal = self.temporal_proj(temporal_features) # (B, T, D)
        
        # 2. Cross Attention
        # Q=Spatial, K=Temporal, V=Temporal -> "What spatial parts need temporal context?"
        # Using positional arguments (q, k, v) to avoid keyword mismatch errors
        spat_attended, _ = self.spatial_to_temporal(spatial, temporal, temporal)
        
        # Q=Temporal, K=Spatial, V=Spatial -> "What temporal parts need spatial detail?"
        temp_attended, _ = self.temporal_to_spatial(temporal, spatial, spatial)
        
        # 3. Combine
        combined = torch.cat([spat_attended, temp_attended], dim=-1)
        fused = self.output_proj(combined)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable weights.
    Dynamically decides whether to trust spatial or temporal features more 
    at each timestep.
    Medium computational cost.
    """
    def __init__(self, spatial_dim, temporal_dim, output_dim):
        super().__init__()
        
        # Project inputs to output dimension first
        self.spatial_proj = nn.Linear(spatial_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        
        # Gate mechanism: Takes concatenation, outputs probability (0 to 1)
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, spatial_features, temporal_features):
        # 1. Project features
        s_proj = self.spatial_proj(spatial_features)
        t_proj = self.temporal_proj(temporal_features)
        
        # 2. Calculate Gate
        # Values near 1 = Trust Spatial, Values near 0 = Trust Temporal
        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        gate = self.gate_net(combined)
        
        # 3. Weighted Sum
        fused = gate * s_proj + (1 - gate) * t_proj
        fused = self.norm(fused)
        
        return fused

# --- Unit Test ---
if __name__ == "__main__":
    def test_fusion_modules():
        print("🧪 Testing Fusion Strategies...")
        
        B, T = 4, 16
        spatial_dim = 512
        temporal_dim = 256
        output_dim = 256
        
        s_feats = torch.randn(B, T, spatial_dim)
        t_feats = torch.randn(B, T, temporal_dim)
        
        # 1. Concatenation
        print("\n1. Testing ConcatenationFusion...")
        fusion1 = ConcatenationFusion(spatial_dim, temporal_dim, output_dim)
        out1 = fusion1(s_feats, t_feats)
        print(f"   Output: {out1.shape}")
        assert out1.shape == (B, T, output_dim)
        
        # 2. Cross-Attention
        print("\n2. Testing CrossAttentionFusion...")
        fusion2 = CrossAttentionFusion(spatial_dim, temporal_dim, output_dim)
        out2 = fusion2(s_feats, t_feats)
        print(f"   Output: {out2.shape}")
        assert out2.shape == (B, T, output_dim)
        
        # 3. Gated
        print("\n3. Testing GatedFusion...")
        fusion3 = GatedFusion(spatial_dim, temporal_dim, output_dim)
        out3 = fusion3(s_feats, t_feats)
        print(f"   Output: {out3.shape}")
        assert out3.shape == (B, T, output_dim)
        
        # 4. Gradient Check (Gated)
        loss = out3.sum()
        loss.backward()
        print("   ✅ Gradients flow correctly")
        
        print("\n✅ All fusion modules work!")

    test_fusion_modules()