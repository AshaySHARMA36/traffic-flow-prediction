import torch
import torch.nn as nn
import sys
import os

# --- Path Hack for direct execution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------

from src.models.multi_head_attention import MultiHeadAttention
from src.models.positional_encoding import PositionalEncoding

class TemporalTransformerBlock(nn.Module):
    """
    Transformer block combining multi-head attention with feed-forward network.
    Uses Pre-Norm architecture for better training stability.
    
    Architecture:
        Input → LayerNorm → Multi-Head Attention → Residual Add
              → LayerNorm → Feed-Forward → Residual Add → Output
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward hidden dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model=256, num_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (B, T, d_model)
            attention_weights: (B, num_heads, T, T)
        """
        
        # Sublayer 1: Attention (Pre-Norm)
        norm_x = self.norm1(x)
        attn_output, attn_weights = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output) # Residual
        
        # Sublayer 2: Feed-Forward (Pre-Norm)
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output # Residual
        
        return x, attn_weights


class TemporalTransformer(nn.Module):
    """
    Stack of temporal transformer blocks for sequence modeling.
    Designed to process temporal features (e.g., from TCN or CNN).
    """
    
    def __init__(self, 
                 d_model=256, 
                 num_layers=6, 
                 num_heads=8, 
                 d_ff=1024, 
                 dropout=0.1, 
                 use_positional_encoding=True):
        super().__init__()
        
        # Positional Encoding
        self.use_pos_enc = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Stack of Transformer Blocks
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final Normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, d_model) - temporal features
            mask: Optional attention mask
        
        Returns:
            output: (B, T, d_model)
            attention_weights: List of attention weights from each layer
        """
        
        # 1. Add Positional Encoding
        if self.use_pos_enc:
            x = self.pos_encoding(x)
            
        # 2. Process through Layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights_list.append(attn_weights)
            
        # 3. Final Norm
        x = self.norm(x)
        
        return x, attention_weights_list

# --- Comprehensive Test ---
if __name__ == "__main__":
    def test_temporal_transformer():
        print("🧪 Testing Temporal Transformer...")
        print("=" * 60)
        
        # Config
        B, T, D = 4, 16, 256
        
        # Initialize
        transformer = TemporalTransformer(
            d_model=D,
            num_layers=3,
            num_heads=8,
            d_ff=512,
            dropout=0.1
        )
        
        # Dummy Input (Simulating TCN output)
        x = torch.randn(B, T, D)
        
        print(f"   Input Shape: {x.shape}")
        
        # 1. Forward Pass
        out, attn_maps = transformer(x)
        
        print(f"   Output Shape: {out.shape}")
        print(f"   Num Layers: {len(attn_maps)}")
        print(f"   Attn Map Shape: {attn_maps[0].shape}")
        
        assert out.shape == (B, T, D), "Output shape mismatch"
        assert len(attn_maps) == 3, "Wrong number of attention maps returned"
        
        # 2. Gradient Flow
        loss = out.sum()
        loss.backward()
        
        # Check gradients on the first layer's FFN
        first_layer_grad = transformer.layers[0].ffn[0].weight.grad
        if torch.norm(first_layer_grad) > 0:
            print("   ✅ Gradients flow correctly through stack")
        else:
            print("   ❌ Gradient flow failed")
            
        # 3. Visualize Attention (Using helper from Task 3)
        try:
            from src.models.multi_head_attention import visualize_multi_head_attention
            print("\n   Generating Attention Visualization...")
            visualize_multi_head_attention(attn_maps[0], head_idx=0, save_path='experiments/eda_results/transformer_attention.png')
        except ImportError:
            print("   ⚠️ Visualization helper not found, skipping plot.")

        print("\n✅ All Temporal Transformer tests passed!")

    # Ensure output dir exists
    os.makedirs('experiments/eda_results', exist_ok=True)
    test_temporal_transformer()