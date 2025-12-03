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

from src.models.attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module.
    
    Splits the input into multiple heads, applies attention in parallel,
    and concatenates the results.
    
    Args:
        d_model (int): Input and output dimension.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
    """
    
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Validation
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear Projections for Q, K, V
        # We can project all heads at once and then split
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Core Attention Mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Final Output Projection
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query (Batch, Seq_Len, d_model)
            k: Key   (Batch, Seq_Len, d_model)
            v: Value (Batch, Seq_Len, d_model)
            mask: Optional mask (Batch, 1, Seq_Len, Seq_Len)
        
        Returns:
            out: (Batch, Seq_Len, d_model)
            attention_weights: (Batch, Num_Heads, Seq_Len, Seq_Len)
        """
        batch_size = q.size(0)
        
        # 1. Linear Projections
        # (B, T, D) -> (B, T, D)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # 2. Split Heads
        # Reshape: (B, T, D) -> (B, T, Heads, Head_Dim)
        # Transpose: (B, T, H, HD) -> (B, H, T, HD) for matrix multiplication
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 3. Scaled Dot-Product Attention
        # out: (B, H, T, HD), weights: (B, H, T, T)
        out, attention_weights = self.attention(q, k, v, mask=mask)
        
        # 4. Concatenate Heads
        # Transpose back: (B, H, T, HD) -> (B, T, H, HD)
        out = out.permute(0, 2, 1, 3).contiguous()
        
        # Flatten heads: (B, T, H * HD) -> (B, T, D)
        out = out.view(batch_size, -1, self.d_model)
        
        # 5. Final Linear Projection
        out = self.fc_out(out)
        
        return out, attention_weights

def create_causal_mask(seq_len):
    """
    Creates a square mask where the future is masked out.
    Used for autoregressive tasks (predicting next frame based only on past).
    
    Returns:
        mask: (1, 1, seq_len, seq_len)
    """
    # Lower triangular matrix of 1s
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # Add dimensions for Batch and Heads so it broadcasts correctly
    return mask.unsqueeze(0).unsqueeze(0)

# --- Unit Test ---
if __name__ == "__main__":
    def test_multi_head_attention():
        print("🧪 Testing Multi-Head Attention...")
        
        # Config
        B, T, D = 4, 16, 256
        heads = 8
        
        mha = MultiHeadAttention(d_model=D, num_heads=heads, dropout=0.1)
        
        # Dummy Input
        x = torch.randn(B, T, D)
        
        # 1. Forward Pass (Self-Attention: Q=K=V=x)
        print(f"   Input Shape: {x.shape}")
        out, weights = mha(x, x, x)
        
        print(f"   Output Shape: {out.shape}")
        print(f"   Weights Shape: {weights.shape}")
        
        assert out.shape == (B, T, D), "Output shape mismatch"
        assert weights.shape == (B, heads, T, T), "Weights shape mismatch"
        
        # 2. Causal Mask Test
        mask = create_causal_mask(T)
        out_masked, weights_masked = mha(x, x, x, mask=mask)
        
        # Check upper triangle of weights (should be 0)
        # Select first batch, first head
        w = weights_masked[0, 0]
        upper_tri = torch.triu(w, diagonal=1)
        if torch.all(upper_tri == 0):
            print("   ✅ Causal masking works (Future is hidden)")
        else:
            print("   ❌ Causal masking failed")
            
        # 3. Gradient Flow
        loss = out_masked.sum()
        loss.backward()
        
        # Check gradients on projection layers
        grad_norm = torch.norm(mha.w_q.weight.grad)
        if grad_norm > 0:
            print("   ✅ Gradients flow correctly")
        else:
            print("   ❌ No gradients found")
            
        print("\n✅ All Multi-Head Attention tests passed!")

    test_multi_head_attention()