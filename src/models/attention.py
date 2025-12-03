import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# --- Path Hack for direct execution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    The core mathematical operation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        dropout (float): Dropout probability for attention weights.
        scale (bool): Whether to scale scores by sqrt(d_k).
    """
    
    def __init__(self, dropout=0.1, scale=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor (Batch, Heads, Time_Q, Dim_Head)
            k: Key tensor   (Batch, Heads, Time_K, Dim_Head)
            v: Value tensor (Batch, Heads, Time_K, Dim_Head)
            mask: Optional mask (Batch, 1, Time_Q, Time_K) 
                  0 = Masked (Ignore), 1 = Unmasked (Keep)
        
        Returns:
            output: (Batch, Heads, Time_Q, Dim_Head)
            attn_weights: (Batch, Heads, Time_Q, Time_K)
        """
        # 1. Get dimension of Key (d_k)
        # q shape: [B, H, T, D]
        d_k = q.size(-1)
        
        # 2. Compute Attention Scores (QK^T)
        # We transpose the last two dimensions of K to align for multiplication
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 3. Scaling
        # Prevents dot products from getting too large (which kills gradients in softmax)
        if self.scale:
            scores = scores / math.sqrt(d_k)
            
        # 4. Apply Mask (Optional)
        # Used for Causal Masking (prevent looking at future) or Padding
        if mask is not None:
            # Replace scores where mask is 0 with -infinity
            # Softmax(-inf) -> 0, so these tokens get 0 attention
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 5. Softmax
        # Convert scores to probabilities (rows sum to 1)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Dropout
        # Randomly zero out some attention weights (regularization)
        attn_weights = self.dropout(attn_weights)
        
        # 7. Compute Weighted Sum (Attention * V)
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

# --- Unit Test ---
if __name__ == "__main__":
    def test_scaled_dot_product_attention():
        print("🧪 Testing Scaled Dot-Product Attention...")
        
        # Initialize
        attention = ScaledDotProductAttention(dropout=0.0) # No dropout for deterministic check
        
        # Parameters
        B, num_heads, T, d_k = 4, 8, 16, 64
        
        # Create dummy Q, K, V
        q = torch.randn(B, num_heads, T, d_k)
        k = torch.randn(B, num_heads, T, d_k)
        v = torch.randn(B, num_heads, T, d_k)
        
        print(f"   Query Shape: {q.shape}")
        
        # 1. Forward Pass
        output, attn_weights = attention(q, k, v)
        
        print(f"   Output Shape: {output.shape}")
        print(f"   Weights Shape: {attn_weights.shape}")
        
        assert output.shape == (B, num_heads, T, d_k), "Output shape mismatch"
        assert attn_weights.shape == (B, num_heads, T, T), "Weights shape mismatch"
        
        # 2. Check Softmax property (Rows sum to 1)
        weights_sum = attn_weights.sum(dim=-1)
        # Tolerance for float precision
        is_valid_prob = torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
        
        if is_valid_prob:
            print("   ✅ Attention weights sum to 1.0")
        else:
            print(f"   ❌ Attention weights sum failed: {weights_sum[0,0,0]}")

        # 3. Test Causal Masking
        # Mask future tokens (Upper triangle is 0)
        # We use a lower-triangular matrix of 1s
        causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
        
        out_masked, attn_masked = attention(q, k, v, mask=causal_mask)
        
        # Check if upper triangle is strictly 0
        upper_tri = torch.triu(attn_masked, diagonal=1)
        if torch.all(upper_tri == 0):
            print("   ✅ Causal masking works (Future tokens are 0)")
        else:
            print("   ❌ Causal masking failed (Future leaks detected)")
            
        print("\n✅ All Attention tests passed!")

    test_scaled_dot_product_attention()