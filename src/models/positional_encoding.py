import torch
import torch.nn as nn
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Path Hack for direct execution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    Standard Transformer PE (Vaswani et al., 2017).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model=256, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term
        # exp(arange(0, d, 2) * -(log(10000.0) / d))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) - input sequences
        
        Returns:
            x: (B, T, d_model) - sequences with positional encoding added
        """
        # Add positional encoding (slice to current seq length)
        # x is (B, T, D), pe is (1, MaxLen, D) -> Broadcasts correctly
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings (alternative to sinusoidal).
    Can be more flexible but requires more data to train and doesn't generalize 
    beyond max_len.
    """
    
    def __init__(self, d_model=256, max_len=100, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        
        Returns:
            x: (B, T, d_model)
        """
        T = x.size(1)
        # Slice to current sequence length
        x = x + self.pos_embedding[:, :T, :]
        return self.dropout(x)

def visualize_positional_encoding(pe_matrix, save_path='positional_encoding.png'):
    """
    Visualize positional encoding patterns as a heatmap.
    
    Args:
        pe_matrix: (T, d_model) positional encoding matrix
    """
    plt.figure(figsize=(12, 6))
    # Transpose to show Dimensions on Y, Time on X
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Position (Time Step)')
    plt.ylabel('Embedding Dimension')
    plt.title('Sinusoidal Positional Encoding Pattern')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved positional encoding visualization to {save_path}")

def compare_positional_encodings():
    """Compare sinusoidal vs learned positional encodings"""
    print("\nComparing Encodings:")
    print("=" * 60)
    
    d_model = 256
    pos_sin = PositionalEncoding(d_model)
    pos_learned = LearnedPositionalEncoding(d_model)
    
    # 1. Sinusoidal
    print("Sinusoidal Positional Encoding:")
    print(f"  - No trainable parameters (Fixed)")
    print(f"  - Generalizes to longer sequences")
    print(f"  - Deterministic wave patterns")
    
    # 2. Learned
    print(f"\nLearned Positional Encoding:")
    params_learned = sum(p.numel() for p in pos_learned.parameters())
    print(f"  - {params_learned:,} trainable parameters")
    print(f"  - Flexible adaptation to data")
    print(f"  - Limited to max_len ({pos_learned.pos_embedding.shape[1]})")
    
    print("\nRecommendation: Use Sinusoidal for Traffic Prediction (better generalization).")

# --- Unit Test ---
if __name__ == "__main__":
    def test_positional_encoding():
        print("🧪 Testing Positional Encoding...")
        print("=" * 60)
        
        d_model = 256
        max_len = 100
        
        # 1. Test Sinusoidal
        print("1. Testing Sinusoidal PE...")
        pos_enc_sin = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        
        # Create dummy input (B=4, T=16, D=256)
        x = torch.zeros(4, 16, d_model) # Zeros so output is purely PE
        x_pos = pos_enc_sin(x)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {x_pos.shape}")
        
        # Verify output is not zero (PE was added)
        assert torch.count_nonzero(x_pos) > 0, "PE not added!"
        
        # Visualize
        pe_viz_data = pos_enc_sin.pe[0, :50, :].numpy()
        visualize_positional_encoding(pe_viz_data, save_path='experiments/eda_results/positional_encoding.png')
        
        # 2. Test Learned
        print("\n2. Testing Learned PE...")
        pos_enc_learned = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)
        x_pos_learned = pos_enc_learned(x)
        print(f"   Output shape: {x_pos_learned.shape}")
        
        # 3. Comparison
        compare_positional_encodings()
        
        print("\n✅ Positional encoding tests passed!")

    # Ensure output dir exists
    os.makedirs('experiments/eda_results', exist_ok=True)
    test_positional_encoding()