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

class TemporalConvBlock(nn.Module):
    """
    Single temporal convolution block with residual connection and dilation.
    
    Architecture:
        Input (B, C_in, T) 
        → Conv1d (Dilated) 
        → BatchNorm 
        → Activation 
        → Dropout 
        → Residual Add 
        → Output (B, C_out, T)
        
    Args:
        in_channels (int): Input channel dimension
        out_channels (int): Output channel dimension
        kernel_size (int): Temporal kernel size (default: 3)
        dilation (int): Dilation rate for dilated convolution
        dropout (float): Dropout probability
        activation (str): 'relu', 'gelu', or 'swish'
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilation=1, 
                 dropout=0.2, 
                 activation='relu'):
        super(TemporalConvBlock, self).__init__()
        
        # 1. Calculate Padding
        # Goal: Maintain sequence length T.
        # Formula for 'same' padding with dilation: P = (K-1) * D / 2
        # This assumes the kernel size is odd (3, 5, 7)
        assert kernel_size % 2 == 1, "Kernel size must be odd to maintain symmetry"
        padding = (kernel_size - 1) * dilation // 2
        
        # 2. Temporal Convolution
        # 1D convolution slides across the Time dimension
        # Bias is False because BatchNorm follows immediately
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False 
        )
        
        # 3. Batch Normalization
        # Normalizes across the Channel dimension for stable training
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 4. Activation Function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # 5. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 6. Residual Connection
        # If input and output channels differ, we need a 1x1 convolution
        # to project the input to the correct size for addition.
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()
            
    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Channels, Time)
            
        Returns:
            out: Output tensor (Batch, Out_Channels, Time)
        """
        # Save input for residual connection
        residual = x
        
        # Main path
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual path
        # Project residual if channel dimensions changed
        residual = self.residual_proj(residual)
        
        # Add (Skip Connection)
        out = out + residual
        
        return out

def calculate_receptive_field(kernel_size, dilation):
    """
    Calculates the receptive field increase for a single layer.
    RF_layer = (Kernel - 1) * Dilation + 1
    Note: For a stack, you sum (Kernel-1)*Dilation
    """
    return (kernel_size - 1) * dilation + 1

# --- Unit Test ---
if __name__ == "__main__":
    def test_temporal_conv_block():
        print("🧪 Testing TemporalConvBlock...")
        
        # Test 1: Identity Residual (Same Channels)
        print("\n1. Testing Identity Residual (64 -> 64)...")
        block1 = TemporalConvBlock(64, 64, kernel_size=3, dilation=1)
        x1 = torch.randn(8, 64, 16) # (B, C, T)
        out1 = block1(x1)
        
        print(f"   Input: {x1.shape}")
        print(f"   Output: {out1.shape}")
        assert x1.shape == out1.shape, "Shape mismatch in identity block"
        
        # Test 2: Projection Residual (Different Channels, Dilated)
        print("\n2. Testing Projection Residual (64 -> 128, Dilation=2)...")
        block2 = TemporalConvBlock(64, 128, kernel_size=3, dilation=2)
        x2 = torch.randn(8, 64, 16)
        out2 = block2(x2)
        
        print(f"   Input: {x2.shape}")
        print(f"   Output: {out2.shape}")
        assert out2.shape == (8, 128, 16), "Shape mismatch in projection block"
        
        # Test 3: High Dilation (RF Check)
        print("\n3. Testing High Dilation (D=8)...")
        block3 = TemporalConvBlock(128, 128, kernel_size=3, dilation=8)
        x3 = torch.randn(8, 128, 16)
        out3 = block3(x3)
        print(f"   Output: {out3.shape}")
        
        # Test 4: Gradient Flow
        print("\n4. Checking Gradient Flow...")
        loss = out3.sum()
        loss.backward()
        
        grad = block3.conv.weight.grad
        assert grad is not None and torch.norm(grad) > 0, "Gradients are zero/None!"
        print("   ✅ Gradients flow correctly")
        
        # RF Calc check
        rf = calculate_receptive_field(3, 8)
        print(f"\n   Receptive field for K=3, D=8: {rf} frames (Expected 17)")
        
        print("\n✅ All TemporalConvBlock tests passed!")

    test_temporal_conv_block()