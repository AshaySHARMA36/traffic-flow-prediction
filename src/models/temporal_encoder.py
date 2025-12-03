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

from src.models.temporal_block import TemporalConvBlock

class TemporalConvNet(nn.Module):
    """
    Temporal Convolution Network (TCN) with exponentially increasing dilation.
    
    Stacks multiple TemporalConvBlock layers with increasing dilation rates
    to capture both short-term and long-term temporal dependencies.
    
    Architecture:
        Input (B, feature_dim, T) 
        → TemporalConvBlock(dilation=1)
        → TemporalConvBlock(dilation=2)
        → TemporalConvBlock(dilation=4)
        → TemporalConvBlock(dilation=8)
        → Output (B, out_channels, T)
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden dimensions for each layer [64, 128, 128]
        kernel_size: Kernel size for all temporal convolutions
        dilation_base: Base for exponential dilation (default: 2)
        dropout: Dropout rate
    """
    
    def __init__(self, 
                 input_dim, 
                 hidden_dims=[64, 128, 128, 256], 
                 kernel_size=3, 
                 dilation_base=2, 
                 dropout=0.2,
                 activation='relu'):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_layers = len(hidden_dims)
        
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            dilation = dilation_base ** i  # Exponential dilation: 1, 2, 4, 8...
            
            layers.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Store configuration for receptive field calculation
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Input_Channels)
               Note: TCN expects (Batch, Channels, Time), so we permute inside.
               
        Returns:
            out: (Batch, Time, Output_Channels)
        """
        # 1. Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(0, 2, 1) 
        
        # 2. Pass through TCN stack
        y = self.network(x)
        
        # 3. Permute back to (Batch, Time, Channels) for compatibility with Linear layers
        return y.permute(0, 2, 1)
    
    def receptive_field(self):
        """
        Calculate the effective receptive field of the entire network.
        RF = 1 + sum((K-1) * Dilation)
        
        Returns:
            receptive_field: Number of timesteps the network can see
        """
        rf = 1
        for i in range(self.num_layers):
            dilation = self.dilation_base ** i
            rf += (self.kernel_size - 1) * dilation
        return rf

class MultiScaleTCN(nn.Module):
    """
    TCN that outputs features from multiple scales.
    Useful for capturing both short-term (early layers) and long-term (later layers) patterns.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 128, 128, 256], kernel_size=3, dilation_base=2, dropout=0.2):
        super().__init__()
        
        # Create individual temporal conv blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_ch = input_dim if i == 0 else hidden_dims[i-1]
            out_ch = hidden_dims[i]
            dilation = dilation_base ** i
            
            self.blocks.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
    
    def forward(self, x):
        """
        Returns:
            outputs: List of outputs from each scale
            final_output: Output from the last layer
        """
        # Permute for Conv1d
        x = x.permute(0, 2, 1)
        
        outputs = []
        for block in self.blocks:
            x = block(x)
            # Store permuted back (B, T, C)
            outputs.append(x.permute(0, 2, 1))
        
        return outputs, outputs[-1]

def visualize_receptive_field(num_layers=4, kernel_size=3, dilation_base=2):
    """
    Print ASCII visualization of receptive field growth.
    """
    print("\nReceptive Field Growth:")
    print("-" * 50)
    
    rf = 1
    for i in range(num_layers):
        dilation = dilation_base ** i
        layer_increase = (kernel_size - 1) * dilation
        rf += layer_increase
        
        print(f"Layer {i+1} (dilation={dilation:2d}): "
              f"Gain={layer_increase:2d}, Cumulative RF={rf:2d} frames")
    print("-" * 50)

# --- Comprehensive Test Function ---
def test_temporal_conv_net():
    """Test TCN with different configurations"""
    
    print("=" * 60)
    print("Testing Temporal Convolution Network")
    print("=" * 60)
    
    # Test 1: Basic Forward Pass
    print("\n1. Testing Basic Forward Pass...")
    # Matches Spatial Encoder output (B, T, C)
    tcn = TemporalConvNet(input_dim=512, hidden_dims=[64, 128, 256], kernel_size=3)
    
    # Input: (Batch=4, Time=16, Channels=512)
    x = torch.randn(4, 16, 512)
    out = tcn(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    
    # Expect (Batch, Time, Last_Hidden)
    assert out.shape == (4, 16, 256), "Output shape mismatch!"
    
    # Test 2: Receptive Field Calculation
    print("\n2. Checking Receptive Field...")
    rf = tcn.receptive_field()
    print(f"   Calculated RF: {rf} frames")
    
    # Verify manually: 1 + (2*1) + (2*2)