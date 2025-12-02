import torch
import torch.nn as nn
import sys
import os

# --- Path Hack: Allow imports from src when running this script directly ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------------------------------

from src.models.conv_lstm import ConvLSTM

class TrafficFlowPredictor(nn.Module):
    """
    Complete baseline model for traffic flow prediction using ConvLSTM.
    
    Architecture:
        Input (B, T, C, H, W) 
        → ConvLSTM Backbone (Spatio-Temporal Features)
        → Global Average Pooling (Spatial Reduction)
        → Fully Connected Layers (Regression Head)
        → Output (B, 1) - Predicted Flow Value
    """
    
    def __init__(
        self, 
        input_channels: int = 3,
        hidden_dims: list = [64, 128],
        kernel_sizes: list = [3, 3],
        fc_dims: list = [256],
        dropout: float = 0.3,
        num_layers: int = 2
    ):
        super(TrafficFlowPredictor, self).__init__()
        
        self.num_layers = len(hidden_dims)
        assert self.num_layers == len(kernel_sizes), "Hidden dims and kernel sizes must match"
        
        # 1. ConvLSTM Backbone
        # Captures space-time dynamics
        self.conv_lstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=self.num_layers,
            batch_first=True,
            return_all_layers=False
        )
        
        # 2. Prediction Head
        # Reduces 3D feature map to 1D vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Build Fully Connected Layers dynamically
        fc_layers = []
        in_dim = hidden_dims[-1] # Output of last ConvLSTM layer
        
        for hidden_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
            
        # Final projection to scalar (Traffic Flow Value)
        fc_layers.append(nn.Linear(in_dim, 1))
        
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - Batch of video sequences
        
        Returns:
            predictions: (B, 1) - Predicted traffic flow values
        """
        # 1. Pass through ConvLSTM
        # Returns: output_last_layer, last_states
        # output_last_layer shape: (B, T, Hidden, H, W)
        lstm_out, _ = self.conv_lstm(x)
        
        # 2. Extract last timestep
        # We want to predict based on the entire history, so we take the final state
        # Shape: (B, Hidden, H, W)
        last_timestep = lstm_out[:, -1, :, :, :]
        
        # 3. Spatial Pooling
        # Shape: (B, Hidden, 1, 1)
        pooled = self.pool(last_timestep)
        
        # 4. Flatten
        # Shape: (B, Hidden)
        flattened = pooled.view(pooled.size(0), -1)
        
        # 5. Regression Head
        # Shape: (B, 1)
        prediction = self.head(flattened)
        
        return prediction

def print_model_summary(model, input_size=(16, 3, 224, 224)):
    """
    Prints model architecture and parameter count.
    """
    print("\n" + "="*60)
    print(f"Model: {model.__class__.__name__}")
    print("-" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # --- Integration Test ---
    def test_predictor():
        print("🧪 Testing TrafficFlowPredictor...")
        
        # Configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, T, C, H, W = 4, 10, 3, 64, 64 # Reduced size for quick test
        
        # Instantiate Model
        model = TrafficFlowPredictor(
            input_channels=C,
            hidden_dims=[32, 64],
            kernel_sizes=[3, 3],
            fc_dims=[128],
            dropout=0.2
        ).to(device)
        
        # Print Summary
        print_model_summary(model)
        
        # Dummy Input
        x = torch.randn(B, T, C, H, W).to(device)
        
        # Forward Pass
        try:
            out = model(x)
            print(f"✅ Forward Pass Successful")
            print(f"   Input:  {x.shape}")
            print(f"   Output: {out.shape}")
            
            # Assertions
            assert out.shape == (B, 1), f"Expected output (B, 1), got {out.shape}"
            assert not torch.isnan(out).any(), "Output contains NaNs"
            
        except Exception as e:
            print(f"❌ Forward Pass Failed: {e}")
            raise e

    test_predictor()