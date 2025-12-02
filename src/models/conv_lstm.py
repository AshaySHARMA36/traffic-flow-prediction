import torch
import torch.nn as nn
import sys
import os

# --- Path Hack: Allow imports from src when running this script directly ---
# This appends the project root (../../) to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------------------------------

from typing import List, Tuple, Union
import time

# --- Embedded ConvLSTMCell Class ---
class ConvLSTMCell(nn.Module):
    """
    A single Convolutional LSTM Cell.
    
    Unlike a standard LSTM which flattens inputs into 1D vectors (destroying spatial info),
    ConvLSTM uses 2D Convolutions for all internal gate operations.
    
    Equations:
        i_t = σ(W_xi * X_t + W_hi * H_{t-1} + b_i)  (Input Gate)
        f_t = σ(W_xf * X_t + W_hf * H_{t-1} + b_f)  (Forget Gate)
        o_t = σ(W_xo * X_t + W_ho * H_{t-1} + b_o)  (Output Gate)
        g_t = tanh(W_xc * X_t + W_hc * H_{t-1} + b_c) (Cell Candidate)
        
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t  (New Cell State)
        H_t = o_t ⊙ tanh(C_t)            (New Hidden State)
        
    Args:
        input_dim (int): Number of channels in input tensor.
        hidden_dim (int): Number of channels in hidden state.
        kernel_size (int): Size of the convolutional kernel (must be odd, e.g., 3).
        bias (bool): Whether to add a bias term.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Padding ensures the output spatial dimension (H, W) matches the input
        # padding = kernel_size // 2 works perfectly for odd kernels (3 -> 1, 5 -> 2)
        padding = kernel_size // 2
        
        # We perform one big convolution for performance, then split the output.
        # Input to conv is: Concatenation of (Input X) and (Hidden State H)
        # Total Input Channels = input_dim + hidden_dim
        # Total Output Channels = 4 * hidden_dim (one set for each gate: i, f, o, g)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, hidden_state=None):
        """
        Forward pass for a single timestep.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            hidden_state: Tuple (h_prev, c_prev) from previous timestep.
                          Both shapes are (B, hidden_dim, H, W).
                          If None, initializes to zeros.
                          
        Returns:
            h_next, c_next: The updated hidden and cell states.
        """
        batch_size, _, height, width = x.size()
        
        # 1. Initialize hidden state if this is the first timestep
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, (height, width), x.device)
            
        h_prev, c_prev = hidden_state
        
        # 2. Concatenation
        # We stack the current input X and previous hidden state H along the channel dimension.
        # Why? This allows the convolution kernel to "see" both the current visual data
        # and the memory of motion simultaneously to make decisions.
        # Shape: (B, input_dim + hidden_dim, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        
        # 3. Convolution
        # Perform the unified convolution operation
        # Shape: (B, 4 * hidden_dim, H, W)
        combined_conv = self.conv(combined)
        
        # 4. Split Gates
        # Chunk the output into 4 tensors along channel dimension
        # Each part has shape: (B, hidden_dim, H, W)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 5. Activations
        i = torch.sigmoid(cc_i) # Input gate: What to write? (0-1)
        f = torch.sigmoid(cc_f) # Forget gate: What to keep? (0-1)
        o = torch.sigmoid(cc_o) # Output gate: How much to reveal? (0-1)
        g = torch.tanh(cc_g)    # Cell candidate: New information (-1 to 1)
        
        # 6. Update Memory (Cell State)
        # C_t = f_t * C_{t-1} + i_t * g_t
        c_next = f * c_prev + i * g
        
        # 7. Update Output (Hidden State)
        # H_t = o_t * tanh(C_t)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        """
        Creates zero-initialized states for the first timestep.
        
        Args:
            batch_size: Int
            image_size: Tuple (Height, Width)
            device: torch.device (cpu or cuda)
        
        Returns:
            (h, c) tuple of zeros
        """
        height, width = image_size
        
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        
        return h, c

# --- ConvLSTM Module ---
class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module for sequence processing.
    Stacks multiple ConvLSTMCell layers to capture complex spatio-temporal patterns.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        kernel_sizes: List[int], 
        num_layers: int, 
        batch_first: bool = True, 
        return_all_layers: bool = False
    ):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        
        # Validation
        assert len(hidden_dims) == num_layers, "hidden_dims length must match num_layers"
        assert len(kernel_sizes) == num_layers, "kernel_sizes length must match num_layers"
        
        cell_list = []
        for i in range(self.num_layers):
            # Input dim for first layer is input_dim
            # Input dim for subsequent layers is hidden_dim of previous layer
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i],
                    bias=True
                )
            )
            
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Union[List[Tuple[torch.Tensor, torch.Tensor]], None] = None
    ):
        """
        Args:
            x: Input tensor.
               Shape (B, T, C, H, W) if batch_first=True
               Shape (T, B, C, H, W) if batch_first=False
            hidden_state: List of (h, c) tuples for each layer.
                          If None, initializes to zeros.
        
        Returns:
            layer_output_list: List of output tensors from each layer (or just last layer)
                               Shape of each: (B, T, Hidden_Dim, H, W)
            last_state_list: List of (h, c) tuples for each layer at the last timestep
        """
        # 1. Permute if necessary to get (T, B, C, H, W) for easier looping
        if self.batch_first:
            # (B, T, C, H, W) -> (T, B, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)
            
        seq_len, batch_size, _, height, width = x.size()
        
        # 2. Initialize Hidden States automatically if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width), x.device)
            
        # Storage for outputs
        # layer_output_list[layer_idx] = [output_t0, output_t1, ...]
        layer_output_list = [] 
        last_state_list = []
        
        # We need a copy of the current hidden states that we can update in the loop
        # hidden_state is a list of [(h1, c1), (h2, c2), ...]
        current_states = list(hidden_state)
        
        # 3. Process Sequence
        # For each layer, we collect the sequence of outputs
        
        # Current input starts as the video frames
        # Shape: (T, B, C, H, W)
        layer_input = x 
        
        all_layer_outputs = [] # To store stacked outputs for all layers
        
        for layer_idx in range(self.num_layers):
            cell = self.cell_list[layer_idx]
            h, c = current_states[layer_idx]
            
            output_inner = [] # Outputs for this layer across all timesteps
            
            for t in range(seq_len):
                # Input to this layer at time t
                # If layer=0, input is frame[t]
                # If layer>0, input is output of prev_layer[t]
                if layer_idx == 0:
                    inp = x[t] 
                else:
                    inp = layer_input[t] # The output sequence from prev layer
                
                h, c = cell(inp, (h, c))
                output_inner.append(h)
                
            # Stack timesteps for this layer: (T, B, H_dim, H, W)
            layer_output = torch.stack(output_inner, dim=0)
            
            # This layer's output becomes the input for the next layer (if any)
            layer_input = layer_output
            
            # Update the state for this layer to the final state
            current_states[layer_idx] = (h, c)
            
            # Handle return formatting
            if self.batch_first:
                # (T, B, ...) -> (B, T, ...)
                layer_output = layer_output.permute(1, 0, 2, 3, 4)
                
            all_layer_outputs.append(layer_output)

        # 4. Return Logic
        if self.return_all_layers:
            return all_layer_outputs, current_states
        else:
            return all_layer_outputs[-1], current_states

    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size, device)
            )
        return init_states

# --- Unit Test ---
if __name__ == "__main__":
    def test_convlstm_module():
        print("🧪 Testing Stacked ConvLSTM Module...")
        
        # Config
        B, T, C, H, W = 2, 5, 3, 32, 32
        HIDDEN_DIMS = [16, 32]
        KERNEL_SIZES = [3, 3]
        NUM_LAYERS = 2
        
        # Initialize
        model = ConvLSTM(
            input_dim=C,
            hidden_dims=HIDDEN_DIMS,
            kernel_sizes=KERNEL_SIZES,
            num_layers=NUM_LAYERS,
            batch_first=True,
            return_all_layers=False
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Dummy Input
        x = torch.randn(B, T, C, H, W).to(device)
        
        # 1. Forward Pass Speed Test
        print("   Running forward pass...")
        start = time.time()
        output, states = model(x)
        end = time.time()
        
        print(f"   Forward pass time: {end - start:.4f}s")
        print(f"   Input shape:   {x.shape}")
        print(f"   Output shape:  {output.shape}")
        
        # Assertions
        expected_shape = (B, T, HIDDEN_DIMS[-1], H, W)
        assert output.shape == expected_shape, f"Shape Mismatch: Got {output.shape}, Expected {expected_shape}"
        assert len(states) == NUM_LAYERS, "Incorrect number of hidden states returned"
        assert not torch.isnan(output).any(), "Output contains NaNs"
        
        # 2. Gradient Flow Test
        print("   Checking gradient flow...")
        loss = output.sum()
        loss.backward()
        
        # Check if first layer gradients exist
        first_layer_grad = model.cell_list[0].conv.weight.grad
        assert first_layer_grad is not None, "Gradients did not flow to first layer!"
        assert torch.norm(first_layer_grad) > 0, "Gradients are zero!"
        
        print("✅ Stacked ConvLSTM Module Test PASSED")

    test_convlstm_module()