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

from src.models.dual_stream_model import DualStreamSpatioTemporalTransformer

def print_model_summary(model, input_size=(2, 16, 3, 224, 224)):
    """
    Print comprehensive model summary.
    
    Shows:
    - Layer-wise architecture
    - Parameter counts
    - Output shapes
    - Memory usage
    """
    
    print("\n" + "=" * 80)
    print("DUAL-STREAM SPATIO-TEMPORAL TRANSFORMER - MODEL SUMMARY")
    print("=" * 80)
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Component-wise breakdown
    print("\n" + "-" * 80)
    print("COMPONENT-WISE BREAKDOWN")
    print("-" * 80)
    
    components = {
        'Spatial Encoder': model.spatial_encoder,
        'Temporal CNN (TCN)': model.tcn,
        'Temporal Transformer': model.transformer,
        'Fusion Module': model.fusion,
        'Prediction Head': model.predictor
    }
    
    for name, component in components.items():
        params = sum(p.numel() for p in component.parameters())
        print(f"{name:.<40} {params:>12,} ({params/1e6:>6.2f}M)")
    
    # Receptive field info
    print("\n" + "-" * 80)
    print("TEMPORAL RECEPTIVE FIELD")
    print("-" * 80)
    tcn_rf = model.tcn.receptive_field()
    print(f"TCN Receptive Field: {tcn_rf} frames")
    print(f"Transformer: Full sequence attention (Global)")
    
    # Memory estimate
    print("\n" + "-" * 80)
    print("ESTIMATED MEMORY USAGE")
    print("-" * 80)
    
    x = torch.randn(*input_size)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Forward Pass (Batch Size {input_size[0]}): {memory_mb:.2f} MB")
        
        # Estimate training memory (roughly 3x forward pass for gradients + optim state)
        print(f"Training (estimated): {memory_mb * 3.5:.2f} MB")
    else:
        print("CUDA not available. Skipping memory profiling.")
    
    print("\n" + "=" * 80)


def visualize_architecture(save_path='architecture_diagram.txt'):
    """
    Create ASCII architecture diagram and save to file.
    """
    
    diagram = """
    ╔════════════════════════════════════════════════════════════════╗
    ║         DUAL-STREAM SPATIO-TEMPORAL TRANSFORMER                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Input Video Sequence
    (Batch, Time=16, Channels=3, Height=224, Width=224)
            │
            ├──────────────────────────────────────────────┐
            │                                              │
            ▼                                              │
    ┏━━━━━━━━━━━━━━━━━━━━━┓                              │
    ┃  SPATIAL ENCODER    ┃                              │
    ┃  ─────────────────  ┃                              │
    ┃  • ResNet18         ┃                              │
    ┃  • Frozen Backbone  ┃                              │
    ┃  • Feature: 512-dim ┃                              │
    ┗━━━━━━━━━━━━━━━━━━━━━┛                              │
            │                                              │
            │ Spatial Features (B, T, 512)                 │
            │                                              │
            ▼                                              │
    ┏━━━━━━━━━━━━━━━━━━━━━┓                              │
    ┃  TEMPORAL CNN       ┃                              │
    ┃  ──────────────     ┃                              │
    ┃  • Dilated Conv     ┃                              │
    ┃  • Layers: [64,128] ┃                              │
    ┃  • RF: ~15 frames   ┃                              │
    ┗━━━━━━━━━━━━━━━━━━━━━┛                              │
            │                                              │
            │ Local Features (B, T, 256)                   │
            │                                              │
            ▼                                              │
    ┏━━━━━━━━━━━━━━━━━━━━━┓                              │
    ┃  TRANSFORMER        ┃                              │
    ┃  ──────────────     ┃                              │
    ┃  • 2 Layers         ┃                              │
    ┃  • 4 Heads          ┃                              │
    ┃  • Global Attention ┃                              │
    ┗━━━━━━━━━━━━━━━━━━━━━┛                              │
            │                                              │
            │ Global Features (B, T, 256)                  │
            │                                              │
            └──────────────────┬────────────────────────────┘
                               │
                               ▼
                        ┏━━━━━━━━━━━━━━┓
                        ┃ FUSION MODULE ┃
                        ┃ ───────────── ┃
                        ┃ Cross-Attn    ┃
                        ┗━━━━━━━━━━━━━━┛
                               │
                               │ Fused Features (B, T, 256)
                               │
                               ▼
                        ┏━━━━━━━━━━━━━━┓
                        ┃ PREDICTION HD ┃
                        ┃ ───────────── ┃
                        ┃ • Pooling     ┃
                        ┃ • FC Layers   ┃
                        ┗━━━━━━━━━━━━━━┛
                               │
                               ▼
                    Traffic Flow Prediction
                         (Batch, 1)
    """
    
    print(diagram)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(diagram)
    
    print(f"\nArchitecture diagram saved to {save_path}")


def compare_with_baseline():
    """Compare dual-stream model with ConvLSTM baseline"""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Dual-Stream vs ConvLSTM Baseline")
    print("=" * 80)
    
    # Dual-stream model
    ds_model = DualStreamSpatioTemporalTransformer(
        spatial_feature_dim=512,
        tcn_hidden_dims=[64, 128, 256],
        d_model=256
    )
    ds_params = sum(p.numel() for p in ds_model.parameters() if p.requires_grad)
    
    # Baseline ConvLSTM (Import locally to avoid circular dependencies if any)
    try:
        from src.models.baseline import TrafficFlowPredictor
        baseline_model = TrafficFlowPredictor(input_channels=3, hidden_dims=[64, 32])
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
    except ImportError:
        print("Baseline model not found. Using estimated params.")
        baseline_params = 270337 # From Day 4 logs
    
    print(f"\n{'Model':<30} {'Parameters':<20} {'Receptive Field':<20}")
    print("-" * 70)
    print(f"{'ConvLSTM Baseline':<30} {baseline_params/1e6:>8.2f}M           {'Full sequence':<20}")
    print(f"{'Dual-Stream Transformer':<30} {ds_params/1e6:>8.2f}M           {'Full + Local':<20}")
    
    print(f"\nParameter increase: {(ds_params/baseline_params):.1f}x")
    print("Expected performance gain: 15-25% (based on similar architectures)")

if __name__ == "__main__":
    # Create output dir
    os.makedirs("experiments/eda_results", exist_ok=True)
    
    # Instantiate Model
    model = DualStreamSpatioTemporalTransformer(
        spatial_feature_dim=512,
        tcn_hidden_dims=[64, 128, 256],
        transformer_layers=2,
        num_heads=4,
        d_model=256
    )
    
    print_model_summary(model)
    visualize_architecture(save_path="experiments/eda_results/architecture_diagram.txt")
    compare_with_baseline()