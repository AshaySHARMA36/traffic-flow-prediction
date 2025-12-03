import torch
import torch.nn as nn
import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Path Hack ---
sys.path.append(os.getcwd())

from src.models.temporal_encoder import TemporalConvNet

def analyze_tcn_performance():
    print("\n" + "=" * 70)
    print("🚀 TEMPORAL CONVOLUTION NETWORK - PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Test different configurations
    configs = [
        {'name': 'Lightweight', 'hidden_dims': [32, 64, 128]},
        {'name': 'Standard', 'hidden_dims': [64, 128, 256]}, # Our Day 6 choice
        {'name': 'Heavy', 'hidden_dims': [128, 256, 512]},
    ]

    for config in configs:
        print(f"\n🔹 {config['name']} Configuration:")
        print("-" * 30)
        
        tcn = TemporalConvNet(
            input_dim=512,
            hidden_dims=config['hidden_dims'],
            kernel_size=3
        ).to(device)
        
        # 1. Parameter Efficiency
        params = sum(p.numel() for p in tcn.parameters())
        print(f"   Parameters:      {params:,} ({params/1e6:.2f}M)")
        
        # 2. Receptive Field
        rf = tcn.receptive_field()
        print(f"   Receptive Field: {rf} frames")
        
        # 3. Memory Usage (Forward Pass)
        x = torch.randn(16, 16, 512).to(device) # (B, T, C)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = tcn(x)
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"   GPU Memory:      {mem:.2f} MB")
        
        # 4. Inference Speed
        # Warmup
        for _ in range(10): _ = tcn(x)
        
        start = time.time()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                _ = tcn(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = time.time() - start
        throughput = (iterations * 16) / elapsed # Sequences per second
        latency = (elapsed * 1000) / iterations  # ms per batch
        
        print(f"   Throughput:      {throughput:.2f} sequences/sec")
        print(f"   Latency:         {latency:.2f} ms/batch")

def compare_temporal_models():
    print("\n" + "=" * 70)
    print("⚔️  MODEL COMPARISON (TCN vs LSTM vs Standard Conv)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 512
    hidden_dim = 256
    seq_len = 16
    
    # 1. TCN (Dilated)
    tcn = TemporalConvNet(input_dim, [64, 128, 256]).to(device)
    
    # 2. LSTM (Standard RNN Baseline)
    lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True).to(device)
    
    # 3. Standard Conv (No Dilation)
    std_conv = nn.Sequential(
        nn.Conv1d(input_dim, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 256, 3, padding=1)
    ).to(device)
    
    models = {
        'TCN (Dilated)': tcn,
        'LSTM (RNN)': lstm,
        'Conv (No Dilation)': std_conv
    }
    
    print(f"{'Model':<20} {'Params':<15} {'Receptive Field':<20} {'Speed (seq/s)':<15}")
    print("-" * 70)
    
    x = torch.randn(16, 16, 512).to(device) # (B, T, C)
    
    for name, model in models.items():
        # Params
        params = sum(p.numel() for p in model.parameters())
        
        # RF Calculation
        if name == 'TCN (Dilated)':
            rf = model.receptive_field()
        elif name == 'LSTM (RNN)':
            rf = seq_len # Theoretically infinite
        else:
            rf = 1 + (3-1)*3 # 7 frames for 3 layers of kernel 3
            
        # Speed Test
        # Warmup
        for _ in range(10): 
            if 'Conv' in name:
                _ = model(x.transpose(1, 2) if 'TCN' not in name else x)
            else:
                _ = model(x)
                
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                if name == 'TCN (Dilated)':
                    _ = model(x)
                elif name == 'LSTM (RNN)':
                    _ = model(x)
                else:
                    # Standard conv needs (B, C, T) manually
                    _ = model(x.transpose(1, 2))
                    
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        elapsed = time.time() - start
        speed = (100 * 16) / elapsed
        
        print(f"{name:<20} {params/1e6:>6.2f}M       {rf:>3d} frames          {speed:>8.2f}")

if __name__ == "__main__":
    analyze_tcn_performance()
    compare_temporal_models()
    
    # Save the standard TCN for Day 7
    print("\n💾 Saving TCN Checkpoint for Day 7...")
    tcn = TemporalConvNet(input_dim=512, hidden_dims=[64, 128, 256])
    os.makedirs('experiments/checkpoints', exist_ok=True)
    torch.save(tcn.state_dict(), 'experiments/checkpoints/tcn_day6.pth')
    print("✅ Saved to experiments/checkpoints/tcn_day6.pth")