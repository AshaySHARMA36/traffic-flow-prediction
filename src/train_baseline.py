import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score

# --- Path Setup ---
# Allows importing from src when running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.baseline import TrafficFlowPredictor
from src.data.data_loader import create_dataloaders

# --- Configuration ---
CONFIG = {
    'lr': 1e-4,
    'epochs': 50,
    'batch_size': 4,
    'hidden_dims': [64, 32],
    'fc_dims': [128],
    'sequence_length': 16,
    'patience': 5, # Early stopping patience
    'save_dir': 'experiments/checkpoints'
}

def compute_metrics(predictions, targets):
    """
    Compute comprehensive evaluation metrics.
    Args:
        predictions: np.array of shape (N,)
        targets: np.array of shape (N,)
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # R-squared Score (Coefficient of Determination)
    # Handles cases where R2 is undefined (e.g. single sample)
    if len(targets) > 1:
        r2 = r2_score(targets, predictions)
    else:
        r2 = 0.0

    # Mean Absolute Percentage Error (Handle division by zero)
    # Add epsilon to denominator
    epsilon = 1e-7
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (frames, targets) in enumerate(pbar):
        # 1. Move data to device
        frames = frames.to(device)
        targets = targets.to(device)
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        outputs = model(frames)
        
        # 4. Compute loss
        loss = criterion(outputs, targets)
        
        # 5. Backward pass
        loss.backward()
        
        # 6. Optimizer step
        optimizer.step()
        
        # 7. Statistics
        running_loss += loss.item()
        
        # Detach and move to cpu for metrics
        all_preds.extend(outputs.detach().cpu().numpy().flatten())
        all_targets.extend(targets.detach().cpu().numpy().flatten())
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = running_loss / len(train_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    
    return avg_loss, metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for frames, targets in pbar:
            frames = frames.to(device)
            targets = targets.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    avg_loss = running_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    
    return avg_loss, metrics

def train_model(config):
    """Main training loop with validation and early stopping"""
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 2. Data
    # NOTE: Using the optimized 224p dataset we created on Day 3
    print("📂 Loading Data...")
    train_loader, val_loader = create_dataloaders(
        metadata_path='data/processed/dataset_metadata.csv',
        root_dir='data/processed/videos_224', 
        batch_size=config['batch_size'],
        num_workers=0 # Set to 4 on Linux/Cloud, 0 on Windows
    )
    
    # 3. Model
    print("🧠 Initializing Model...")
    model = TrafficFlowPredictor(
        input_channels=3,
        hidden_dims=config['hidden_dims'],
        fc_dims=config['fc_dims']
    ).to(device)
    
    # 4. Optimizer & Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 5. Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🔥 Starting Training for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        duration = time.time() - start_time
        
        # Logging
        print(f"\nEpoch {epoch}/{config['epochs']} ({duration:.1f}s)")
        print(f"   Train Loss: {train_loss:.4f} | MAE: {train_metrics['mae']:.4f} | R2: {train_metrics['r2']:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}")
        
        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(config['save_dir'], 'best_baseline_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"   ⚠️ No improvement. Patience: {patience_counter}/{config['patience']}")
            
        if patience_counter >= config['patience']:
            print("\n🛑 Early stopping triggered.")
            break

def quick_test():
    """Runs a tiny 2-epoch training run to verify pipeline integrity"""
    print("\n⚡ RUNNING QUICK TEST MODE (2 Epochs, Small Batch) ⚡")
    test_config = CONFIG.copy()
    test_config['epochs'] = 2
    test_config['batch_size'] = 2
    train_model(test_config)

if __name__ == "__main__":
    # Check if we want full training or quick test
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        train_model(CONFIG)
    else:
        # Default to quick test to prevent accidental long runs
        quick_test()
        print("\nTo run full training: python src/train_baseline.py --full")