#!/usr/bin/env python3
"""
Checkpoint Verification Script
Inspects PyTorch .pth/.pt files to verify training completion
"""

import torch
import sys
from pathlib import Path
from datetime import datetime

def format_size(num_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def inspect_checkpoint(ckpt_path):
    """Inspect a checkpoint file and print detailed information"""
    ckpt_path = Path(ckpt_path)
    
    print("\n" + "="*80)
    print(f"CHECKPOINT: {ckpt_path.name}")
    print("="*80)
    
    # File info
    if not ckpt_path.exists():
        print(f"❌ ERROR: File not found at {ckpt_path}")
        return False
    
    file_size = ckpt_path.stat().st_size
    print(f"\n📁 File Info:")
    print(f"   Path: {ckpt_path}")
    print(f"   Size: {format_size(file_size)}")
    
    try:
        # Load checkpoint
        print(f"\n⏳ Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"   ✅ Loaded successfully!")
        
        # Determine checkpoint type
        print(f"\n📦 Checkpoint Contents:")
        if isinstance(ckpt, dict):
            print(f"   Keys: {list(ckpt.keys())}")
            
            # Training info
            print(f"\n🎯 Training Information:")
            
            if 'epoch' in ckpt:
                epoch = ckpt['epoch']
                print(f"   Epoch: {epoch}")
                
                # Check if training seems complete
                if epoch < 10:
                    print(f"   ⚠️  WARNING: Only trained for {epoch} epochs (might be incomplete!)")
                elif epoch < 20:
                    print(f"   ⚠️  Trained for {epoch} epochs (could benefit from more training)")
                else:
                    print(f"   ✅ Trained for {epoch} epochs (good!)")
            
            if 'stage1_epochs' in ckpt and 'stage2_epochs' in ckpt:
                print(f"   Stage 1 Epochs: {ckpt['stage1_epochs']}")
                print(f"   Stage 2 Epochs: {ckpt['stage2_epochs']}")
                total = ckpt['stage1_epochs'] + ckpt['stage2_epochs']
                print(f"   Total Epochs: {total}")
            
            if 'training_complete' in ckpt:
                status = "✅ COMPLETE" if ckpt['training_complete'] else "❌ INCOMPLETE"
                print(f"   Training Status: {status}")
            
            # Metrics
            print(f"\n📊 Performance Metrics:")
            
            # Check various metric keys
            metric_keys = {
                'best_metric': 'Best Metric',
                'best_mae': 'Best MAE',
                'best_val_iou': 'Best Val IoU',
                'best_val_auroc': 'Best Val AUROC',
                'val_loss': 'Validation Loss',
                'test_mae': 'Test MAE',
                'test_auroc': 'Test AUROC',
            }
            
            metrics_found = False
            for key, label in metric_keys.items():
                if key in ckpt:
                    print(f"   {label}: {ckpt[key]:.4f}")
                    metrics_found = True
            
            # Check nested metrics
            if 'metrics' in ckpt and isinstance(ckpt['metrics'], dict):
                print(f"   Additional Metrics:")
                for k, v in ckpt['metrics'].items():
                    if v is not None:
                        print(f"      {k}: {v:.4f}" if isinstance(v, (int, float)) else f"      {k}: {v}")
                metrics_found = True
            
            if 'val_metrics' in ckpt and isinstance(ckpt['val_metrics'], dict):
                print(f"   Validation Metrics:")
                for k, v in ckpt['val_metrics'].items():
                    if isinstance(v, (int, float)):
                        print(f"      {k}: {v:.4f}")
                metrics_found = True
            
            if 'val_stats' in ckpt and isinstance(ckpt['val_stats'], dict):
                print(f"   Validation Stats:")
                for k, v in ckpt['val_stats'].items():
                    if isinstance(v, (int, float)):
                        print(f"      {k}: {v:.4f}")
                metrics_found = True
            
            if not metrics_found:
                print(f"   ⚠️  No metrics found in checkpoint")
            
            # Model architecture
            print(f"\n🏗️  Model Architecture:")
            
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                num_params = sum(p.numel() for p in state_dict.values())
                num_layers = len(state_dict)
                print(f"   Total Parameters: {num_params:,}")
                print(f"   Number of Layers: {num_layers}")
                print(f"   ✅ Model weights present")
            elif 'model' in ckpt:
                state_dict = ckpt['model']
                num_params = sum(p.numel() for p in state_dict.values())
                num_layers = len(state_dict)
                print(f"   Total Parameters: {num_params:,}")
                print(f"   Number of Layers: {num_layers}")
                print(f"   ✅ Model weights present")
            else:
                print(f"   ⚠️  WARNING: No model_state_dict found!")
            
            # Optimizer
            if 'optimizer_state_dict' in ckpt or 'optimizer' in ckpt:
                print(f"   ✅ Optimizer state present")
            
            # History
            if 'history' in ckpt:
                history = ckpt['history']
                if isinstance(history, dict):
                    print(f"\n📈 Training History:")
                    for key in history.keys():
                        if isinstance(history[key], list):
                            print(f"   {key}: {len(history[key])} epochs recorded")
        
        else:
            # Direct state dict (less common)
            print(f"   Type: Direct state_dict (no metadata)")
            num_params = sum(p.numel() for p in ckpt.values())
            print(f"   Total Parameters: {num_params:,}")
            print(f"   ⚠️  WARNING: No training metadata available")
        
        # Model-specific assessments
        print(f"\n✅ VERIFICATION SUMMARY:")
        
        has_model = ('model_state_dict' in ckpt or 'model' in ckpt) if isinstance(ckpt, dict) else True
        has_metrics = metrics_found if isinstance(ckpt, dict) else False
        has_epoch = ('epoch' in ckpt) if isinstance(ckpt, dict) else False
        
        issues = []
        if not has_model:
            issues.append("Missing model weights")
        if not has_metrics:
            issues.append("Missing performance metrics")
        if not has_epoch:
            issues.append("Missing epoch information")
        
        if isinstance(ckpt, dict) and 'epoch' in ckpt:
            if ckpt['epoch'] < 15:
                issues.append(f"Only trained for {ckpt['epoch']} epochs (may need more training)")
        
        if issues:
            print(f"   ⚠️  Issues found:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print(f"   ✅ Checkpoint appears complete and valid!")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to check one or multiple checkpoints"""
    
    print("\n" + "="*80)
    print("PYTORCH CHECKPOINT VERIFICATION TOOL")
    print("="*80)
    
    if len(sys.argv) < 2:
        print("\nUsage: python check_checkpoints.py <checkpoint1.pth> [checkpoint2.pt] [...]")
        print("\nOr provide checkpoint paths when prompted:")
        
        paths = []
        while True:
            path = input("\nEnter checkpoint path (or press Enter to finish): ").strip()
            if not path:
                break
            paths.append(path)
        
        if not paths:
            print("\n❌ No checkpoints provided!")
            return
    else:
        paths = sys.argv[1:]
    
    # Check each checkpoint
    results = {}
    for path in paths:
        valid = inspect_checkpoint(path)
        results[path] = valid
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for path, valid in results.items():
        status = "✅ VALID" if valid else "⚠️  CHECK NEEDED"
        print(f"   {Path(path).name}: {status}")
    
    print("\n" + "="*80)
    
    all_valid = all(results.values())
    if all_valid:
        print("✅ All checkpoints appear complete and ready to use!")
    else:
        print("⚠️  Some checkpoints may need attention - review details above")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()