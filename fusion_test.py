#!/usr/bin/env python3
"""
Deep inspection of fusion checkpoint
"""

import torch
import sys
from pathlib import Path

def inspect_fusion(ckpt_path):
    print("\n" + "="*80)
    print("FUSION CHECKPOINT DEEP INSPECTION")
    print("="*80)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print(f"\n📦 Top-level keys: {list(ckpt.keys())}")
    
    # Check best_val_auc
    if 'best_val_auc' in ckpt:
        print(f"\n🎯 Best Validation AUC: {ckpt['best_val_auc']}")
        if isinstance(ckpt['best_val_auc'], (int, float)):
            if ckpt['best_val_auc'] > 0.7:
                print("   ✅ Good performance!")
            elif ckpt['best_val_auc'] > 0.5:
                print("   ⚠️  Moderate performance")
            else:
                print("   ❌ Poor performance - may need retraining")
    
    # Check training history
    if 'training_history' in ckpt:
        history = ckpt['training_history']
        print(f"\n📈 Training History:")
        print(f"   Type: {type(history)}")
        
        if isinstance(history, dict):
            print(f"   Keys: {list(history.keys())}")
            
            # Check for epoch info
            for key in ['epoch', 'epochs', 'train_loss', 'val_loss']:
                if key in history:
                    data = history[key]
                    if isinstance(data, list):
                        print(f"   {key}: {len(data)} epochs recorded")
                        if len(data) > 0:
                            print(f"      First: {data[0]:.4f}" if isinstance(data[0], (int, float)) else f"      First: {data[0]}")
                            print(f"      Last: {data[-1]:.4f}" if isinstance(data[-1], (int, float)) else f"      Last: {data[-1]}")
                    else:
                        print(f"   {key}: {data}")
            
            # Check for AUC history
            for key in ['val_auc', 'val_auroc', 'auc', 'auroc']:
                if key in history:
                    data = history[key]
                    if isinstance(data, list):
                        print(f"   {key}: Best = {max(data):.4f}, Latest = {data[-1]:.4f}")
        
        elif isinstance(history, list):
            print(f"   Length: {len(history)} epochs")
            if len(history) > 0:
                print(f"   Sample entry: {history[0]}")
    
    # Check model config
    if 'model_config' in ckpt:
        config = ckpt['model_config']
        print(f"\n🏗️  Model Config:")
        print(f"   Type: {type(config)}")
        if isinstance(config, dict):
            for k, v in config.items():
                print(f"   {k}: {v}")
    
    # Check timestamp
    if 'timestamp' in ckpt:
        print(f"\n⏰ Timestamp: {ckpt['timestamp']}")
    
    # Model architecture details
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        print(f"\n🔍 Model Layers:")
        for i, (name, param) in enumerate(list(state.items())[:10]):  # First 10 layers
            print(f"   {name}: {param.shape}")
        if len(state) > 10:
            print(f"   ... ({len(state) - 10} more layers)")
    
    # Final assessment
    print(f"\n" + "="*80)
    print("ASSESSMENT:")
    print("="*80)
    
    has_weights = 'model_state_dict' in ckpt
    has_auc = 'best_val_auc' in ckpt
    has_history = 'training_history' in ckpt
    
    print(f"   Model weights: {'✅' if has_weights else '❌'}")
    print(f"   Performance metric: {'✅' if has_auc else '❌'}")
    print(f"   Training history: {'✅' if has_history else '❌'}")
    
    if has_weights and has_auc:
        print(f"\n   ✅ Fusion model appears to be trained and saved correctly!")
        print(f"   ✅ Ready for inference!")
    elif has_weights:
        print(f"\n   ⚠️  Model has weights but missing performance info")
        print(f"   → Still usable for inference, but can't verify quality")
    else:
        print(f"\n   ❌ Model appears incomplete")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_fusion.py <fusion_checkpoint.pth>")
        sys.exit(1)
    
    inspect_fusion(sys.argv[1])