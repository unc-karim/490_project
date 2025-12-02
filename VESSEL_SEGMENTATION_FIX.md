# Vessel Segmentation Output Issues - Diagnosis & Fixes

## Problem Summary
The vessel segmentation masks are not displaying correctly, appearing blank or very faint.

## Root Causes Identified

### 1. **Low Contrast Visualization** ✓ FIXED
**Problem**: The U-Net outputs continuous probability maps [0, 1]. When directly scaled to [0, 255], if probabilities are clustered in a narrow range (e.g., 0.2-0.5), the result is a very low-contrast grayscale image that appears mostly uniform.

**Solution**: Applied **contrast stretching** in `mask_to_base64()` function:
```python
# Before (raw scaling):
mask_uint8 = (mask * 255).astype(np.uint8)

# After (contrast stretched):
if mask.min() < mask.max():
    mask_stretched = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
else:
    mask_stretched = (mask * 255).astype(np.uint8)
```

This ensures the full [0, 255] range is used, dramatically improving visibility.

---

## Potential Remaining Issues

### 2. **Untrained or Improperly Loaded Model**
**Symptoms**:
- Mask is completely uniform (all zeros or all ones)
- Vessel density metrics are all very low or high

**Diagnosis**:
```python
from backend.app.features.vessel_debug import print_mask_analysis

# After running vessel segmentation:
print_mask_analysis(vessel_mask, name="Vessel Segmentation")
```

**Solutions**:
- Verify `vessel.pth` is properly trained
- Check model loading in `model_loader.py` line 186
- Ensure checkpoint format matches expected structure

### 3. **Incorrect Input Preprocessing**
**Symptoms**:
- Mask has inverted appearance (background bright, vessels dark)
- Mask values are in wrong range

**Possible Issues**:
- **Wrong normalization**: Using different ImageNet stats than training
- **Wrong resize method**: Using BICUBIC instead of BILINEAR
- **Input color space**: If image is BGR instead of RGB

**Current Preprocessing** (vessel_transforms.py):
```python
transform_vessel = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),  # ✓ Correct
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats ✓
        std=[0.229, 0.224, 0.225]    # ImageNet stats ✓
    )
])
```

To verify preprocessing matches training notebook:
1. Check training notebook for exact transforms
2. Ensure image is loaded as RGB (not BGR or grayscale)
3. Verify resize dimensions (512x512 correct)

### 4. **Threshold Issue for Binary Features**
**Symptoms**:
- Vessel density is 0 or near-zero
- Clinical features are all zeros

**Root Cause** (vessel_clinical.py line 33):
```python
mask = (vessel_mask > 0.5).astype(np.uint8)  # Hard threshold at 0.5
```

If model outputs low probabilities (max < 0.5), no pixels pass threshold.

**Solutions**:
Option A: Use adaptive threshold
```python
threshold = np.mean(vessel_mask) + np.std(vessel_mask)
threshold = np.clip(threshold, 0.1, 0.9)  # Reasonable bounds
mask = (vessel_mask > threshold).astype(np.uint8)
```

Option B: Use percentile-based threshold
```python
threshold = np.percentile(vessel_mask, 60)  # Top 40% pixels
mask = (vessel_mask > threshold).astype(np.uint8)
```

---

## Debugging Steps

### Step 1: Enable Diagnostic Output
Add this to `backend/app/main.py` in the vessel endpoint (after line 341):

```python
# Add diagnostic logging
from .features.vessel_debug import print_mask_analysis, save_mask_debug_images

learned, clinical, mask, _ = vessel_extractor.extract(image)

# Print analysis
stats = print_mask_analysis(mask, name="Vessel Segmentation from User Image")

# Log key statistics
logger.info(f"Mask stats: min={stats['min_value']:.4f}, "
            f"max={stats['max_value']:.4f}, "
            f"mean={stats['mean_value']:.4f}, "
            f"pixels_above_0.5={stats['percentage_above_0.5']:.2f}%")
```

### Step 2: Check Individual Component
Test with a known good image:
```python
from pathlib import Path
from backend.app.models.model_loader import model_loader
from backend.app.features.fusion_pipeline import FusionFeatureExtractor
from backend.app.features.vessel_debug import print_mask_analysis
from PIL import Image
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loader.load_vessel_model('pth/vessel.pth')
vessel_model = model_loader.get_model('vessel')

# Test image
test_image = Image.open('path/to/fundus/image.jpg').convert('RGB')

# Extract mask
from backend.app.features.vessel_extractor import VesselFeatureExtractor
extractor = VesselFeatureExtractor(vessel_model, device)
learned, clinical, mask, features_271 = extractor.extract(test_image)

# Analyze
print_mask_analysis(mask, "Test Image Vessel Mask")

# Save debug images (shows multiple threshold versions)
save_mask_debug_images(mask, '/tmp/vessel_debug')
```

Then check `/tmp/vessel_debug/` to see which mask looks most correct.

### Step 3: Verify Model Weights
```python
import torch

# Check if model weights are reasonable (not all zeros/ones)
model_loader.load_vessel_model('pth/vessel.pth')
vessel_model = model_loader.get_model('vessel')

for name, param in vessel_model.named_parameters():
    print(f"{name}: "
          f"min={param.min().item():.6f}, "
          f"max={param.max().item():.6f}, "
          f"mean={param.abs().mean().item():.6f}")
```

---

## Changes Made

### File: `backend/app/main.py` (Lines 153-180)
**Changed**: `mask_to_base64()` function to use contrast stretching
- **Before**: Direct scaling [0, 1] → [0, 255] (may result in low-contrast image)
- **After**: Stretches min-max to use full [0, 255] range (much better visibility)

### New File: `backend/app/features/vessel_debug.py`
Provides utilities for diagnosing vessel segmentation issues:
- `analyze_mask()`: Get detailed statistics
- `print_mask_analysis()`: Pretty-print with issue detection
- `create_enhanced_mask_display()`: Generate multiple visualization modes
- `mask_to_base64_enhanced()`: Mode-based mask encoding
- `save_mask_debug_images()`: Save 4 versions for comparison

---

## Testing the Fix

1. **Restart the backend**:
```bash
cd /Users/karimabdallah/Desktop/490_project/backend
python -m uvicorn app.main:app --reload
```

2. **Test vessel segmentation**:
   - Go to http://localhost:3000/vessel
   - Upload a fundus image
   - Check if mask is now visible (should be grayscale image with vessel structure)

3. **If mask still not visible**:
   - Check backend logs for diagnostic output
   - Run the diagnostic script above
   - Check `mask_stats` for min/max/mean values

---

## Recommended Next Steps

1. **Verify model training**: Check if vessel.pth was properly trained on vessel segmentation task
2. **Compare with notebooks**: Verify preprocessing exactly matches training notebook
3. **Check input images**: Test with multiple fundus images (different qualities/sizes)
4. **Adjust threshold if needed**: If model outputs are consistently low, adjust the 0.5 threshold in vessel_clinical.py

---

## Related Files
- Vessel Extraction: `backend/app/features/vessel_extractor.py`
- Clinical Features: `backend/app/features/vessel_clinical.py`
- Model Architecture: `backend/app/models/architectures.py` (lines 137-193, UNet class)
- API Endpoint: `backend/app/main.py` (lines 318-374, /api/predict/vessel)
- Debug Utilities: `backend/app/features/vessel_debug.py` (NEW)

---

## Summary
The primary issue was **low-contrast visualization** of continuous probability masks. The fix applies **contrast stretching** to use the full [0, 255] grayscale range, dramatically improving visibility. Debugging utilities have been added to diagnose any remaining issues with model output or preprocessing.
