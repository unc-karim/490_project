"""
Vessel segmentation debugging utilities

This module helps diagnose issues with vessel segmentation masks
"""

import numpy as np
from PIL import Image
import io
import base64


def analyze_mask(mask: np.ndarray, name: str = "Mask") -> dict:
    """
    Analyze vessel mask to diagnose issues

    Args:
        mask: Segmentation mask [H, W] with values in [0, 1]
        name: Name of mask for reporting

    Returns:
        Dictionary with diagnostic information
    """

    stats = {
        'name': name,
        'shape': mask.shape,
        'dtype': str(mask.dtype),
        'min_value': float(np.min(mask)),
        'max_value': float(np.max(mask)),
        'mean_value': float(np.mean(mask)),
        'median_value': float(np.median(mask)),
        'std_value': float(np.std(mask)),
        'percentage_nonzero': float(100 * np.count_nonzero(mask) / mask.size),
        'percentage_above_0.5': float(100 * np.count_nonzero(mask > 0.5) / mask.size),
        'percentage_above_0.3': float(100 * np.count_nonzero(mask > 0.3) / mask.size),
        'percentage_above_0.1': float(100 * np.count_nonzero(mask > 0.1) / mask.size),
    }

    return stats


def print_mask_analysis(mask: np.ndarray, name: str = "Mask"):
    """Pretty print mask analysis"""

    stats = analyze_mask(mask, name)

    print(f"\n{'='*60}")
    print(f"MASK ANALYSIS: {stats['name']}")
    print(f"{'='*60}")
    print(f"Shape:                {stats['shape']}")
    print(f"Data Type:            {stats['dtype']}")
    print(f"\nValue Statistics (Expected [0, 1]):")
    print(f"  Min:                {stats['min_value']:.6f}")
    print(f"  Max:                {stats['max_value']:.6f}")
    print(f"  Mean:               {stats['mean_value']:.6f}")
    print(f"  Median:             {stats['median_value']:.6f}")
    print(f"  Std Dev:            {stats['std_value']:.6f}")
    print(f"\nMask Content Analysis:")
    print(f"  Non-zero pixels:    {stats['percentage_nonzero']:.2f}%")
    print(f"  Above 0.5 (binary): {stats['percentage_above_0.5']:.2f}%")
    print(f"  Above 0.3:          {stats['percentage_above_0.3']:.2f}%")
    print(f"  Above 0.1:          {stats['percentage_above_0.1']:.2f}%")

    # Diagnose issues
    print(f"\n{'POTENTIAL ISSUES':^60}")
    print(f"{'-'*60}")

    issues = []

    if stats['max_value'] < 0.1:
        issues.append("⚠️  MAX VALUE < 0.1: Model outputs very low probabilities")
        issues.append("   → Check model weights are loaded correctly")
        issues.append("   → Verify preprocessing matches training")

    if stats['percentage_above_0.5'] < 0.1:
        issues.append("⚠️  <0.1% pixels above 0.5: Almost no vessels detected")
        issues.append("   → Consider lowering threshold to 0.3 or 0.1")
        issues.append("   → Check if image preprocessing is correct")

    if stats['mean_value'] < 0.05:
        issues.append("⚠️  VERY LOW MEAN: Mask is mostly black (background)")
        issues.append("   → Possibly untrained model or wrong input preprocessing")

    if stats['mean_value'] > 0.95:
        issues.append("⚠️  VERY HIGH MEAN: Mask is mostly white (all vessels)")
        issues.append("   → Model might be inverted or broken")

    if stats['std_value'] < 0.01:
        issues.append("⚠️  VERY LOW STD DEV: No variation in mask")
        issues.append("   → Model outputs constant values (untrained)")

    if len(issues) == 0:
        print("✓ Mask analysis looks normal!")
    else:
        for issue in issues:
            print(issue)

    print(f"{'='*60}\n")

    return stats


def create_enhanced_mask_display(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Create an enhanced version of mask for better visualization

    Applies:
    - Adaptive thresholding
    - Histogram equalization
    - Contrast stretching

    Args:
        mask: Continuous mask [0, 1]
        threshold: Binary threshold

    Returns:
        Enhanced uint8 image [0, 255]
    """

    # Method 1: Binary with threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Method 2: Contrast stretched (use full 0-255 range)
    if mask.min() < mask.max():
        stretched = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
    else:
        stretched = (mask * 255).astype(np.uint8)

    # Method 3: Histogram equalization
    try:
        import cv2
        equalized = cv2.equalizeHist(stretched)
    except:
        equalized = stretched

    return binary_mask, stretched, equalized


def mask_to_base64_enhanced(mask: np.ndarray, mode: str = 'stretched') -> str:
    """
    Convert mask to base64 with optional enhancement

    Modes:
    - 'raw': Direct scaling [0,1] → [0, 255]
    - 'binary': Threshold at 0.5
    - 'stretched': Use full value range for contrast
    - 'adaptive': Smart thresholding based on mask statistics

    Args:
        mask: Continuous mask [0, 1]
        mode: Enhancement mode

    Returns:
        Data URI string
    """

    if mode == 'raw':
        # Default: direct scaling
        mask_uint8 = (mask * 255).astype(np.uint8)

    elif mode == 'binary':
        # Hard threshold at 0.5
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

    elif mode == 'stretched':
        # Use full value range
        if mask.min() < mask.max():
            mask_uint8 = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
        else:
            mask_uint8 = (mask * 255).astype(np.uint8)

    elif mode == 'adaptive':
        # Adaptive thresholding based on statistics
        threshold = np.mean(mask) + np.std(mask)
        threshold = max(0.1, min(0.9, threshold))  # Clamp to reasonable range
        mask_uint8 = (mask > threshold).astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: raw, binary, stretched, adaptive")

    # Convert to PIL and encode
    mask_image = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    buffer.seek(0)

    b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_string}"


def save_mask_debug_images(mask: np.ndarray, output_dir: str = '/tmp'):
    """
    Save multiple versions of mask for visual inspection

    Args:
        mask: Continuous mask [0, 1]
        output_dir: Directory to save images
    """

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Raw scaling
    raw = (mask * 255).astype(np.uint8)
    Image.fromarray(raw, mode='L').save(f"{output_dir}/mask_raw.png")

    # Binary threshold
    binary = (mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(binary, mode='L').save(f"{output_dir}/mask_binary_0.5.png")

    # Lower threshold
    binary_03 = (mask > 0.3).astype(np.uint8) * 255
    Image.fromarray(binary_03, mode='L').save(f"{output_dir}/mask_binary_0.3.png")

    # Stretched contrast
    if mask.min() < mask.max():
        stretched = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
    else:
        stretched = (mask * 255).astype(np.uint8)
    Image.fromarray(stretched, mode='L').save(f"{output_dir}/mask_stretched.png")

    print(f"✓ Debug images saved to {output_dir}")
    print(f"  - mask_raw.png")
    print(f"  - mask_binary_0.5.png")
    print(f"  - mask_binary_0.3.png")
    print(f"  - mask_stretched.png")
