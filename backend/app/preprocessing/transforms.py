"""
Image preprocessing and transforms for each model
Must match notebook transforms exactly
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ============================================================================
# HTN MODEL TRANSFORMS
# ============================================================================

transform_htn = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

"""
HTN Transform Summary:
- Input: PIL Image or numpy array
- Resize to 224x224 (BICUBIC interpolation - matches notebook)
- Convert to tensor [3, 224, 224]
- Normalize with ImageNet stats
- Output: [3, 224, 224] tensor
"""


# ============================================================================
# CIMT MODEL TRANSFORMS
# ============================================================================

transform_cimt = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

"""
CIMT Transform Summary:
- Input: PIL Image or numpy array
- Resize to 512x512 (BILINEAR interpolation - matches notebook)
- Convert to tensor [3, 512, 512]
- Normalize with ImageNet stats
- Output: [3, 512, 512] tensor
"""


# ============================================================================
# VESSEL MODEL TRANSFORMS
# ============================================================================

transform_vessel = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # NO normalization - U-Net trained on raw pixel values [0, 1]
])

"""
Vessel Transform Summary:
- Input: PIL Image or numpy array
- Resize to 512x512 (BILINEAR interpolation - matches notebook)
- Convert to tensor [3, 512, 512]
- NO normalization applied - uses raw pixel values in [0, 1]
- Output: [3, 512, 512] tensor with values in [0, 1]
"""


# ============================================================================
# UTILITY: Get transforms by model name
# ============================================================================

def get_transform(model_name):
    """
    Get preprocessing transform for a specific model

    Args:
        model_name: 'htn', 'cimt', or 'vessel'

    Returns:
        Transform object
    """
    transforms_dict = {
        'htn': transform_htn,
        'cimt': transform_cimt,
        'vessel': transform_vessel,
    }

    if model_name.lower() not in transforms_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(transforms_dict.keys())}")

    return transforms_dict[model_name.lower()]
