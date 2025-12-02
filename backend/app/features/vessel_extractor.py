"""
Vessel feature extractor

Extracts:
- 256 learned features (from UNet encoder)
- 15 handcrafted clinical features
Total: 271 features
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ..preprocessing.transforms import transform_vessel
from .vessel_clinical import extract_clinical_vessel_features


class VesselFeatureExtractor:
    """Extract vessel features (256 learned + 15 clinical) from fundus image"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: UNet instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def extract(self, image):
        """
        Extract vessel features from image

        Args:
            image: PIL Image or path to image

        Returns:
            learned_features: np.array[256], encoder features from UNet
            clinical_features: np.array[15], handcrafted vessel features
            segmentation_mask: np.array[512, 512], vessel segmentation mask
            features_271: np.array[271], concatenated features
        """

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Apply transforms
        image_tensor = transform_vessel(image).unsqueeze(0).to(self.device)

        # Forward pass to get learned features
        with torch.no_grad():
            learned_feat = self.model(image_tensor, return_features=True)
            # learned_feat shape: [1, 256]
            learned_features_np = learned_feat.cpu().numpy()[0]  # [256]

        # Forward pass to get segmentation mask
        with torch.no_grad():
            mask_logits = self.model(image_tensor, return_features=False)
            # mask_logits shape: [1, 1, 512, 512]
            vessel_mask = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]  # [512, 512]

        # Extract handcrafted clinical features
        _, clinical_features_np = extract_clinical_vessel_features(vessel_mask)  # [15]

        # Concatenate: [learned, clinical] = [256, 15] = 271
        features_271 = np.concatenate([learned_features_np, clinical_features_np])

        return learned_features_np, clinical_features_np, vessel_mask, features_271

    def extract_batch(self, images):
        """
        Extract features from batch of images

        Args:
            images: list of PIL Images or paths

        Returns:
            learned_features: np.array[N, 256]
            clinical_features: np.array[N, 15]
            masks: list of np.array[512, 512]
            features_271: np.array[N, 271]
        """

        all_learned = []
        all_clinical = []
        all_masks = []

        for image in images:
            learned, clinical, mask, _ = self.extract(image)
            all_learned.append(learned)
            all_clinical.append(clinical)
            all_masks.append(mask)

        learned_features = np.array(all_learned)
        clinical_features = np.array(all_clinical)
        features_271 = np.concatenate([learned_features, clinical_features], axis=1)

        return learned_features, clinical_features, all_masks, features_271
