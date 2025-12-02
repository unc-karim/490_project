"""
Fusion feature extraction pipeline

Orchestrates all three models (HTN, CIMT, Vessel) to extract exactly 1425 features:
- HTN: 1025 (1 prob + 1024 emb)
- CIMT: 129 (1 pred + 128 emb)
- Vessel: 271 (256 learned + 15 clinical)
Total: 1425
"""

import numpy as np
from PIL import Image
from pathlib import Path
from .htn_extractor import HTNFeatureExtractor
from .cimt_extractor import CIMTFeatureExtractor
from .vessel_extractor import VesselFeatureExtractor


class FusionFeatureExtractor:
    """
    Extract complete fusion features (1425 dims) from bilateral fundus images
    and clinical metadata
    """

    def __init__(self, htn_model, cimt_model, vessel_model, device='cpu'):
        """
        Initialize with all three models

        Args:
            htn_model: RETFoundClassifier instance
            cimt_model: SiameseMultimodalCIMTRegression instance
            vessel_model: UNet instance
            device: 'cpu' or 'cuda'
        """
        self.htn_extractor = HTNFeatureExtractor(htn_model, device=device)
        self.cimt_extractor = CIMTFeatureExtractor(cimt_model, device=device)
        self.vessel_extractor = VesselFeatureExtractor(vessel_model, device=device)
        self.device = device

    def extract_all_features(self, left_image, right_image, age, gender):
        """
        Extract complete 1425-dimensional feature vector

        Args:
            left_image: PIL Image or path to left eye image
            right_image: PIL Image or path to right eye image (optional)
            age: int, age in years
            gender: int, 0=female, 1=male

        Returns:
            features: np.array[1425], concatenated features
            metadata: dict with individual model outputs for interpretation
        """

        # Load images if paths provided
        if isinstance(left_image, (str, Path)):
            left_image = Image.open(left_image).convert('RGB')
        if isinstance(right_image, (str, Path)):
            right_image = Image.open(right_image).convert('RGB')

        # Handle missing eyes: duplicate available eye
        if left_image is None and right_image is not None:
            left_image = right_image
        elif right_image is None and left_image is not None:
            right_image = left_image

        if left_image is None or right_image is None:
            raise ValueError("At least one image (left or right) must be provided")

        metadata = {}

        # ====================================================================
        # HTN Features: Average both eyes (1025 total)
        # ====================================================================
        htn_prob_L, htn_emb_L, _ = self.htn_extractor.extract(left_image)
        htn_prob_R, htn_emb_R, _ = self.htn_extractor.extract(right_image)

        # Average probability and embedding
        htn_prob_avg = (htn_prob_L + htn_prob_R) / 2.0
        htn_emb_avg = (htn_emb_L + htn_emb_R) / 2.0

        htn_features = np.concatenate([[htn_prob_avg], htn_emb_avg])  # [1025]
        metadata['htn_probability'] = float(htn_prob_avg)
        metadata['htn_probability_left'] = float(htn_prob_L)
        metadata['htn_probability_right'] = float(htn_prob_R)

        # ====================================================================
        # CIMT Features: Bilateral pair (129 total)
        # ====================================================================
        cimt_pred, cimt_emb, _ = self.cimt_extractor.extract(
            left_image, right_image, age, gender
        )

        cimt_features = np.concatenate([[cimt_pred], cimt_emb])  # [129]
        metadata['cimt_prediction_mm'] = float(cimt_pred)
        metadata['age'] = int(age)
        metadata['gender'] = int(gender)

        # ====================================================================
        # Vessel Features: Average both eyes (271 total)
        # ====================================================================
        vessel_learned_L, vessel_clinical_L, mask_L, _ = self.vessel_extractor.extract(
            left_image
        )
        vessel_learned_R, vessel_clinical_R, mask_R, _ = self.vessel_extractor.extract(
            right_image
        )

        # Average learned and clinical features
        vessel_learned_avg = (vessel_learned_L + vessel_learned_R) / 2.0
        vessel_clinical_avg = (vessel_clinical_L + vessel_clinical_R) / 2.0

        vessel_features = np.concatenate([vessel_learned_avg, vessel_clinical_avg])  # [271]
        metadata['vessel_density_left'] = float(vessel_clinical_L[0])
        metadata['vessel_density_right'] = float(vessel_clinical_R[0])
        metadata['vessel_density_avg'] = float(vessel_clinical_avg[0])

        # ====================================================================
        # CONCATENATE ALL: 1025 + 129 + 271 = 1425
        # ====================================================================
        features_1425 = np.concatenate([
            htn_features,       # [1025]
            cimt_features,      # [129]
            vessel_features     # [271]
        ]).astype(np.float32)

        assert features_1425.shape == (1425,), f"Feature shape mismatch: {features_1425.shape}"

        # Store segmentation masks for visualization
        metadata['vessel_mask_left'] = mask_L
        metadata['vessel_mask_right'] = mask_R

        return features_1425, metadata

    def extract_batch(self, left_images, right_images, ages, genders):
        """
        Extract features from batch of samples

        Args:
            left_images: list of PIL Images or paths
            right_images: list of PIL Images or paths
            ages: list of ages
            genders: list of genders

        Returns:
            features_batch: np.array[N, 1425]
            metadata_list: list of dict with per-sample metadata
        """

        assert len(left_images) == len(right_images) == len(ages) == len(genders)

        all_features = []
        all_metadata = []

        for left_img, right_img, age, gender in zip(left_images, right_images, ages, genders):
            features, metadata = self.extract_all_features(left_img, right_img, age, gender)
            all_features.append(features)
            all_metadata.append(metadata)

        features_batch = np.array(all_features, dtype=np.float32)
        return features_batch, all_metadata
