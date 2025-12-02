"""
CIMT (Carotid Intima-Media Thickness) feature extractor

Extracts:
- 1 prediction (regression value in mm)
- 128 embedding (from fusion layer)
Total: 129 features
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ..preprocessing.transforms import transform_cimt


class CIMTFeatureExtractor:
    """Extract CIMT features (1 pred + 128 embedding) from bilateral fundus images"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: SiameseMultimodalCIMTRegression instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def extract(self, left_image, right_image, age, gender):
        """
        Extract CIMT features from bilateral fundus images

        Args:
            left_image: PIL Image or path to left eye image
            right_image: PIL Image or path to right eye image
            age: int, age in years
            gender: int, 0=female, 1=male

        Returns:
            pred: float, CIMT prediction in mm
            embedding: np.array[128], fusion layer embedding
            features_129: np.array[129], concatenated [pred, embedding]
        """

        # Load images if paths provided
        if isinstance(left_image, (str, Path)):
            left_image = Image.open(left_image).convert('RGB')
        if isinstance(right_image, (str, Path)):
            right_image = Image.open(right_image).convert('RGB')

        # Handle missing eyes (use available eye for both)
        if left_image is None and right_image is not None:
            left_image = right_image
        elif right_image is None and left_image is not None:
            right_image = left_image

        if left_image is None or right_image is None:
            raise ValueError("At least one image (left or right) must be provided")

        # Apply transforms
        left_tensor = transform_cimt(left_image).unsqueeze(0).to(self.device)
        right_tensor = transform_cimt(right_image).unsqueeze(0).to(self.device)

        # Create clinical features: [age/100, gender_one_hot, 0]
        # Format: [age/100.0, 1.0 if gender==1 else 0.0, 0.0]
        clinical = torch.tensor(
            [[age / 100.0, 1.0 if gender == 1 else 0.0, 0.0]],
            dtype=torch.float32,
            device=self.device
        )

        # Forward pass
        with torch.no_grad():
            pred, embedding = self.model(
                left_tensor, right_tensor, clinical,
                return_embedding=True
            )

        # Extract values
        pred_value = pred.cpu().numpy()[0, 0]
        embedding_np = embedding.cpu().numpy()[0]  # [128]

        # Concatenate: [pred, embedding] = [1, 128] = 129
        features_129 = np.concatenate([[pred_value], embedding_np])

        return pred_value, embedding_np, features_129

    def extract_batch(self, left_images, right_images, ages, genders):
        """
        Extract features from batch of bilateral images

        Args:
            left_images: list of PIL Images or paths
            right_images: list of PIL Images or paths
            ages: list of ages
            genders: list of genders

        Returns:
            preds: np.array[N], CIMT predictions
            embeddings: np.array[N, 128]
            features_129: np.array[N, 129]
        """

        assert len(left_images) == len(right_images) == len(ages) == len(genders)

        all_preds = []
        all_embeddings = []

        for left_img, right_img, age, gender in zip(left_images, right_images, ages, genders):
            pred, embedding, _ = self.extract(left_img, right_img, age, gender)
            all_preds.append(pred)
            all_embeddings.append(embedding)

        preds = np.array(all_preds)
        embeddings = np.array(all_embeddings)
        features_129 = np.concatenate([preds[:, np.newaxis], embeddings], axis=1)

        return preds, embeddings, features_129
