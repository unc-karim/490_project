"""
HTN (Hypertension) feature extractor

Extracts:
- 1 probability (sigmoid of logits)
- 1024 embedding (from ViT backbone)
Total: 1025 features
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ..preprocessing.transforms import transform_htn


class HTNFeatureExtractor:
    """Extract HTN features (1 prob + 1024 embedding) from fundus image"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: RETFoundClassifier instance
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.eval()

    def extract(self, image):
        """
        Extract HTN features from image

        Args:
            image: PIL Image or path to image

        Returns:
            prob: float, probability of hypertension [0, 1]
            embedding: np.array[1024], backbone features
            features_1025: np.array[1025], concatenated [prob, embedding]
        """

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Apply transforms
        image_tensor = transform_htn(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits, embedding = self.model(image_tensor, return_embedding=True)

        # Extract values
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        embedding_np = embedding.cpu().numpy()[0]  # [1024]

        # Concatenate: [prob, embedding] = [1, 1024] = 1025
        features_1025 = np.concatenate([[prob], embedding_np])

        return prob, embedding_np, features_1025

    def extract_batch(self, images):
        """
        Extract features from batch of images

        Args:
            images: list of PIL Images or paths

        Returns:
            probs: np.array[N], probabilities
            embeddings: np.array[N, 1024]
            features_1025: np.array[N, 1025]
        """

        all_probs = []
        all_embeddings = []

        for image in images:
            prob, embedding, _ = self.extract(image)
            all_probs.append(prob)
            all_embeddings.append(embedding)

        probs = np.array(all_probs)
        embeddings = np.array(all_embeddings)
        features_1025 = np.concatenate([probs[:, np.newaxis], embeddings], axis=1)

        return probs, embeddings, features_1025
