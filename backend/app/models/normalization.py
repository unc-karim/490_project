"""
Feature normalization using StandardScaler statistics

Applies (features - mean) / (std + 1e-8) normalization
using statistics computed from training data
"""

import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalize features using pre-computed statistics

    The mean and std should be computed from training data using:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
    """

    def __init__(self, stats_path: str = None):
        """
        Initialize normalizer

        Args:
            stats_path: Path to normalization_stats.pkl
                       Format: {'mean': np.array[1425], 'std': np.array[1425]}
        """

        self.mean = None
        self.std = None
        self.stats_path = stats_path

        if stats_path and Path(stats_path).exists():
            self.load_stats(stats_path)

    def load_stats(self, stats_path: str):
        """
        Load normalization statistics from pickle file

        Args:
            stats_path: Path to normalization_stats.pkl
        """

        stats_path = Path(stats_path)

        if not stats_path.exists():
            logger.warning(f"Normalization stats file not found: {stats_path}")
            logger.warning("You must run compute_normalization_stats.py before inference!")
            self.mean = None
            self.std = None
            return

        try:
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)

            self.mean = stats['mean'].astype(np.float32)
            self.std = stats['std'].astype(np.float32)

            assert self.mean.shape == (1425,), f"Mean shape mismatch: {self.mean.shape}"
            assert self.std.shape == (1425,), f"Std shape mismatch: {self.std.shape}"

            logger.info(f"✓ Loaded normalization stats from {stats_path}")
            logger.info(f"  Mean: shape {self.mean.shape}, range [{self.mean.min():.4f}, {self.mean.max():.4f}]")
            logger.info(f"  Std:  shape {self.std.shape}, range [{self.std.min():.4f}, {self.std.max():.4f}]")

        except Exception as e:
            logger.error(f"Failed to load normalization stats: {e}")
            raise

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization: (features - mean) / (std + 1e-8)

        Args:
            features: np.array or single sample
                     Shape: [1425] for single sample or [N, 1425] for batch

        Returns:
            normalized_features: Same shape as input, normalized

        Raises:
            RuntimeError: If stats not loaded
        """

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Normalization stats not loaded! "
                "Run compute_normalization_stats.py first"
            )

        features = np.asarray(features, dtype=np.float32)

        # Handle both single and batch inputs
        if features.ndim == 1:
            assert features.shape == (1425,), f"Feature shape mismatch: {features.shape}"
        elif features.ndim == 2:
            assert features.shape[1] == 1425, f"Feature shape mismatch: {features.shape}"
        else:
            raise ValueError(f"Invalid feature shape: {features.shape}")

        # Apply normalization
        normalized = (features - self.mean) / (self.std + 1e-8)

        return normalized.astype(np.float32)

    def denormalize(self, features_normalized: np.ndarray) -> np.ndarray:
        """
        Reverse normalization: features * std + mean

        Useful for interpretation or debugging

        Args:
            features_normalized: Normalized features

        Returns:
            original_features: Features in original scale
        """

        if self.mean is None or self.std is None:
            raise RuntimeError("Normalization stats not loaded")

        features_normalized = np.asarray(features_normalized, dtype=np.float32)

        # Reverse: x = z * std + mean
        denormalized = features_normalized * self.std + self.mean

        return denormalized.astype(np.float32)

    def is_ready(self) -> bool:
        """Check if normalizer is ready"""
        return self.mean is not None and self.std is not None

    def __repr__(self) -> str:
        if self.is_ready():
            return (
                f"FeatureNormalizer(mean_range=[{self.mean.min():.4f}, {self.mean.max():.4f}], "
                f"std_range=[{self.std.min():.4f}, {self.std.max():.4f}])"
            )
        else:
            return "FeatureNormalizer(not ready)"
