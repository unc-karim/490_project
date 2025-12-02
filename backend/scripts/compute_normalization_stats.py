#!/usr/bin/env python3
"""
Compute normalization statistics from training data

MUST RUN BEFORE DEPLOYMENT!

This script extracts features from the training set used in the fusion model
and computes mean/std for normalization. These statistics are saved to
normalization_stats.pkl and used during inference.

Usage:
    python compute_normalization_stats.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import logging
from tqdm import tqdm

from app.config import settings
from app.models.model_loader import model_loader
from app.features.fusion_pipeline import FusionFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_stats():
    """
    Extract features from training data and compute normalization statistics

    IMPORTANT: This script needs access to the training data images from
    deep_fusion_notskewed.ipynb. If you don't have the training data locally,
    you have two options:

    Option 1: Extract from notebook
    - Run the notebook to extract training features
    - Load the features_extracted.pkl from notebook output
    - Use those features to compute stats

    Option 2: Use pre-computed statistics
    - If the notebook already saved features and stats
    - Copy them here

    This is a PLACEHOLDER that shows the expected format.
    """

    logger.info("=" * 80)
    logger.info("COMPUTING NORMALIZATION STATISTICS")
    logger.info("=" * 80)

    logger.warning("\n⚠️  IMPORTANT NOTES:")
    logger.warning("=" * 80)
    logger.warning(
        "This script requires training data to compute normalization statistics.\n"
        "\n"
        "The training data comes from the China-Fundus-CIMT dataset\n"
        "(https://figshare.com/articles/dataset/Fundus_CIMT_2903/27907056)\n"
        "\n"
        "OPTION 1: Extract from training notebook\n"
        "- Run deep_fusion_notskewed.ipynb in Google Colab or Jupyter\n"
        "- The notebook extracts X_train_fusion and saves statistics\n"
        "- Copy the computed stats file here\n"
        "\n"
        "OPTION 2: For demonstration\n"
        "- Generate synthetic statistics matching expected format\n"
        "- Features should be [N, 1425] where N is number of training samples\n"
        "\n"
        "OPTION 3: Manual computation\n"
        "- If you have extracted training features\n"
        "- Pass them as numpy array\n"
        "=" * 80
    )

    # Create placeholder statistics
    logger.info("\nCreating PLACEHOLDER statistics...")
    logger.info("(These are synthetic - use real training data for production!)")

    # For testing purposes, create zero-centered normalized features
    # In production, load real training data
    n_samples = 2322  # Training set size from notebook
    n_features = 1425

    # Generate synthetic features (this is just a placeholder)
    np.random.seed(42)
    X_train_synthetic = np.random.randn(n_samples, n_features).astype(np.float32)

    logger.info(f"Synthetic training features shape: {X_train_synthetic.shape}")

    # Compute statistics
    logger.info("\nComputing normalization statistics...")
    mean = X_train_synthetic.mean(axis=0).astype(np.float32)
    std = X_train_synthetic.std(axis=0).astype(np.float32)

    logger.info(f"Mean shape: {mean.shape}, range: [{mean.min():.6f}, {mean.max():.6f}]")
    logger.info(f"Std shape: {std.shape}, range: [{std.min():.6f}, {std.max():.6f}]")

    # Save statistics
    stats = {'mean': mean, 'std': std}
    output_path = settings.NORMALIZATION_STATS_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)

    logger.info(f"\n✓ Saved normalization stats to {output_path}")

    # Verify
    logger.info("\nVerifying saved statistics...")
    with open(output_path, 'rb') as f:
        stats_loaded = pickle.load(f)

    assert np.allclose(stats_loaded['mean'], mean), "Mean mismatch after loading"
    assert np.allclose(stats_loaded['std'], std), "Std mismatch after loading"

    logger.info("✓ Verification successful")

    logger.info("\n" + "=" * 80)
    logger.info("✓ NORMALIZATION STATISTICS COMPUTED AND SAVED")
    logger.info("=" * 80)
    logger.info("\nFOR PRODUCTION USE:")
    logger.info("1. Run deep_fusion_notskewed.ipynb to get real training data")
    logger.info("2. Extract training features from the notebook")
    logger.info("3. Compute statistics using real data")
    logger.info("4. Replace synthetic stats with real statistics")
    logger.info("=" * 80)


def load_real_stats_from_notebook():
    """
    Alternative: Load real statistics from notebook extraction

    If you've already run the notebook and saved features, use this.
    """

    logger.info("Attempting to load real statistics from notebook...")

    # Check if notebook saved features_extracted.pkl
    notebook_features_path = settings.PROJECT_ROOT / "fusion" / "features_extracted.pkl"

    if notebook_features_path.exists():
        logger.info(f"Found notebook features at {notebook_features_path}")

        with open(notebook_features_path, 'rb') as f:
            features_data = pickle.load(f)

        if isinstance(features_data, np.ndarray):
            X_train = features_data
        else:
            logger.warning(f"Unexpected format: {type(features_data)}")
            logger.info("Using synthetic statistics instead")
            return False

        logger.info(f"Loaded features shape: {X_train.shape}")

        # Compute and save statistics
        mean = X_train.mean(axis=0).astype(np.float32)
        std = X_train.std(axis=0).astype(np.float32)

        stats = {'mean': mean, 'std': std}
        output_path = settings.NORMALIZATION_STATS_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(stats, f)

        logger.info(f"✓ Saved real statistics from notebook")
        return True

    else:
        logger.info("No notebook features found, using synthetic statistics")
        return False


if __name__ == "__main__":
    logger.info("Starting normalization statistics computation...")

    # Try to load real stats from notebook first
    # If not available, use synthetic (for testing)
    success = load_real_stats_from_notebook()

    if not success:
        compute_stats()

    logger.info("\n✓ Done! The API can now be started.")
    logger.info(f"Normalization stats file: {settings.NORMALIZATION_STATS_PATH}")
