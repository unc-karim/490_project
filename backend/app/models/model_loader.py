"""
Model loading with singleton pattern

Handles efficient loading and caching of all 4 models
Special attention to memory management for 1.5GB HTN model
"""

import torch
from pathlib import Path
from typing import Optional, Dict
import logging

from .architectures import (
    RETFoundClassifier,
    SiameseMultimodalCIMTRegression,
    UNet,
    FusionMetaClassifier
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton for loading and managing all models

    Features:
    - Lazy loading (load on first request)
    - In-memory caching
    - Disabled gradients (30% memory savings)
    - Auto device detection (CUDA/CPU)
    """

    _instance = None
    _models: Dict[str, torch.nn.Module] = {}
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ModelLoader initialized with device: {self.device}")

    def load_htn_model(self, checkpoint_path: str) -> RETFoundClassifier:
        """
        Load HTN model with RETFound backbone

        CRITICAL: This is a 1.5GB model
        - Loaded only once
        - Cached in memory
        - No gradients

        Args:
            checkpoint_path: Path to hypertension.pt

        Returns:
            RETFoundClassifier instance ready for inference
        """

        if 'htn' in self._models:
            logger.info("HTN model already loaded (cached)")
            return self._models['htn']

        logger.info(f"Loading HTN model from {checkpoint_path}...")
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HTN checkpoint not found: {checkpoint_path}")

        try:
            # Load RETFound backbone weights first
            logger.info("Loading RETFound backbone...")
            try:
                from huggingface_hub import hf_hub_download
                retfound_ckpt_path = hf_hub_download(
                    repo_id="YukunZhou/RETFound_mae_natureCFP",
                    filename="RETFound_mae_natureCFP.pth",
                    cache_dir="/tmp/.cache"
                )
                retfound_ckpt = torch.load(
                    retfound_ckpt_path,
                    map_location=self.device,
                    weights_only=False
                )
            except Exception as e:
                logger.warning(f"Could not load RETFound backbone from HuggingFace: {e}")
                logger.info("Proceeding without RETFound pre-trained weights")
                retfound_ckpt = None

            # Initialize model
            model = RETFoundClassifier(dropout=0.65).to(self.device)

            # Load RETFound backbone if available
            if retfound_ckpt is not None:
                retfound_state = retfound_ckpt.get("model", retfound_ckpt)
                backbone_state_dict = {}
                for key, value in retfound_state.items():
                    if not key.startswith("decoder"):
                        new_key = key.replace("encoder.", "")
                        backbone_state_dict[new_key] = value
                model.backbone.load_state_dict(backbone_state_dict, strict=False)

            # Load fine-tuned weights
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(state_dict, strict=False)

            # Disable gradients
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self._models['htn'] = model
            logger.info(f"✓ HTN model loaded successfully ({self.get_model_size('htn')} MB)")
            return model

        except Exception as e:
            logger.error(f"Failed to load HTN model: {e}")
            raise

    def load_cimt_model(self, checkpoint_path: str) -> SiameseMultimodalCIMTRegression:
        """
        Load CIMT regression model

        Args:
            checkpoint_path: Path to cimt_reg.pth

        Returns:
            SiameseMultimodalCIMTRegression instance ready for inference
        """

        if 'cimt' in self._models:
            logger.info("CIMT model already loaded (cached)")
            return self._models['cimt']

        logger.info(f"Loading CIMT model from {checkpoint_path}...")
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CIMT checkpoint not found: {checkpoint_path}")

        try:
            model = SiameseMultimodalCIMTRegression().to(self.device)

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            model.load_state_dict(state_dict, strict=False)

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self._models['cimt'] = model
            logger.info(f"✓ CIMT model loaded successfully ({self.get_model_size('cimt')} MB)")
            return model

        except Exception as e:
            logger.error(f"Failed to load CIMT model: {e}")
            raise

    def load_vessel_model(self, checkpoint_path: str) -> UNet:
        """
        Load vessel segmentation model (U-Net)

        Args:
            checkpoint_path: Path to vessel.pth

        Returns:
            UNet instance ready for inference
        """

        if 'vessel' in self._models:
            logger.info("Vessel model already loaded (cached)")
            return self._models['vessel']

        logger.info(f"Loading vessel model from {checkpoint_path}...")
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Vessel checkpoint not found: {checkpoint_path}")

        try:
            model = UNet(in_ch=3, out_ch=1).to(self.device)

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self._models['vessel'] = model
            logger.info(f"✓ Vessel model loaded successfully ({self.get_model_size('vessel')} MB)")
            return model

        except Exception as e:
            logger.error(f"Failed to load vessel model: {e}")
            raise

    def load_fusion_model(self, checkpoint_path: str) -> FusionMetaClassifier:
        """
        Load fusion meta-classifier

        Args:
            checkpoint_path: Path to fusion_cvd_noskewed.pth

        Returns:
            FusionMetaClassifier instance ready for inference
        """

        if 'fusion' in self._models:
            logger.info("Fusion model already loaded (cached)")
            return self._models['fusion']

        logger.info(f"Loading fusion model from {checkpoint_path}...")
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")

        try:
            model = FusionMetaClassifier(
                input_dim=1425,
                hidden_dims=[512, 256],
                dropout=0.3
            ).to(self.device)

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            model.load_state_dict(state_dict, strict=False)

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            self._models['fusion'] = model
            logger.info(f"✓ Fusion model loaded successfully ({self.get_model_size('fusion')} MB)")
            return model

        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")
            raise

    def load_all_models(self, htn_path: str, cimt_path: str, vessel_path: str, fusion_path: str):
        """
        Load all models at startup

        Args:
            htn_path: Path to hypertension.pt
            cimt_path: Path to cimt_reg.pth
            vessel_path: Path to vessel.pth
            fusion_path: Path to fusion_cvd_noskewed.pth
        """

        logger.info("=" * 80)
        logger.info("LOADING ALL MODELS")
        logger.info("=" * 80)

        try:
            self.load_htn_model(htn_path)
            self.load_cimt_model(cimt_path)
            self.load_vessel_model(vessel_path)
            self.load_fusion_model(fusion_path)

            self._loaded = True
            logger.info("=" * 80)
            logger.info("✓ ALL MODELS LOADED SUCCESSFULLY")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def get_model(self, model_name: str) -> torch.nn.Module:
        """Get cached model by name"""

        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self._models.keys())}")

        return self._models[model_name]

    def get_model_size(self, model_name: str) -> float:
        """Get model size in MB"""

        model = self._models.get(model_name)
        if model is None:
            return 0.0

        total_params = sum(p.numel() for p in model.parameters())
        # Assuming float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 ** 2)
        return round(size_mb, 2)

    def is_ready(self) -> bool:
        """Check if all models are loaded"""
        return self._loaded

    def clear_cache(self):
        """Clear cached models to free memory"""

        logger.info("Clearing model cache...")
        self._models.clear()
        self._loaded = False
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("✓ Model cache cleared")


# Global singleton instance
model_loader = ModelLoader()
