"""
Model architectures - exact replicas from training notebooks
CRITICAL: Must match notebook implementations exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ============================================================================
# 1. HYPERTENSION MODEL (RETFound ViT)
# ============================================================================

class RETFoundClassifier(nn.Module):
    """
    Hypertension classification using RETFound Vision Transformer backbone.

    Input: [B, 3, 224, 224]
    Output: [B, 1] logits (binary classification)

    When return_embedding=True:
        Output: ([B, 1] logits, [B, 1024] embeddings)
    """

    def __init__(self, dropout=0.65):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool='token'
        )
        embed_dim = self.backbone.num_features  # Should be 1024

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)  # [B, 1024]
        logits = self.head(features)  # [B, 1]

        if return_embedding:
            return logits, features
        return logits


# ============================================================================
# 2. CIMT REGRESSION MODEL (Siamese Multimodal)
# ============================================================================

class SiameseMultimodalCIMTRegression(nn.Module):
    """
    CIMT regression using bilateral fundus images + clinical features.

    Input:
        - left_img: [B, 3, 512, 512]
        - right_img: [B, 3, 512, 512]
        - clinical: [B, 3] with [age/100, gender_01, 0]

    Output: [B, 1] CIMT prediction in mm

    When return_embedding=True:
        Output: ([B, 1] prediction, [B, 128] embeddings)
    """

    def __init__(self):
        super().__init__()

        # Shared backbone for both eyes
        self.backbone = timm.create_model(
            "seresnext50_32x4d",
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        backbone_out_dim = self.backbone.num_features  # 2048

        # Clinical feature processor
        CLINICAL_INPUT_DIM = 3
        CLINICAL_HIDDEN_DIM = 128
        DROPOUT_RATE = 0.5

        self.clinical_fc = nn.Sequential(
            nn.Linear(CLINICAL_INPUT_DIM, CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )

        # Fusion layers (bilateral features + clinical features)
        fusion_input_dim = backbone_out_dim * 2 + CLINICAL_HIDDEN_DIM  # 2048*2 + 128 = 4224
        FUSION_HIDDEN_DIMS = [512, 128]

        layers = []
        in_dim = fusion_input_dim
        for hidden_dim in FUSION_HIDDEN_DIMS:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # Final regression output
        self.fusion = nn.Sequential(*layers)

    def forward(self, left_img, right_img, clinical, return_embedding=False):
        # Extract features from both eyes (shared backbone)
        left_features = self.backbone(left_img)  # [B, 2048]
        right_features = self.backbone(right_img)  # [B, 2048]
        bilateral_features = torch.cat([left_features, right_features], dim=1)  # [B, 4096]

        # Process clinical features
        clinical_features = self.clinical_fc(clinical)  # [B, 128]

        # Fuse all features
        fused = torch.cat([bilateral_features, clinical_features], dim=1)  # [B, 4224]

        if return_embedding:
            # Extract embedding before final layer (128-dim)
            embedding = fused
            for layer in self.fusion[:-1]:  # All layers except last
                embedding = layer(embedding)
            prediction = self.fusion[-1](embedding)
            return prediction, embedding

        return self.fusion(fused)


# ============================================================================
# 3. VESSEL SEGMENTATION MODEL (U-Net)
# ============================================================================

class UNet(nn.Module):
    """
    U-Net for vessel segmentation.

    Input: [B, 3, 512, 512]
    Output: [B, 1, 512, 512] segmentation mask

    When return_features=True:
        Output: [B, 256] encoder features
    """

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = CBR(in_ch, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = CBR(256 + 128, 128)
        self.dec2 = CBR(128 + 64, 64)

        # Final output
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x, return_features=False):
        # Encoder
        e1 = self.enc1(x)  # [B, 64, 512, 512]
        e2 = self.enc2(self.pool(e1))  # [B, 128, 256, 256]
        e3 = self.enc3(self.pool(e2))  # [B, 256, 128, 128]

        if return_features:
            # Return encoder features: global average pool to [B, 256]
            features = F.adaptive_avg_pool2d(e3, (1, 1)).flatten(1)
            return features

        # Decoder
        d3 = self.up(e3)  # [B, 256, 256, 256]
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # [B, 128, 256, 256]

        d2 = self.up(d3)  # [B, 128, 512, 512]
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # [B, 64, 512, 512]

        # Final output
        return self.final(d2)  # [B, 1, 512, 512]


# ============================================================================
# 4. FUSION META-CLASSIFIER
# ============================================================================

class FusionMetaClassifier(nn.Module):
    """
    Meta-classifier for CVD risk prediction from multi-modal features.

    Input: [B, 1425] concatenated features
        - HTN: 1025 (1 prob + 1024 emb)
        - CIMT: 129 (1 pred + 128 emb)
        - Vessel: 271 (256 learned + 15 clinical)

    Output: [B, 1] CVD risk probability
    """

    def __init__(self, input_dim=1425, hidden_dims=None, dropout=0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
