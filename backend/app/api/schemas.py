"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class RiskLevel(str, Enum):
    """CVD risk level"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Gender(int, Enum):
    """Gender encoding"""
    FEMALE = 0
    MALE = 1


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Status: healthy or unhealthy")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    models: List[str] = Field(..., description="List of loaded models")


class HTNPredictionResponse(BaseModel):
    """HTN classification response"""

    prediction: int = Field(..., description="Prediction: 0=Normal, 1=Hypertensive")
    probability: float = Field(..., description="Probability of hypertension [0, 1]")
    confidence: str = Field(..., description="Confidence level: Low/Medium/High")
    label: str = Field(..., description="Prediction label")


class CIMTValue(BaseModel):
    """CIMT prediction details"""

    value_mm: float = Field(..., description="CIMT value in mm")
    risk_category: str = Field(..., description="Risk category: Normal/Elevated/High")
    threshold_mm: float = Field(..., description="Clinical threshold (0.9mm)")
    clinical_significance: str = Field(..., description="Clinical interpretation")


class VesselFeatures(BaseModel):
    """Vessel segmentation features"""

    vessel_density: float = Field(..., description="Overall vessel density")
    peripheral_density: float = Field(..., description="Peripheral vessel density")
    avg_vessel_width: float = Field(..., description="Average vessel width in pixels")
    fractal_dimension: float = Field(..., description="Fractal dimension")
    branching_density: float = Field(..., description="Branching point density")
    avg_tortuosity: float = Field(..., description="Average vessel tortuosity")


class VesselSegmentationResponse(BaseModel):
    """Vessel segmentation response"""

    vessel_density: float = Field(..., description="Overall vessel density")
    features: VesselFeatures = Field(..., description="Extracted vessel features")
    segmentation_mask_base64: str = Field(..., description="Base64-encoded segmentation mask")


class ContributingFactors(BaseModel):
    """Contributing factors to CVD risk"""

    hypertension_probability: float = Field(..., description="HTN probability")
    cimt_elevated: bool = Field(..., description="CIMT above threshold")
    vessel_abnormalities: float = Field(..., description="Vessel abnormality score")


class FusionPredictionResponse(BaseModel):
    """
    Complete CVD risk prediction from fusion model

    This is the MAIN response containing all information
    """

    cvd_risk_prediction: int = Field(..., description="Prediction: 0=Low Risk, 1=High Risk")
    cvd_probability: float = Field(..., description="CVD risk probability [0, 1]")
    risk_level: RiskLevel = Field(..., description="Risk level: Low/Medium/High")

    # Individual model results
    hypertension: HTNPredictionResponse = Field(..., description="HTN classification result")
    cimt: CIMTValue = Field(..., description="CIMT regression result")
    vessel: VesselSegmentationResponse = Field(..., description="Vessel segmentation result")

    # Contributing factors
    contributing_factors: ContributingFactors = Field(..., description="Factors contributing to risk")

    # Clinical recommendation
    recommendation: str = Field(..., description="Clinical recommendation")

    # Processing metadata
    processing_time_seconds: float = Field(..., description="Total processing time")


class APIResponse(BaseModel):
    """Generic API response wrapper"""

    status: str = Field(..., description="success or error")
    result: Optional[Dict] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if status=error")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


# ============================================================================
# REQUEST MODELS (for documentation)
# ============================================================================

class ImageUploadRequest(BaseModel):
    """
    File upload request
    Note: File upload is multipart/form-data, not JSON
    This is for documentation only
    """

    image: str = Field(..., description="Image file (base64 or file)")
    class Config:
        json_schema_extra = {
            "example": {
                "image": "<file_content>"
            }
        }


class BilateralImageRequest(BaseModel):
    """Bilateral image request for documentation"""

    left_image: str = Field(..., description="Left eye image")
    right_image: Optional[str] = Field(None, description="Right eye image (optional)")
    age: int = Field(..., description="Age in years", ge=1, le=150)
    gender: Gender = Field(..., description="Gender: 0=Female, 1=Male")
