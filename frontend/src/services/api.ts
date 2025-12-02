/**
 * API service for communicating with backend
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// TYPES
// ============================================================================

export interface HTNPrediction {
  prediction: number;
  probability: number;
  confidence: string;
  label: string;
}

export interface CIMTResult {
  value_mm: number;
  risk_category: string;
  threshold_mm: number;
  clinical_significance: string;
}

export interface VesselFeatures {
  vessel_density: number;
  peripheral_density: number;
  avg_vessel_width: number;
  fractal_dimension: number;
  branching_density: number;
  avg_tortuosity: number;
}

export interface VesselSegmentation {
  vessel_density: number;
  features: VesselFeatures;
  segmentation_mask_base64: string;
}

export interface FusionPrediction {
  cvd_risk_prediction: number;
  cvd_probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  hypertension: HTNPrediction;
  cimt: CIMTResult;
  vessel: VesselSegmentation;
  contributing_factors: {
    hypertension_probability: number;
    cimt_elevated: boolean;
    vessel_abnormalities: number;
  };
  recommendation: string;
  processing_time_seconds: number;
}

export interface APIResponse<T> {
  status: 'success' | 'error';
  result?: T;
  error?: string;
  processing_time?: number;
}

export interface HealthCheckResponse {
  status: string;
  models_loaded: boolean;
  models: string[];
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

export const api = {
  // Health check
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await apiClient.get<HealthCheckResponse>('/health');
    return response.data;
  },

  // HTN prediction from single image
  async predictHTN(imageFile: File): Promise<APIResponse<HTNPrediction>> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await apiClient.post<APIResponse<HTNPrediction>>(
      '/api/predict/htn',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  // CIMT prediction from bilateral images
  async predictCIMT(
    leftImage: File,
    rightImage: File | null,
    age: number,
    gender: number
  ): Promise<APIResponse<CIMTResult>> {
    const formData = new FormData();
    formData.append('left_image', leftImage);
    if (rightImage) {
      formData.append('right_image', rightImage);
    }
    formData.append('age', age.toString());
    formData.append('gender', gender.toString());

    const response = await apiClient.post<APIResponse<CIMTResult>>(
      '/api/predict/cimt',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  // Vessel segmentation from single image
  async predictVessel(imageFile: File): Promise<APIResponse<VesselSegmentation>> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await apiClient.post<APIResponse<VesselSegmentation>>(
      '/api/predict/vessel',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  // Fusion prediction - MAIN ENDPOINT
  async predictFusion(
    leftImage: File,
    rightImage: File | null,
    age: number,
    gender: number
  ): Promise<APIResponse<FusionPrediction>> {
    const formData = new FormData();
    formData.append('left_image', leftImage);
    if (rightImage) {
      formData.append('right_image', rightImage);
    }
    formData.append('age', age.toString());
    formData.append('gender', gender.toString());

    const response = await apiClient.post<APIResponse<FusionPrediction>>(
      '/api/predict/fusion',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },
};

export default api;
