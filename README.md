# Cardiovascular Disease Risk Prediction System

A production-ready web application for predicting cardiovascular disease (CVD) risk from retinal fundus images using deep learning models.

## 🏥 Overview

This system combines three specialized models to predict CVD risk:

1. **Hypertension Classification** (RETFound ViT)
   - Detects hypertensive retinopathy from single fundus image
   - Output: Binary classification + probability

2. **CIMT Regression** (Siamese Multimodal)
   - Predicts carotid intima-media thickness from bilateral images + clinical data
   - Output: Continuous CIMT value in mm

3. **Vessel Segmentation** (U-Net)
   - Segments retinal blood vessels and extracts clinical features
   - Output: Segmentation mask + 15 vessel features

4. **Fusion Meta-Classifier** (MLP)
   - Combines outputs from all three models for comprehensive CVD risk assessment
   - Input: 1425 concatenated features
   - Output: Binary CVD risk prediction + probability

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for development)
- Node.js 18+ (for frontend development)
- 4GB+ RAM (minimum)
- GPU optional but recommended for faster inference

### Installation & Running

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
cd /Users/karimabdallah/Desktop/490_project

# Ensure model files exist
# pth/hypertension.pt (1.5 GB)
# pth/cimt_reg.pth (318 MB)
# pth/vessel.pth (11 MB)
# fusion/fusion_cvd_noskewed.pth (9.9 MB)

# Compute normalization statistics (required once)
python backend/scripts/compute_normalization_stats.py

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

#### Option 2: Manual Setup (Development)

**Backend:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Compute normalization statistics
python scripts/compute_normalization_stats.py

# Run backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
REACT_APP_API_URL=http://localhost:8000 npm start

# Build for production
npm run build
```

## 📊 API Endpoints

### Health Check
```
GET /health
```

### Individual Models
```
POST /api/predict/htn
- Input: Single fundus image
- Output: Hypertension probability

POST /api/predict/cimt
- Input: Left + right images, age, gender
- Output: CIMT value in mm

POST /api/predict/vessel
- Input: Single fundus image
- Output: Segmentation mask + vessel features
```

### Fusion (Main)
```
POST /api/predict/fusion
- Input: Left + right images, age, gender
- Output: Complete CVD risk assessment
```

## 📁 Project Structure

```
490_project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Settings
│   │   ├── models/
│   │   │   ├── architectures.py # Model definitions
│   │   │   ├── model_loader.py  # Singleton loader
│   │   │   └── normalization.py # Feature normalization
│   │   ├── features/            # Feature extraction
│   │   ├── preprocessing/       # Image transforms
│   │   └── api/                 # API schemas & routes
│   ├── scripts/
│   │   └── compute_normalization_stats.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/               # React pages
│   │   ├── components/          # Reusable components
│   │   ├── services/            # API client
│   │   └── App.tsx
│   ├── public/
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── pth/                         # Model weights (not in repo)
├── fusion/                      # Fusion models (not in repo)
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Backend Settings

Edit `backend/app/config.py`:

```python
# Model paths
HTN_CHECKPOINT = "pth/hypertension.pt"
CIMT_CHECKPOINT = "pth/cimt_reg.pth"
VESSEL_CHECKPOINT = "pth/vessel.pth"
FUSION_CHECKPOINT = "fusion/fusion_cvd_noskewed.pth"

# Device
DEVICE = "cuda"  # or "cpu"

# Image settings
MAX_IMAGE_SIZE_MB = 10
```

### Frontend Configuration

Environment variables:
```bash
REACT_APP_API_URL=http://localhost:8000  # Backend URL
```

## 📈 Feature Extraction Pipeline

The fusion model requires exactly **1425 features**:

| Component | Features | Details |
|-----------|----------|---------|
| HTN | 1025 | 1 probability + 1024 embedding |
| CIMT | 129 | 1 prediction + 128 embedding |
| Vessel | 271 | 256 learned + 15 handcrafted |
| **Total** | **1425** | Concatenated features |

### Feature Normalization

Features are normalized using StandardScaler statistics computed from training data:

```
normalized = (features - mean) / (std + 1e-8)
```

Statistics are stored in `backend/normalization_stats.pkl`

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Run frontend tests
cd frontend
npm test
```

## 📋 Requirements

### Backend
- fastapi 0.104.1
- torch 2.0.0
- torchvision 0.15.0
- timm 0.9.12
- scikit-image 0.22.0
- numpy 1.24.3

### Frontend
- react 18.2.0
- react-router-dom 6.18.0
- axios 1.6.0
- @mui/material 5.14.0

## 🔐 Security & Privacy

⚠️ **Important:**
- **No image storage**: Images are processed in-memory and not saved
- **No PHI logging**: Patient data is never logged
- **Research use only**: Not approved for clinical diagnosis
- **HTTPS recommended**: Use HTTPS in production
- **Rate limiting**: Implement rate limiting for production

## ⚠️ Disclaimer

This is a **research tool** and should **NOT** be used for clinical diagnosis or treatment decisions. All predictions must be reviewed by qualified healthcare professionals. The accuracy and safety of these predictions should be verified with clinical assessment.

## 📝 Citation

If you use this system in research, please cite:

```
[Citation information to be added]
```

## 📞 Support & Contributing

For issues, questions, or contributions, please contact the development team.

## 📄 License

[License information to be added]

## 🙏 Acknowledgments

- RETFound for Vision Transformer backbone
- Figshare for China-Fundus-CIMT dataset
- FastAPI, React, and PyTorch communities

---

**Last Updated**: December 2024
**Version**: 1.0.0
