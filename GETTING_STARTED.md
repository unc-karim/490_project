# Getting Started Guide

## Prerequisites

Before starting, ensure you have:

1. **Model Files** (must be in place before running):
   - `pth/hypertension.pt` (1.5 GB)
   - `pth/cimt_reg.pth` (318 MB)
   - `pth/vessel.pth` (11 MB)
   - `fusion/fusion_cvd_noskewed.pth` (9.9 MB)

2. **System Requirements**:
   - Docker & Docker Compose (recommended)
   - OR Python 3.10+ & Node.js 18+
   - 4GB+ RAM
   - GPU (optional, recommended)

## Step 1: Prepare Model Files

Ensure all trained model files are in the correct directories:

```
490_project/
├── pth/
│   ├── hypertension.pt          (1.5 GB)
│   ├── cimt_reg.pth             (318 MB)
│   └── vessel.pth               (11 MB)
├── fusion/
│   └── fusion_cvd_noskewed.pth  (9.9 MB)
└── ...
```

## Step 2: Create Normalization Statistics

This is **CRITICAL** and must be done once before running the API.

### Option A: From Saved Training Data (Recommended)

If you have the training data with pre-extracted features:

```bash
cd /Users/karimabdallah/Desktop/490_project

# Copy pre-extracted features to fusion/ directory if available
# (This comes from running the notebooks)

# Run the normalization script
python backend/scripts/compute_normalization_stats.py
```

### Option B: Using Synthetic Statistics (For Testing)

The script creates synthetic statistics if training data isn't available:

```bash
python backend/scripts/compute_normalization_stats.py
```

⚠️ **Note**: Synthetic statistics may produce inaccurate predictions. Use real training data for production.

### Option C: Manual Extraction from Notebook

If you have Google Colab access:

1. Run `deep_fusion_notskewed.ipynb` in Colab
2. The notebook computes training features: `X_train_fusion` shape (2322, 1425)
3. Download the computed statistics
4. Save to `backend/normalization_stats.pkl`

## Step 3: Choose Installation Method

### Method A: Docker (Recommended - One Command)

```bash
cd /Users/karimabdallah/Desktop/490_project

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Access**:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Method B: Manual Installation (Development)

#### Backend Setup

```bash
cd /Users/karimabdallah/Desktop/490_project/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend is ready at**: http://localhost:8000

#### Frontend Setup (New Terminal)

```bash
cd /Users/karimabdallah/Desktop/490_project/frontend

# Install dependencies
npm install

# Set API URL
export REACT_APP_API_URL=http://localhost:8000

# Start development server
npm start
```

**Frontend will open at**: http://localhost:3000

## Step 4: Verify Installation

### Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "models": ["htn", "cimt", "vessel", "fusion"]
}
```

### Check API Documentation

Open: http://localhost:8000/docs

You should see all available endpoints with interactive testing capability.

### Test Frontend

Open: http://localhost:3000

You should see:
- Navigation bar with model options
- Home page with landing section
- "Full Assessment" button working

## Step 5: Run Predictions

### Using Web Interface (Easiest)

1. Go to http://localhost:3000
2. Click "Full Assessment" or choose individual model
3. Upload fundus images
4. Enter clinical data (age, gender)
5. Click "Predict"
6. View results

### Using API Directly (cURL)

**Health Check**:
```bash
curl http://localhost:8000/health
```

**HTN Classification**:
```bash
curl -X POST http://localhost:8000/api/predict/htn \
  -F "file=@path/to/image.jpg"
```

**CIMT Prediction**:
```bash
curl -X POST http://localhost:8000/api/predict/cimt \
  -F "left_image=@left.jpg" \
  -F "right_image=@right.jpg" \
  -F "age=65" \
  -F "gender=1"
```

**Fusion Prediction**:
```bash
curl -X POST http://localhost:8000/api/predict/fusion \
  -F "left_image=@left.jpg" \
  -F "right_image=@right.jpg" \
  -F "age=65" \
  -F "gender=1"
```

## Common Issues & Solutions

### Issue 1: Model Files Not Found

**Error**: `FileNotFoundError: Models not found`

**Solution**:
- Verify model files exist in correct directories
- Check paths in `backend/app/config.py`
- Models must be in `pth/` and `fusion/` directories

### Issue 2: Normalization Stats Not Found

**Error**: `Normalization stats not loaded!`

**Solution**:
```bash
# Run this to create the stats file
python backend/scripts/compute_normalization_stats.py
```

### Issue 3: Out of Memory (OOM)

**Error**: `CUDA out of memory` or `MemoryError`

**Solution**:
- Run backend on CPU: Set `DEVICE=cpu` in config.py
- Reduce batch size (API processes one image at a time, so this is usually not the issue)
- Use smaller image sizes

### Issue 4: API Connection Refused

**Error**: `Connection refused` or `Failed to fetch`

**Solution**:
- Check backend is running: `curl http://localhost:8000/health`
- Check frontend API URL: `echo $REACT_APP_API_URL`
- If using Docker, ensure both containers are running: `docker-compose ps`

### Issue 5: Slow Predictions

**Expected times**:
- Single model: 0.3-0.5 seconds
- Fusion: 1-2 seconds
- Total with upload: 2-5 seconds

**If much slower**:
- Check GPU usage: `nvidia-smi`
- Verify running on CPU is not bottleneck
- Check network latency if using remote API

## Next Steps

1. **Test with Sample Images**: Use provided examples or fundus image databases
2. **Integration Testing**: Test all endpoints thoroughly
3. **Performance Tuning**: Optimize inference based on your hardware
4. **Production Deployment**:
   - Use HTTPS
   - Implement rate limiting
   - Add authentication if needed
   - Use proper logging
   - Monitor resource usage

## Development Tips

### Running Tests

```bash
cd backend
pytest tests/
```

### Viewing Backend Logs

```bash
# Docker
docker-compose logs -f backend

# Manual
# Check terminal where backend is running
```

### Frontend Debug Mode

```bash
# In browser dev tools (F12 -> Console)
# Or set environment variable
export DEBUG=true
```

### Restarting Services

```bash
# Docker
docker-compose restart backend
docker-compose restart frontend

# Manual
# Stop and restart the terminal running each service
```

## Important Notes

⚠️ **Before Production**:
1. Use real training data for normalization statistics
2. Implement HTTPS/SSL
3. Add authentication & authorization
4. Implement rate limiting
5. Set up monitoring and logging
6. Test with diverse fundus images
7. Have healthcare professionals validate results

⚠️ **Clinical Disclaimer**:
This is a **research tool** and should **NOT** be used for clinical diagnosis. All results must be reviewed by qualified healthcare professionals.

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Verify model files are present
3. Run health check: `curl http://localhost:8000/health`
4. Check API docs: http://localhost:8000/docs

---

**Congratulations!** Your CVD Prediction System is now running. 🎉
