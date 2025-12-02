/**
 * Hypertension classification page
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CircularProgress,
  Grid,
  LinearProgress,
  Paper,
  Grow,
  Typography,
} from '@mui/material';
import ImageUpload from '../components/ImageUpload';
import api, { HTNPrediction } from '../services/api';

interface HomePageProps {
  apiReady: boolean;
}

const HTNPage: React.FC<HomePageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [image, setImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<HTNPrediction | null>(null);
  const [error, setError] = useState<string>('');

  const handleImageSelect = (file: File) => {
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
    setError('');
    setResult(null);
  };

  const handleClear = () => {
    if (imageUrl && imageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(imageUrl);
    }
    setImage(null);
    setImageUrl('');
    setError('');
  };

  const handleClearAll = () => {
    if (imageUrl && imageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(imageUrl);
    }
    setImage(null);
    setImageUrl('');
    setResult(null);
    setError('');
  };

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setError('');
    try {
      const response = await api.predictHTN(image);
      if (response.result) {
        setResult(response.result);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)',
        position: 'fixed',
        width: '100vw',
        height: '100vh',
        left: 0,
        top: 0,
        py: 0,
        overflowY: 'auto',
      }}
    >
      {/* Page Header */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #6A4DF5 0%, #A680FF 100%)',
          color: 'white',
          py: { xs: 4, md: 5 },
          px: { xs: 3, sm: 4 },
          mb: 6,
          borderBottomLeftRadius: { xs: 24, sm: 32 },
          borderBottomRightRadius: { xs: 24, sm: 32 },
          boxShadow: '0 8px 32px rgba(106, 77, 245, 0.25)',
        }}
      >
        <Box sx={{ maxWidth: 1200, mx: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: { xs: 2, sm: 4 } }}>
          <Box sx={{ flex: 1 }}>
            <Typography
              variant="h2"
              sx={{
                fontWeight: 800,
                mb: 1,
                fontSize: { xs: '1.8rem', sm: '2rem', md: '2.4rem' },
                letterSpacing: '-0.3px',
              }}
            >
              Hypertensive Retinopathy Detection
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.9, fontWeight: 400, fontSize: { xs: '0.95rem', sm: '1rem' }, lineHeight: 1.5 }}>
              AI-powered classification from retinal fundus imaging
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2, flexShrink: 0, flexWrap: { xs: 'wrap', sm: 'nowrap' } }}>
            <Button
              onClick={() => navigate('/')}
              disabled={loading}
              sx={{
                textTransform: 'none',
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                fontWeight: 600,
                transition: 'all 0.15s ease',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.3)',
                  transform: 'scale(1.02)',
                },
              }}
            >
              ← Home
            </Button>
            <Button
              onClick={handleClearAll}
              disabled={!image && !result && !error}
              variant="contained"
              sx={{
                backgroundColor: 'white',
                color: '#6A4DF5',
                fontWeight: 700,
                transition: 'all 0.15s ease',
                '&:hover': {
                  backgroundColor: '#f5f5f5',
                  transform: 'scale(1.02)',
                },
              }}
            >
              Clear All
            </Button>
          </Box>
        </Box>
      </Box>

      <Box sx={{ width: '100%', maxWidth: 800, mx: 'auto', px: { xs: 2, sm: 3 }, pb: 6 }}>
        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12}>
            <Paper
              sx={{
                p: { xs: 3, md: 4 },
                backgroundColor: '#ffffff',
                boxShadow: '0 4px 20px rgba(0,0,0,0.10)',
                border: '1px solid rgba(106, 77, 245, 0.08)',
                borderRadius: 3,
                maxWidth: '100%',
                mx: 'auto',
              }}
            >
              <Typography
                sx={{
                  mb: 3,
                  fontWeight: 700,
                  color: '#6A4DF5',
                  fontSize: '1.1rem',
                  letterSpacing: '-0.2px',
                }}
              >
                📸 Upload Fundus Image
              </Typography>
              <Box sx={{ width: '100%' }}>
                <ImageUpload
                  onImageSelect={handleImageSelect}
                  label="Retinal Fundus Image (Single Eye)"
                  imageUrl={imageUrl}
                  onClear={handleClear}
                  disabled={loading}
                />
              </Box>
              <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={handlePredict}
                  disabled={!image || loading || !apiReady}
                  size="large"
                  sx={{
                    fontWeight: 700,
                    fontSize: '1.05rem',
                    py: 1.75,
                    borderRadius: 2,
                    background: 'linear-gradient(135deg, #6A4DF5 0%, #8B6FFF 100%)',
                    boxShadow: '0 4px 16px rgba(106, 77, 245, 0.3)',
                    transition: 'all 0.15s ease',
                    '&:hover': {
                      transform: 'scale(1.02)',
                      boxShadow: '0 6px 24px rgba(106, 77, 245, 0.4)',
                      background: 'linear-gradient(135deg, #5939E0 0%, #7A5FEE 100%)',
                    },
                  }}
                >
                  {loading ? (
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <CircularProgress size={20} color="inherit" />
                      Analyzing...
                    </Box>
                  ) : (
                    '🔍 Analyze Image'
                  )}
                </Button>
                {image && (
                  <Button
                    variant="outlined"
                    fullWidth
                    onClick={handleClear}
                    disabled={loading}
                    sx={{ fontWeight: 600 }}
                  >
                    Clear
                  </Button>
                )}
              </Box>
              {error && (
                <Paper
                  sx={{
                    mt: 2,
                    p: 2,
                    backgroundColor: '#ffebee',
                    border: '1px solid #ef5350',
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="body2" sx={{ color: '#c62828', fontWeight: 500 }}>
                    {error}
                  </Typography>
                </Paper>
              )}
              {/* Results Section - now below input */}
              {result && (
                <Grow in={true} timeout={500}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 4 }}>
                    {/* Main Result Card */}
                    <Card
                      sx={{
                        p: 4,
                        background: result.prediction === 1
                          ? 'linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)'
                          : 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)',
                        border: `3px solid ${result.prediction === 1 ? '#d32f2f' : '#388e3c'}`,
                        borderRadius: 2,
                        boxShadow: result.prediction === 1
                          ? '0 8px 24px rgba(211, 47, 47, 0.2)'
                          : '0 8px 24px rgba(56, 142, 60, 0.2)',
                        transition: 'all 0.3s ease',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: -2,
                          left: -2,
                          right: -2,
                          bottom: -2,
                          background: result.prediction === 1
                            ? 'linear-gradient(135deg, #d32f2f, #f44336)'
                            : 'linear-gradient(135deg, #388e3c, #66bb6a)',
                          opacity: 0.1,
                          zIndex: -1,
                        },
                      }}
                    >
                      <Typography variant="overline" sx={{ mb: 1, fontWeight: 700, color: '#666', letterSpacing: 1, fontSize: '0.8rem' }}>
                        📊 Analysis Result
                      </Typography>
                      <Typography
                        variant="h2"
                        sx={{
                          fontWeight: 900,
                          mb: 2,
                          fontSize: '2.8rem',
                          color: result.prediction === 1 ? '#d32f2f' : '#388e3c',
                          animation: 'pulse 2s ease-in-out infinite',
                          '@keyframes pulse': {
                            '0%, 100%': { opacity: 1 },
                            '50%': { opacity: 0.8 },
                          },
                        }}
                      >
                        {result.prediction === 1 ? '⚠️ DETECTED' : '✓ NOT DETECTED'}
                      </Typography>
                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            Probability
                          </Typography>
                          <Typography variant="h6" sx={{ fontWeight: 700, color: '#0066cc' }}>
                            {(result.probability * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={result.probability * 100}
                          sx={{
                            height: 12,
                            borderRadius: 2,
                            backgroundColor: '#e0e0e0',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor:
                                result.probability > 0.7
                                  ? '#d32f2f'
                                  : result.probability > 0.4
                                  ? '#f57c00'
                                  : '#388e3c',
                            },
                          }}
                        />
                      </Box>
                      <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: 1.5, mt: 2, border: '1px solid rgba(0,0,0,0.05)' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
                          Model Confidence
                        </Typography>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: 700,
                            fontSize: '1.1rem',
                            color:
                              result.confidence === 'High'
                                ? '#d32f2f'
                                : result.confidence === 'Medium'
                                ? '#f57c00'
                                : '#388e3c',
                          }}
                        >
                          {result.confidence} Confidence
                        </Typography>
                      </Box>
                    </Card>
                    {/* Clinical Interpretation */}
                    <Paper sx={{ p: 3, backgroundColor: '#fafafa', border: '1px solid #e8e8e8', borderRadius: 1.5 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600, mb: 1.5, color: '#0066cc' }}>
                        Clinical Interpretation
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#555', lineHeight: 1.8, fontSize: '0.95rem' }}>
                        {result.label}
                      </Typography>
                    </Paper>
                  </Box>
                </Grow>
              )}
            </Paper>
          </Grid>
        </Grid>
        {/* Info Section */}
        <Box sx={{ mt: 5 }}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5', border: '1px solid #e0e0e0' }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: '#0066cc' }}>
              About This Analysis
            </Typography>
            <Typography variant="body2" paragraph sx={{ color: '#666', lineHeight: 1.6 }}>
              This model classifies hypertensive retinopathy from fundus images using the RETFound Vision Transformer backbone. It analyzes retinal vascular and structural features associated with hypertension to provide clinical decision support.
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mt: 2 }}>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, color: '#0066cc', mb: 0.5 }}>
                  Input
                </Typography>
                <Typography variant="body2" sx={{ color: '#666' }}>
                  Single fundus image (PNG or JPG, max 10MB)
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, color: '#0066cc', mb: 0.5 }}>
                  Output
                </Typography>
                <Typography variant="body2" sx={{ color: '#666' }}>
                  Binary classification with probability score
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default HTNPage;
