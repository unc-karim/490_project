/**
 * Vessel segmentation page
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CircularProgress,
  Container,
  Grid,
  Paper,
  Typography,
  Grow,
} from '@mui/material';
import ImageUpload from '../components/ImageUpload';
import api, { VesselSegmentation } from '../services/api';

interface VesselPageProps {
  apiReady: boolean;
}

const VesselPage: React.FC<VesselPageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [image, setImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VesselSegmentation | null>(null);
  const [error, setError] = useState('');

  const handleClearAll = () => {
    handleClear();
  };

  const handlePredict = async () => {
    if (!image) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.predictVessel(image);
      if (response.status === 'success' && response.result) {
        setResult(response.result);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setImage(null);
    if (imageUrl && imageUrl.startsWith('blob:')) {
      try { URL.revokeObjectURL(imageUrl); } catch (e) {}
    }
    setImageUrl('');
    setResult(null);
    setError('');
  };

  return (
    <Box sx={{ background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)', minHeight: '100vh' }}>
      {/* Header */}
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
        <Container maxWidth="lg">
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: { xs: 2, sm: 4 } }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h2" sx={{ fontWeight: 800, mb: 1, fontSize: { xs: '1.8rem', sm: '2rem', md: '2.4rem' }, letterSpacing: '-0.3px' }}>
                Vessel Segmentation
              </Typography>
              <Typography variant="body1" sx={{ opacity: 0.9, fontWeight: 400, fontSize: { xs: '0.95rem', sm: '1rem' }, lineHeight: 1.5 }}>
                Automatic analysis of retinal blood vessel features
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
                  '&:hover': { backgroundColor: 'rgba(255,255,255,0.3)', transform: 'scale(1.02)' },
                }}
              >
                ← Home
              </Button>
              <Button
                onClick={handleClearAll}
                disabled={loading}
                variant="contained"
                sx={{
                  backgroundColor: 'white',
                  color: '#6A4DF5',
                  fontWeight: 700,
                  transition: 'all 0.15s ease',
                  '&:hover': { backgroundColor: '#f5f5f5', transform: 'scale(1.02)' },
                }}
              >
                Clear All
              </Button>
            </Box>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ pb: 6 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, backgroundColor: '#f8faff', maxWidth: 700, mx: 'auto' }}>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: '#0066cc' }}>
                Fundus Image Upload
              </Typography>
              <ImageUpload
                onImageSelect={(file) => {
                  setImage(file);
                  setImageUrl(URL.createObjectURL(file));
                }}
                label="Retinal Fundus Image (Single Eye)"
                imageUrl={imageUrl}
              />
              <Button
                variant="contained"
                fullWidth
                onClick={handlePredict}
                disabled={!image || loading || !apiReady}
                size="large"
                sx={{ mt: 3, fontWeight: 600, fontSize: '1rem' }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <CircularProgress size={20} color="inherit" />
                    Analyzing Vessels...
                  </Box>
                ) : (
                  '🔬 Segment Vessels'
                )}
              </Button>
              {image && (
                <Button
                  variant="outlined"
                  fullWidth
                  onClick={handleClearAll}
                  disabled={loading}
                  sx={{ fontWeight: 600, mt: 2 }}
                >
                  Clear
                </Button>
              )}
              {error && (
                <Paper sx={{ mt: 2, p: 2, backgroundColor: '#ffebee', border: '1px solid #ef5350', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ color: '#c62828', fontWeight: 500 }}>
                    {error}
                  </Typography>
                </Paper>
              )}
              {/* Results Section - now below input */}
              {result && (
                <Grow in={true} timeout={500}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 4 }}>
                    <Paper sx={{ p: 2.5 }}>
                      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: '#0066cc' }}>
                        Vessel Segmentation Mask
                      </Typography>
                      <Card sx={{ overflow: 'hidden' }}>
                        <img
                          src={result.segmentation_mask_base64}
                          alt="Vessel Segmentation"
                          style={{ width: '100%', display: 'block', borderRadius: 4 }}
                        />
                      </Card>
                    </Paper>
                    <Paper sx={{ p: 2.5 }}>
                      <Typography variant="h6" sx={{ mb: 2.5, fontWeight: 600, color: '#0066cc' }}>
                        Vessel Metrics
                      </Typography>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                        {/* Main Vessel Density */}
                        <Box sx={{ p: 1.5, backgroundColor: '#f0f5ff', borderRadius: 1, border: '1px solid #bbdefb' }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="body2" sx={{ fontWeight: 500, color: '#555' }}>
                              Vessel Density
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.vessel_density.toFixed(3)}
                            </Typography>
                          </Box>
                          <Typography variant="caption" sx={{ color: '#999' }}>
                            Proportion of vessel pixels in the retinal area
                          </Typography>
                        </Box>

                        {/* Additional Features Grid */}
                        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                          <Box sx={{ p: 1.5, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
                              Peripheral Density
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.features.peripheral_density.toFixed(3)}
                            </Typography>
                          </Box>
                          <Box sx={{ p: 1.5, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
                              Avg Vessel Width
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.features.avg_vessel_width.toFixed(2)}
                            </Typography>
                          </Box>
                          <Box sx={{ p: 1.5, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
                              Fractal Dimension
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.features.fractal_dimension.toFixed(3)}
                            </Typography>
                          </Box>
                          <Box sx={{ p: 1.5, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
                              Branching Density
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.features.branching_density.toFixed(4)}
                            </Typography>
                          </Box>
                          <Box sx={{ p: 1.5, backgroundColor: '#f5f5f5', borderRadius: 1, gridColumn: '1 / -1' }}>
                            <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 0.5, fontWeight: 500 }}>
                              Average Tortuosity
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: '#0066cc' }}>
                              {result.features.avg_tortuosity.toFixed(3)}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Paper>
                  </Box>
                </Grow>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default VesselPage;
