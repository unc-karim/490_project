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
import ScienceIcon from '@mui/icons-material/Science';
import ImageUpload from '../components/ImageUpload';
import MetricCard from '../components/MetricCard';
import ExportButtons from '../components/ExportButtons';
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
      console.log('Vessel API Response:', response);
      if (response.status === 'success' && response.result) {
        console.log('Setting result:', response.result);
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
    <Box sx={{ background: '#ffffff', minHeight: '100vh' }}>
      {/* Header */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #5939E0 0%, #9D7FFF 100%)',
          color: 'white',
          py: { xs: 3, sm: 4, md: 5 },
          px: { xs: 2, sm: 3, md: 4 },
          mb: { xs: 4, md: 6 },
          borderBottomLeftRadius: { xs: 16, sm: 24, md: 32 },
          borderBottomRightRadius: { xs: 16, sm: 24, md: 32 },
          boxShadow: '0 8px 32px rgba(106, 77, 245, 0.25)',
        }}
      >
        <Container maxWidth="lg">
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { xs: 'flex-start', sm: 'center' }, justifyContent: 'space-between', gap: { xs: 3, sm: 2, md: 4 } }}>
            <Box sx={{ flex: 1 }}>
              <Typography
                component="h1"
                variant="h2"
                sx={{
                  mb: 1,
                }}
              >
                Vessel Segmentation
              </Typography>
              <Typography variant="body1" sx={{ opacity: 0.85 }}>
                Automatic analysis of retinal blood vessel features
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1.5, flexShrink: 0, flexWrap: 'wrap', justifyContent: { xs: 'flex-start', sm: 'flex-end' } }}>
              <Button
                onClick={() => navigate('/')}
                disabled={loading}
                variant="text"
                sx={{
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  py: 1,
                  px: 2,
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  },
                }}
              >
                ← Home
              </Button>
              <Button
                onClick={handleClearAll}
                disabled={loading}
                variant="outlined"
                sx={{
                  color: 'white',
                  borderColor: 'white',
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  py: 1,
                  px: 3,
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    borderColor: 'white',
                  },
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
                sx={{
                  mt: 3,
                  fontWeight: 600,
                  fontSize: '1rem',
                  backgroundColor: '#5939E0',
                  boxShadow: '0 2px 8px rgba(89, 57, 224, 0.2)',
                  '&:hover': {
                    backgroundColor: '#4A2DB0',
                    boxShadow: '0 4px 12px rgba(89, 57, 224, 0.25)',
                  },
                }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <CircularProgress size={20} color="inherit" />
                    Analyzing Vessels...
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <ScienceIcon sx={{ fontSize: '1.2rem' }} />
                    Segment Vessels
                  </Box>
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
                        <MetricCard
                          variant="secondary"
                          label="Vessel Density"
                          value={result.vessel_density.toFixed(3)}
                          description="Proportion of vessel pixels in the retinal area"
                          fullWidth
                        />
                        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                          <MetricCard
                            variant="compact"
                            label="Peripheral Density"
                            value={result.features.peripheral_density.toFixed(3)}
                          />
                          <MetricCard
                            variant="compact"
                            label="Avg Vessel Width"
                            value={result.features.avg_vessel_width.toFixed(2)}
                          />
                          <MetricCard
                            variant="compact"
                            label="Fractal Dimension"
                            value={result.features.fractal_dimension.toFixed(3)}
                          />
                          <MetricCard
                            variant="compact"
                            label="Branching Density"
                            value={result.features.branching_density.toFixed(4)}
                          />
                          <Box sx={{ gridColumn: '1 / -1' }}>
                            <MetricCard
                              variant="compact"
                              label="Average Tortuosity"
                              value={result.features.avg_tortuosity.toFixed(3)}
                              fullWidth
                            />
                          </Box>
                        </Box>
                      </Box>
                      <ExportButtons
                        data={{
                          vessel_density: result.vessel_density,
                          features: result.features,
                          timestamp: new Date().toISOString(),
                        }}
                        filename="vessel_segmentation"
                      />
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
