/**
 * Fusion page - Full CVD risk assessment
 * MAIN PAGE - Combines all three models
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
  LinearProgress,
  Paper,
  TextField,
  Typography,
  Alert,
  
  Radio,
  RadioGroup,
  FormControlLabel,
  Grow,
} from '@mui/material';
import ImageUpload from '../components/ImageUpload';
import api, { FusionPrediction } from '../services/api';

interface FusionPageProps {
  apiReady: boolean;
}

const FusionPage: React.FC<FusionPageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  const [age, setAge] = useState(65);
  const [gender, setGender] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FusionPrediction | null>(null);
  const [error, setError] = useState('');

  const handleClearLeftImage = () => {
    setLeftImage(null);
  };

  const handleClearRightImage = () => {
    setRightImage(null);
  };

  const handleClearAll = () => {
    if (leftImage) {
      try { const u = URL.createObjectURL(leftImage); if (u.startsWith('blob:')) URL.revokeObjectURL(u); } catch (e) {}
    }
    if (rightImage) {
      try { const u = URL.createObjectURL(rightImage); if (u.startsWith('blob:')) URL.revokeObjectURL(u); } catch (e) {}
    }
    setLeftImage(null);
    setRightImage(null);
    setResult(null);
    setError('');
  };

  const handlePredict = async () => {
    if (!leftImage) {
      setError('Please select at least left eye image');
      return;
    }

    if (age < 1 || age > 150) {
      setError('Please enter valid age (1-150)');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.predictFusion(leftImage, rightImage, age, gender);
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

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'High':
        return '#f44336';
      case 'Medium':
        return '#ff9800';
      case 'Low':
        return '#4caf50';
      default:
        return '#1976d2';
    }
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
                Full CVD Risk Assessment
              </Typography>
              <Typography variant="body1" sx={{ opacity: 0.9, fontWeight: 400, fontSize: { xs: '0.95rem', sm: '1rem' }, lineHeight: 1.5 }}>
                Comprehensive cardiovascular risk analysis combining all models
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
                Patient Information & Images
              </Typography>

              {/* Clinical Data Section - FIRST */}
              <Box sx={{ p: 2.5, backgroundColor: '#ffffff', borderRadius: 1, border: '1px solid #e0e0e0', mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2.5, fontWeight: 600, color: '#555' }}>
                  Clinical Data
                </Typography>

                <TextField
                  type="number"
                  label="Age (years)"
                  fullWidth
                  value={age}
                  onChange={(e) => setAge(parseInt(e.target.value))}
                  inputProps={{ min: 1, max: 150 }}
                  disabled={loading}
                  sx={{ mb: 2 }}
                  variant="outlined"
                  size="small"
                />

                <Box sx={{ mb: 0 }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500, color: '#555' }}>
                    Sex
                  </Typography>
                  <RadioGroup value={gender.toString()} onChange={(e) => setGender(parseInt(e.target.value))} row>
                    <FormControlLabel value="0" control={<Radio size="small" disabled={loading} />} label="Female" />
                    <FormControlLabel value="1" control={<Radio size="small" disabled={loading} />} label="Male" />
                  </RadioGroup>
                </Box>
              </Box>

              {/* Retinal Images Section - now full width */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: '#555' }}>
                  Fundus Images
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={(file) => setLeftImage(file)}
                        label="Left Eye Image (Required)"
                        imageUrl={leftImage ? URL.createObjectURL(leftImage) : ''}
                        disabled={loading}
                      />
                      {leftImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          onClick={handleClearLeftImage}
                          sx={{ mt: 1, width: '100%' }}
                          disabled={loading}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={(file) => setRightImage(file)}
                        label="Right Eye Image (Optional)"
                        imageUrl={rightImage ? URL.createObjectURL(rightImage) : ''}
                        disabled={loading}
                      />
                      {rightImage && (
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          onClick={handleClearRightImage}
                          sx={{ mt: 1, width: '100%' }}
                          disabled={loading}
                        >
                          Clear Image
                        </Button>
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              {/* Action Button */}
              <Button
                variant="contained"
                fullWidth
                onClick={handlePredict}
                disabled={!leftImage || loading || !apiReady}
                size="large"
                sx={{ mt: 3, fontWeight: 700, fontSize: '1.1rem', py: 2 }}
              >
                {loading ? (
                  <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                    <CircularProgress size={20} color="inherit" />
                    Generating Analysis...
                  </Box>
                ) : (
                  '📋 Generate CVD Risk Assessment'
                )}
              </Button>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              {/* Results Section - now below input */}
              {result && (
                <Grow in={true} timeout={500}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 4 }}>
                    {/* Main Risk Indicator */}
                    <Card
                      sx={{
                        backgroundColor: getRiskColor(result.risk_level),
                        color: 'white',
                        p: 3,
                        borderRadius: 2,
                        boxShadow: '0 8px 20px rgba(0,0,0,0.2)',
                      }}
                    >
                      <Typography variant="overline" sx={{ mb: 1, opacity: 0.9, fontWeight: 600, letterSpacing: 1 }}>
                        CVD Risk Level
                      </Typography>
                      <Typography variant="h2" sx={{ fontWeight: 800, mb: 2.5, fontSize: '3.2rem', lineHeight: 1.1 }}>
                        {result.risk_level}
                      </Typography>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ mb: 1, fontWeight: 600, fontSize: '0.95rem' }}>
                          Risk Probability
                        </Typography>
                        <Typography variant="h4" sx={{ fontWeight: 800, mb: 1.5, fontSize: '1.8rem' }}>
                          {(result.cvd_probability * 100).toFixed(1)}%
                        </Typography>

                        <LinearProgress
                          variant="determinate"
                          value={result.cvd_probability * 100}
                          sx={{
                            height: 10,
                            borderRadius: 2,
                            backgroundColor: 'rgba(255, 255, 255, 0.3)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: 'white',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                            },
                          }}
                        />
                      </Box>
                    </Card>

                    {/* Recommendation Alert */}
                    <Paper sx={{ p: 3, backgroundColor: '#f5f5f5', border: '1px solid #e0e0e0', borderRadius: 2 }}>
                      <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: '#9c27b0' }}>
                        Recommendation
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#555', lineHeight: 1.7 }}>
                        {result.recommendation}
                      </Typography>
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

export default FusionPage;
