/**
 * CIMT regression page
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CircularProgress,
  
  Grid,
  Paper,
  TextField,
  Typography,
  RadioGroup,
  FormControlLabel,
  Radio,
  Grow,
} from '@mui/material';
import ImageUpload from '../components/ImageUpload';
import api, { CIMTResult } from '../services/api';

interface CIMTPageProps {
  apiReady: boolean;
}

const CIMTPage: React.FC<CIMTPageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  const [age, setAge] = useState(65);
  const [gender, setGender] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CIMTResult | null>(null);
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

    setLoading(true);
    setError('');

    try {
      const response = await api.predictCIMT(leftImage, rightImage, age, gender);
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

  return (
    <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, width: '100vw', height: '100vh', minHeight: '100vh', minWidth: '100vw', background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)', overflow: 'auto' }}>
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
        <Box sx={{ maxWidth: 1200, mx: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: { xs: 2, sm: 4 } }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h2" sx={{ fontWeight: 800, mb: 1, fontSize: { xs: '1.8rem', sm: '2rem', md: '2.4rem' }, letterSpacing: '-0.3px' }}>
              CIMT Prediction
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.9, fontWeight: 400, fontSize: { xs: '0.95rem', sm: '1rem' }, lineHeight: 1.5 }}>
              Carotid intima-media thickness estimation from retinal and clinical data
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
      </Box>
      <Box sx={{ pb: 6, width: '100vw' }}>
        <Grid container spacing={3} sx={{ width: '100vw', maxWidth: '1600px', mx: 'auto' }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, backgroundColor: '#f8faff', maxWidth: 700, mx: 'auto' }}>
              <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: '#0066cc' }}>
                Patient Information & Images
              </Typography>

              {/* Clinical Data - FIRST */}
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
                  variant="outlined"
                  size="small"
                  sx={{ mb: 2 }}
                />
                <Box sx={{ mb: 0 }}>
                  <Typography variant="body2" sx={{ mb: 1, fontWeight: 500, color: '#555' }}>
                    Sex
                  </Typography>
                  <RadioGroup value={gender.toString()} onChange={(e) => setGender(parseInt(e.target.value))} row>
                    <FormControlLabel value="0" control={<Radio size="small" />} label="Female" />
                    <FormControlLabel value="1" control={<Radio size="small" />} label="Male" />
                  </RadioGroup>
                </Box>
              </Box>

              {/* Retinal Images - now full width */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: '#555' }}>
                  Fundus Images
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Box sx={{ width: '100%' }}>
                      <ImageUpload
                        onImageSelect={setLeftImage}
                        label="Left Eye Image (Required)"
                        imageUrl={leftImage ? URL.createObjectURL(leftImage) : ''}
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
                        onImageSelect={setRightImage}
                        label="Right Eye Image (Optional)"
                        imageUrl={rightImage ? URL.createObjectURL(rightImage) : ''}
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
                    Calculating...
                  </Box>
                ) : (
                  '📏 Estimate CIMT'
                )}
              </Button>

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
                    {/* Main Result Card */}
                    <Card
                      sx={{
                        p: 4,
                        background:
                          result.risk_category === 'Elevated'
                            ? 'linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)'
                            : result.risk_category === 'Borderline'
                            ? 'linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)'
                            : 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)',
                        border: `3px solid ${
                          result.risk_category === 'Elevated'
                            ? '#d32f2f'
                            : result.risk_category === 'Borderline'
                            ? '#f57c00'
                            : '#388e3c'
                        }`,
                        borderRadius: 2,
                        boxShadow:
                          result.risk_category === 'Elevated'
                            ? '0 8px 24px rgba(211, 47, 47, 0.2)'
                            : result.risk_category === 'Borderline'
                            ? '0 8px 24px rgba(245, 127, 0, 0.2)'
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
                          background: result.risk_category === 'Elevated'
                            ? 'linear-gradient(135deg, #d32f2f, #f44336)'
                            : result.risk_category === 'Borderline'
                            ? 'linear-gradient(135deg, #f57c00, #ffe0b2)'
                            : 'linear-gradient(135deg, #388e3c, #66bb6a)',
                          opacity: 0.1,
                          zIndex: -1,
                        },
                      }}
                    >
                      <Typography variant="overline" sx={{ mb: 1, fontWeight: 700, color: '#666', letterSpacing: 1 }}>
                        📊 CIMT Measurement
                      </Typography>
                      <Typography
                        variant="h2"
                        sx={{
                          fontWeight: 900,
                          mb: 1,
                          fontSize: '2.5rem',
                          color:
                            result.risk_category === 'Elevated'
                              ? '#d32f2f'
                              : result.risk_category === 'Borderline'
                              ? '#f57c00'
                              : '#388e3c',
                          animation: 'pulse 2s ease-in-out infinite',
                          '@keyframes pulse': {
                            '0%, 100%': { opacity: 1 },
                            '50%': { opacity: 0.8 },
                          },
                        }}
                      >
                        {result.value_mm.toFixed(3)} mm
                      </Typography>

                      <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: 1.5, border: '1px solid rgba(0,0,0,0.05)' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
                          Risk Category
                        </Typography>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: 700,
                            fontSize: '1.1rem',
                            color:
                              result.risk_category === 'Elevated'
                                ? '#d32f2f'
                                : result.risk_category === 'Borderline'
                                ? '#f57c00'
                                : '#388e3c',
                          }}
                        >
                          {result.risk_category}
                        </Typography>
                      </Box>

                      <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: 1.5, mt: 2, border: '1px solid rgba(0,0,0,0.05)' }}>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: '#666', mb: 0.8 }}>
                          Threshold
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 700, fontSize: '1.1rem', color: '#0066cc' }}>
                          {result.threshold_mm.toFixed(2)} mm
                        </Typography>
                      </Box>
                    </Card>

                    {/* Clinical Interpretation */}
                    <Paper sx={{ p: 3, backgroundColor: '#fafafa', border: '1px solid #e8e8e8', borderRadius: 1.5 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600, mb: 1.5, color: '#0066cc' }}>
                        Clinical Significance
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#555', lineHeight: 1.8, fontSize: '0.95rem' }}>
                        {result.clinical_significance}
                      </Typography>
                    </Paper>
                  </Box>
                </Grow>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default CIMTPage;
