/**
 * Main App component with routing
 */

import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container, Typography } from '@mui/material';

import api from './services/api';
import HomePage from './pages/HomePage';
import HTNPage from './pages/HTNPage';
import CIMTPage from './pages/CIMTPage';
import VesselPage from './pages/VesselPage';
import FusionPage from './pages/FusionPage';

const theme = createTheme({
  palette: {
    primary: {
      main: '#6A4DF5', // Modern purple
      light: '#8B6FFF',
      dark: '#5739D6',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#A680FF', // Light purple
      light: '#B89FFF',
      dark: '#9366FF',
      contrastText: '#ffffff',
    },
    success: {
      main: '#4caf50', // Green for low risk
    },
    warning: {
      main: '#ff9800', // Orange for medium risk
    },
    error: {
      main: '#f44336', // Red for high risk
    },
    background: {
      default: '#F8F5FF',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Inter", "Poppins", "Segoe UI", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3rem',
      fontWeight: 800,
      letterSpacing: '-0.5px',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.3px',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.375rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.9rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 24,
  },
  shadows: [
    'none',
    '0 2px 8px rgba(0, 0, 0, 0.08)',
    '0 4px 12px rgba(0, 0, 0, 0.10)',
    '0 6px 16px rgba(0, 0, 0, 0.12)',
    '0 8px 24px rgba(0, 0, 0, 0.12)',
    '0 10px 28px rgba(0, 0, 0, 0.13)',
    '0 12px 32px rgba(0, 0, 0, 0.15)',
    '0 14px 36px rgba(0, 0, 0, 0.15)',
    '0 16px 40px rgba(0, 0, 0, 0.16)',
    '0 18px 44px rgba(0, 0, 0, 0.16)',
    '0 20px 48px rgba(0, 0, 0, 0.17)',
    '0 22px 52px rgba(0, 0, 0, 0.17)',
    '0 24px 56px rgba(0, 0, 0, 0.18)',
    '0 26px 60px rgba(0, 0, 0, 0.18)',
    '0 28px 64px rgba(0, 0, 0, 0.19)',
    '0 30px 68px rgba(0, 0, 0, 0.19)',
    '0 32px 72px rgba(0, 0, 0, 0.20)',
    '0 36px 80px rgba(0, 0, 0, 0.20)',
    '0 40px 88px rgba(0, 0, 0, 0.21)',
    '0 48px 104px rgba(0, 0, 0, 0.22)',
    '0 56px 120px rgba(0, 0, 0, 0.23)',
    '0 64px 136px rgba(0, 0, 0, 0.24)',
    '0 80px 160px rgba(0, 0, 0, 0.25)',
    '0 96px 184px rgba(0, 0, 0, 0.26)',
    '0 112px 208px rgba(0, 0, 0, 0.27)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 700,
          borderRadius: 12,
          padding: '12px 28px',
          transition: 'all 0.15s ease',
          fontSize: '1rem',
        },
        contained: {
          background: 'linear-gradient(135deg, #6A4DF5 0%, #8B6FFF 100%)',
          boxShadow: '0 4px 12px rgba(106, 77, 245, 0.25)',
          '&:hover': {
            transform: 'scale(1.02)',
            boxShadow: '0 6px 24px rgba(106, 77, 245, 0.35)',
            background: 'linear-gradient(135deg, #5939E0 0%, #7A5FEE 100%)',
          },
        },
        outlined: {
          borderColor: '#6A4DF5',
          color: '#6A4DF5',
          '&:hover': {
            backgroundColor: 'rgba(106, 77, 245, 0.04)',
            borderColor: '#5939E0',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 24,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.10)',
          transition: 'all 0.15s ease',
          '&:hover': {
            transform: 'scale(1.02)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.10)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.10)',
          backgroundImage: 'linear-gradient(135deg, #6A4DF5 0%, #A680FF 100%)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            transition: 'all 0.15s ease',
            '&:hover fieldset': {
              borderColor: '#6A4DF5',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6A4DF5',
              borderWidth: 2,
            },
          },
        },
      },
    },
  },
});

function App() {
  const [apiReady, setApiReady] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAPI = async () => {
      let retries = 0;
      const maxRetries = 5;
      const retryDelay = 2000; // 2 seconds

      while (retries < maxRetries) {
        try {
          const health = await api.healthCheck();
          setApiReady(health.models_loaded);
          setLoading(false);
          return;
        } catch (error) {
          retries++;
          console.error(`API health check failed (attempt ${retries}/${maxRetries}):`, error);
          if (retries < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, retryDelay));
          }
        }
      }

      // All retries failed
      setApiReady(false);
      setLoading(false);
    };

    checkAPI();
  }, []);

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)',
          }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 800, mb: 2, color: '#6A4DF5' }}>
              CVD Risk Predictor
            </Typography>
            <Typography variant="body1" sx={{ color: '#666', mb: 3 }}>
              Loading AI-powered analysis system...
            </Typography>
            <Box sx={{ width: 40, height: 4, background: 'linear-gradient(90deg, #6A4DF5 0%, #A680FF 100%)', borderRadius: 2, mx: 'auto', animation: 'pulse 1.5s ease-in-out infinite' }} />
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)',
          }}
        >
          {/* Main Content */}
          <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
            {!apiReady && (
              <Box
                sx={{
                  mb: 3,
                  p: 2,
                  backgroundColor: '#fff3cd',
                  border: '1px solid #ffc107',
                  borderRadius: 1,
                  color: '#856404',
                }}
              >
                ⚠️ API is not ready. Predictions may not work correctly.
              </Box>
            )}

            <Routes>
              <Route path="/" element={<HomePage apiReady={apiReady} />} />
              <Route path="/htn" element={<HTNPage apiReady={apiReady} />} />
              <Route path="/cimt" element={<CIMTPage apiReady={apiReady} />} />
              <Route path="/vessel" element={<VesselPage apiReady={apiReady} />} />
              <Route path="/fusion" element={<FusionPage apiReady={apiReady} />} />
            </Routes>
          </Container>

          {/* Footer */}
          <Box
            component="footer"
            sx={{
              py: 4,
              px: { xs: 2, sm: 4 },
              mt: 'auto',
              background: 'rgba(255, 255, 255, 0.5)',
              borderTop: '1px solid rgba(106, 77, 245, 0.1)',
              textAlign: 'center',
              color: '#666',
              backdropFilter: 'blur(10px)',
            }}
          >
            <Typography variant="body2" sx={{ mb: 1, fontWeight: 600, color: '#333' }}>
              CVD Risk Prediction System v1.0
            </Typography>
            <Typography variant="caption" sx={{ color: '#888', fontSize: '0.85rem' }}>
              Research Tool - Not Approved for Clinical Diagnosis | Consult Healthcare Professionals
            </Typography>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
