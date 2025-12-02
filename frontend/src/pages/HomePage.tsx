/**
 * Home page - Landing page with model selection
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Grow,
  Tooltip,
  IconButton,
  Switch,
} from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';

interface HomePageProps {
  apiReady: boolean;
}

const HomePage: React.FC<HomePageProps> = ({ apiReady }) => {
  const navigate = useNavigate();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  const models = [
    {
      title: 'Hypertension Detection',
      description: 'Binary classification of hypertensive retinopathy',
      subtitle: 'RETFound ViT Model',
      icon: '♥️',
      path: '/htn',
      color: '#ff6f61', // soft coral
      bgColor: '#ffeaea',
    },
    {
      title: 'CIMT Regression',
      description: 'Predict carotid intima-media thickness (0.4 - 1.2)',
      subtitle: 'Siamese Multimodal',
      icon: '🩺',
      path: '/cimt',
      color: '#64b5f6', // soft blue
      bgColor: '#e3f2fd',
    },
    {
      title: 'A/V Segmentation',
      description: 'Segment retinal blood vessels and extract features',
      subtitle: 'U-Net Architecture',
      icon: '🩸',
      path: '/vessel',
      color: '#81c784', // soft green
      bgColor: '#e8f5e9',
    },
    {
      title: 'Fusion Model',
      description: 'Complete CVD risk assessment from all models',
      subtitle: 'Meta-Classifier (MLP)',
      icon: '🧬',
      path: '/fusion',
      color: '#b39ddb', // soft purple
      bgColor: '#f3e5f5',
    },
  ];


  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: '100vw',
        height: '100vh',
        minHeight: '100vh',
        minWidth: '100vw',
        background: darkMode
          ? 'linear-gradient(135deg, #2A1B4D 0%, #1F1535 100%)'
          : 'linear-gradient(135deg, #F8F5FF 0%, #EDE7FF 100%)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden',
        transition: 'background 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      {/* Theme Toggle */}
      <Box sx={{ position: 'absolute', top: 24, right: 32, zIndex: 10, display: 'flex', alignItems: 'center', gap: 1 }}>
        <IconButton
          onClick={() => setDarkMode((d) => !d)}
          size="large"
          sx={{
            color: darkMode ? '#A680FF' : '#6A4DF5',
            transition: 'all 0.3s ease',
            '&:hover': {
              backgroundColor: darkMode ? 'rgba(166, 128, 255, 0.1)' : 'rgba(106, 77, 245, 0.1)',
              transform: 'scale(1.05)',
            },
          }}
        >
          {darkMode ? <LightModeIcon sx={{ fontSize: '1.5rem' }} /> : <DarkModeIcon sx={{ fontSize: '1.5rem' }} />}
        </IconButton>
        <Switch
          checked={darkMode}
          onChange={() => setDarkMode((d) => !d)}
          sx={{
            '& .MuiSwitch-switchBase.Mui-checked': {
              color: '#A680FF',
              '&:hover': { backgroundColor: 'rgba(166, 128, 255, 0.1)' },
            },
            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
              backgroundColor: '#A680FF',
            },
          }}
        />
      </Box>
      {/* Main Content */}
      <Box sx={{ flex: 1, width: '100vw', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', pt: { xs: 8, md: 16 }, pb: { xs: 2, md: 4 } }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 8, px: { xs: 2, md: 0 }, width: '100vw' }}>
          <Typography
            variant="h1"
            sx={{
              fontWeight: 800,
              mb: 2,
              background: darkMode
                ? 'linear-gradient(135deg, #E8DAFF 0%, #A680FF 100%)'
                : 'linear-gradient(135deg, #6A4DF5 0%, #8B6FFF 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontSize: { xs: '2.2rem', sm: '3rem', md: '3.5rem' },
              letterSpacing: '-0.5px',
            }}
          >
            Cardiovascular Risk Assessment
          </Typography>
          <Typography
            variant="body1"
            sx={{
              color: darkMode ? '#B8A0D0' : '#7D6B9F',
              fontWeight: 500,
              mb: 3,
              fontSize: { xs: '1rem', sm: '1.125rem' },
              letterSpacing: '0.2px',
              maxWidth: 600,
              mx: 'auto',
            }}
          >
            Advanced AI-Powered Analysis from Retinal Imaging
          </Typography>
          <Box
            sx={{
              width: 80,
              height: 4,
              background: darkMode
                ? 'linear-gradient(90deg, #A680FF 0%, #8B6FFF 100%)'
                : 'linear-gradient(90deg, #6A4DF5 0%, #8B6FFF 100%)',
              mx: 'auto',
              borderRadius: 2,
              boxShadow: darkMode
                ? '0 4px 16px rgba(166, 128, 255, 0.3)'
                : '0 4px 16px rgba(106, 77, 245, 0.3)',
            }}
          />
        </Box>
        {/* Model Selection Grid */}
        <Grid container spacing={4} sx={{ mb: 2, px: { xs: 2, md: 4 }, width: '100vw', maxWidth: '100vw', justifyContent: 'center' }}>
          {models.map((model, index) => (
            <Grow in={true} timeout={300 + index * 100} key={index}>
              <Grid item xs={12} sm={6} md={3} lg={3} xl={3} sx={{ display: 'flex', justifyContent: 'center' }}>
                <Tooltip title={apiReady ? 'Click to analyze' : 'Loading...'} arrow>
                  <Card
                    onMouseEnter={() => setHoveredIndex(index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    onClick={() => apiReady && navigate(model.path)}
                    sx={{
                      cursor: apiReady ? 'pointer' : 'not-allowed',
                      height: { xs: 240, sm: 280, md: 340 },
                      width: { xs: '90%', sm: 280, md: 340 },
                      background: darkMode
                        ? 'linear-gradient(135deg, rgba(106,77,245,0.15) 0%, rgba(166,128,255,0.1) 100%)'
                        : model.bgColor,
                      border: `2px solid ${model.color}`,
                      borderRadius: 28,
                      boxShadow: hoveredIndex === index
                        ? `0 12px 40px rgba(0,0,0,0.15)`
                        : '0 4px 20px rgba(0,0,0,0.10)',
                      opacity: apiReady ? 1 : 0.5,
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.15s cubic-bezier(0.23, 1, 0.32, 1)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backdropFilter: 'blur(8px)',
                      '&:hover': apiReady ? {
                        transform: 'translateY(-8px) scale(1.02)',
                        boxShadow: `0 16px 48px rgba(0,0,0,0.18)`,
                      } : {},
                    }}
                  >
                    <CardContent sx={{ textAlign: 'center', p: { xs: 3, md: 4 }, position: 'relative', zIndex: 1, width: '100%', display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
                      <Box
                        sx={{
                          fontSize: { xs: '2.4rem', sm: '2.8rem', md: '3.2rem' },
                          mb: 2,
                          color: model.color,
                          transition: 'all 0.15s ease',
                          transform: hoveredIndex === index ? 'scale(1.1) translateY(-4px)' : 'scale(1)',
                        }}
                      >
                        {model.icon}
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Typography
                          variant="h5"
                          sx={{
                            fontWeight: 800,
                            mb: 1,
                            color: darkMode ? '#E8DAFF' : model.color,
                            fontSize: { xs: '1.15rem', sm: '1.3rem', md: '1.4rem' },
                            letterSpacing: '-0.3px',
                          }}
                        >
                          {model.title}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: darkMode ? 'rgba(230, 200, 255, 0.7)' : model.color,
                            display: 'block',
                            mb: 2,
                            fontWeight: 600,
                            fontSize: { xs: '0.8rem', sm: '0.85rem' },
                            letterSpacing: '0.3px',
                            textTransform: 'uppercase',
                            opacity: 0.9,
                          }}
                        >
                          {model.subtitle}
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            color: darkMode ? '#C5B0E0' : '#666',
                            lineHeight: 1.6,
                            minHeight: 44,
                            fontSize: { xs: '0.9rem', sm: '0.95rem', md: '1rem' },
                            fontWeight: 500,
                          }}
                        >
                          {model.description}
                        </Typography>
                      </Box>
                      <Typography
                        sx={{
                          fontSize: { xs: '0.85rem', sm: '0.9rem', md: '0.95rem' },
                          color: model.color,
                          fontWeight: 700,
                          letterSpacing: '0.2px',
                          mt: 2,
                          textTransform: 'uppercase',
                        }}
                      >
                        {apiReady ? '✨ Click to Analyze' : '⟳ Loading...'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Tooltip>
              </Grid>
            </Grow>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default HomePage;
