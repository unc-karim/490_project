/**
 * Custom animated loader component with better visual feedback
 */

import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface AnimatedLoaderProps {
  message?: string;
  size?: number;
}

const AnimatedLoader: React.FC<AnimatedLoaderProps> = ({
  message = 'Analyzing...',
  size = 50,
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
        py: 3,
        animation: 'fadeIn 0.4s ease-in-out',
        '@keyframes fadeIn': {
          '0%': { opacity: 0, transform: 'scale(0.95)' },
          '100%': { opacity: 1, transform: 'scale(1)' },
        },
      }}
    >
      <Box sx={{ position: 'relative', width: size, height: size }}>
        {/* Outer rotating ring */}
        <CircularProgress
          variant="indeterminate"
          size={size}
          thickness={4}
          sx={{
            position: 'absolute',
            animation: 'spin 3s linear infinite',
            '@keyframes spin': {
              '0%': { transform: 'rotate(0deg)' },
              '100%': { transform: 'rotate(360deg)' },
            },
            color: '#0066cc',
          }}
        />

        {/* Inner pulsing dot */}
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 12,
            height: 12,
            borderRadius: '50%',
            backgroundColor: '#0066cc',
            animation: 'pulse 1.5s ease-in-out infinite',
            '@keyframes pulse': {
              '0%, 100%': { opacity: 0.3, transform: 'scale(1)' },
              '50%': { opacity: 1, transform: 'scale(1.3)' },
            },
          }}
        />
      </Box>

      <Typography
        variant="body2"
        sx={{
          color: '#666',
          fontWeight: 600,
          animation: 'fadeInOut 1.5s ease-in-out infinite',
          '@keyframes fadeInOut': {
            '0%, 100%': { opacity: 0.6 },
            '50%': { opacity: 1 },
          },
        }}
      >
        {message}
      </Typography>
    </Box>
  );
};

export default AnimatedLoader;
