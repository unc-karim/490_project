/**
 * Animated probability/progress display component
 */

import React, { useEffect, useState } from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface AnimatedProgressProps {
  value: number; // 0-100
  label?: string;
  color?: 'error' | 'success' | 'warning' | 'info';
  showPercentage?: boolean;
  animated?: boolean;
}

const AnimatedProgress: React.FC<AnimatedProgressProps> = ({
  value,
  label,
  color = 'info',
  showPercentage = true,
  animated = true,
}) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (!animated) {
      setDisplayValue(value);
      return;
    }

    let current = 0;
    const increment = value / 50;
    const interval = setInterval(() => {
      current += increment;
      if (current >= value) {
        setDisplayValue(value);
        clearInterval(interval);
      } else {
        setDisplayValue(Math.floor(current));
      }
    }, 20);

    return () => clearInterval(interval);
  }, [value, animated]);

  const getColor = (val: number) => {
    if (val >= 80) return '#d32f2f';
    if (val >= 60) return '#f57c00';
    if (val >= 40) return '#fbc02d';
    return '#388e3c';
  };

  return (
    <Box sx={{ width: '100%' }}>
      {label && (
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            mb: 1,
            color: '#555',
          }}
        >
          {label}
        </Typography>
      )}

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box sx={{ flex: 1 }}>
          <LinearProgress
            variant="determinate"
            value={displayValue}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: '#e8e8e8',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
                background: `linear-gradient(90deg, ${getColor(displayValue)}, ${getColor(displayValue)}dd)`,
                transition: 'background 0.3s ease',
              },
            }}
          />
        </Box>

        {showPercentage && (
          <Typography
            variant="body2"
            sx={{
              fontWeight: 700,
              minWidth: 45,
              textAlign: 'right',
              color: getColor(displayValue),
              fontSize: '1rem',
            }}
          >
            {displayValue.toFixed(0)}%
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default AnimatedProgress;
