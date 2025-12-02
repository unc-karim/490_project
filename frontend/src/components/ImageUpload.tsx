/**
 * Reusable image upload component with drag-and-drop
 */

import React, { useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Paper, Typography, Button, Card, CardMedia, Fade } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  label?: string;
  imageUrl?: string;
  onClear?: () => void;
  onBack?: () => void;
  disabled?: boolean;
  maxSize?: number;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  label = 'Upload Image',
  imageUrl,
  onClear,
  onBack,
  disabled = false,
  maxSize = 10 * 1024 * 1024,
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      const file = acceptedFiles[0];

      if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
        alert('Please upload a PNG or JPG image');
        return;
      }

      if (file.size > maxSize) {
        alert(`File size must be less than ${maxSize / (1024 * 1024)}MB`);
        return;
      }

      onImageSelect(file);
    },
    [onImageSelect, maxSize]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    disabled,
    multiple: false,
  });

  useEffect(() => {
    return () => {
      if (imageUrl && imageUrl.startsWith('blob:')) {
        try {
          URL.revokeObjectURL(imageUrl);
        } catch (e) {
          // ignore
        }
      }
    };
  }, [imageUrl]);

  return (
    <Box sx={{ width: '100%' }}>
      {label && (
        <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, color: '#555', mb: 1.5 }}>
          {label}
        </Typography>
      )}

      {!imageUrl ? (
        <Paper
          {...getRootProps()}
          sx={{
            border: '2px dashed #5939E0',
            borderRadius: 2,
            padding: { xs: 4, sm: 6 },
            minHeight: { xs: 200, sm: 240 },
            textAlign: 'center',
            cursor: disabled ? 'not-allowed' : 'pointer',
            background: '#ffffff',
            transition: 'all 0.15s cubic-bezier(0.23, 1, 0.32, 1)',
            opacity: disabled ? 0.5 : 1,
            position: 'relative',
            overflow: 'hidden',
            ...(isDragActive ? {
              borderColor: '#4A2DB0',
              boxShadow: '0 12px 40px rgba(89, 57, 224, 0.25)',
              backgroundColor: 'rgba(89, 57, 224, 0.04)',
            } : {
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.08)',
              '&:hover': disabled ? {} : {
                borderColor: '#4A2DB0',
                boxShadow: '0 8px 32px rgba(89, 57, 224, 0.15)',
                backgroundColor: 'rgba(89, 57, 224, 0.02)',
              },
            }),
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon
            sx={{
              fontSize: { xs: 80, sm: 104 },
              color: '#5939E0',
              mb: 3,
              transition: 'all 0.15s ease',
              transform: isDragActive ? 'scale(1.15) translateY(-8px)' : 'scale(1)',
              filter: 'drop-shadow(0 2px 12px rgba(89, 57, 224, 0.15))',
            }}
          />
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              fontWeight: 800,
              color: '#5939E0',
              fontSize: { xs: '1.4rem', sm: '1.8rem' },
              transition: 'all 0.15s ease',
              letterSpacing: '-0.3px',
            }}
          >
            {isDragActive ? 'Drop here' : 'Drag & drop image'}
          </Typography>
          <Typography variant="body1" sx={{ color: '#666', mb: 2, fontWeight: 600, fontSize: { xs: '1rem', sm: '1.1rem' } }}>
            or click to browse
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: '#999',
              display: 'block',
              fontSize: { xs: '0.9rem', sm: '1rem' },
              fontWeight: 500,
            }}
          >
            PNG, JPG, JPEG • Up to 10MB
          </Typography>
        </Paper>
      ) : (
        <Fade in={true} timeout={400}>
          <Card
            sx={{
              overflow: 'hidden',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.08)',
              transition: 'all 0.15s ease',
              borderRadius: 3,
              '&:hover': {
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)',
                transform: 'translateY(-2px)',
              },
            }}
          >
            <Box sx={{ position: 'relative', backgroundColor: '#F8F5FF' }}>
              <CardMedia
                component="img"
                height="320"
                image={imageUrl}
                alt="Uploaded fundus image"
                sx={{
                  objectFit: 'contain',
                  transition: 'all 0.15s ease',
                  p: 1,
                }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  background: 'linear-gradient(135deg, #4caf50 0%, #66bb6a 100%)',
                  color: 'white',
                  px: 2,
                  py: 0.75,
                  borderRadius: 2,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  boxShadow: '0 4px 12px rgba(76, 175, 80, 0.3)',
                }}
              >
                Image uploaded
              </Box>
            </Box>
            <Box
              sx={{
                p: 3,
                display: 'flex',
                gap: 1.5,
                backgroundColor: 'linear-gradient(135deg, rgba(89,57,224,0.02) 0%, #ffffff 100%)',
                alignItems: 'center',
                borderTop: '1px solid rgba(89, 57, 224, 0.1)',
              }}
            >
              {onClear && (
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                  onClick={() => onClear?.()}
                  sx={{
                    fontWeight: 700,
                    borderColor: '#f44336',
                    color: '#f44336',
                    transition: 'all 0.15s ease',
                    '&:hover': {
                      backgroundColor: '#ffebee',
                      borderColor: '#d32f2f',
                      transform: 'scale(1.02)',
                    },
                  }}
                  aria-label="Remove uploaded image"
                >
                  Clear
                </Button>
              )}

              {onBack && (
                <Button
                  size="small"
                  onClick={() => onBack?.()}
                  sx={{
                    ml: 'auto',
                    fontWeight: 600,
                    color: '#5939E0',
                    transition: 'all 0.2s cubic-bezier(0.23, 1, 0.32, 1)',
                    '&:hover': {
                      backgroundColor: 'rgba(89, 57, 224, 0.08)',
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  ← Back
                </Button>
              )}
            </Box>
          </Card>
        </Fade>
      )}
    </Box>
  );
};

export default ImageUpload;
