"""
Vessel clinical feature extraction - CRITICAL FILE

Extracts 15 clinically validated features from vessel segmentation masks.

Based on medical literature:
- Tortuosity: Associated with HTN (Cheung et al., 2011)
- Caliber: Predicts stroke (Wong et al., 2002)
- Fractal dimension: Predicts CVD mortality (Liew et al., 2011)
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage import measure


def extract_clinical_vessel_features(vessel_mask):
    """
    Extract 15 clinically validated vessel features from segmentation mask.

    Args:
        vessel_mask: Binary mask [H, W] or [0, 1] where 1=vessel, 0=background
                     Can be continuous [0, 1] or binary

    Returns:
        features: Dictionary with feature names and values
        feature_vector: Numpy array [15] for ML models
    """

    # Apply threshold to convert continuous probabilities to binary mask
    # Using 0.3 threshold based on FIVES dataset notebook (label propagation seed selection)
    if vessel_mask.max() > 1.5:  # Already binary [0, 255] or [0, 1000]
        mask = (vessel_mask > 127).astype(np.uint8)
    else:  # Continuous [0, 1]
        # Use 0.3 threshold to preserve more vessel information than hard 0.5
        # This matches the notebook approach for seed selection
        mask = (vessel_mask > 0.3).astype(np.uint8)

    # Apply morphological post-processing to improve mask quality
    # This fills small holes and smooths boundaries
    try:
        # Small closing operation to fill holes in vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Small opening operation to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    except Exception as e:
        # If morphological operations fail, continue with original mask
        pass

    h, w = mask.shape

    features = {}

    # ========================================================================
    # 1. DENSITY FEATURES (3 features)
    # ========================================================================

    # Overall vessel density
    features['vessel_density'] = float(mask.mean())

    # Central vs peripheral density (divide into center and periphery)
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    radius_outer = min(h, w) // 3
    peripheral_mask = dist_from_center > radius_outer

    features['peripheral_density'] = float(mask[peripheral_mask].mean()) if peripheral_mask.sum() > 0 else 0.0

    central_mask = dist_from_center <= radius_outer
    central_density = float(mask[central_mask].mean()) if central_mask.sum() > 0 else 0.0

    # Density gradient (peripheral/central)
    if central_density > 0:
        features['density_gradient'] = float(features['peripheral_density'] / central_density)
    else:
        features['density_gradient'] = 0.0

    # ========================================================================
    # 2. MORPHOLOGICAL FEATURES (3 features)
    # ========================================================================

    # Skeleton for morphological analysis
    try:
        skeleton = skeletonize(mask > 0).astype(np.uint8)
    except Exception:
        skeleton = np.zeros_like(mask)

    skeleton_density = float(skeleton.mean())

    # Average vessel thickness (vessel density / skeleton density)
    if skeleton_density > 1e-8:
        features['avg_vessel_thickness'] = float(features['vessel_density'] / skeleton_density)
    else:
        features['avg_vessel_thickness'] = 0.0

    # Number of vessel segments (connected components)
    labeled = measure.label(mask > 0)
    features['num_vessel_segments'] = float(labeled.max())

    # Spatial uniformity (quadrant analysis)
    h_mid, w_mid = h // 2, w // 2
    quadrants = np.array([
        mask[:h_mid, :w_mid].mean(),
        mask[:h_mid, w_mid:].mean(),
        mask[h_mid:, :w_mid].mean(),
        mask[h_mid:, w_mid:].mean()
    ])
    quadrant_mean = np.mean(quadrants)
    if quadrant_mean > 0:
        features['spatial_uniformity'] = float(1.0 - (np.std(quadrants) / (quadrant_mean + 1e-8)))
    else:
        features['spatial_uniformity'] = 0.0

    # ========================================================================
    # 3. TORTUOSITY FEATURES (2 features)
    # ========================================================================

    labeled_skeleton = measure.label(skeleton)
    num_segments = labeled_skeleton.max()

    tortuosities = []
    if num_segments > 0:
        for i in range(1, min(num_segments + 1, 30)):  # Limit to first 30 segments
            segment = (labeled_skeleton == i)
            segment_length = segment.sum()

            if segment_length < 20:
                continue

            y_coords, x_coords = np.where(segment)
            if len(y_coords) < 3:
                continue

            try:
                coords = np.column_stack([x_coords, y_coords])
                mean_coord = coords.mean(axis=0)
                centered = coords - mean_coord

                # PCA for principal axis
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

                projections = centered @ principal_axis
                chord_length = projections.max() - projections.min()

                if chord_length > 10:
                    tortuosity = (segment_length / chord_length) - 1
                    tortuosities.append(max(0, tortuosity))
            except Exception:
                continue

    features['avg_tortuosity'] = float(np.mean(tortuosities)) if len(tortuosities) > 0 else 0.0
    features['max_tortuosity'] = float(np.max(tortuosities)) if len(tortuosities) > 0 else 0.0

    # ========================================================================
    # 4. CALIBER (WIDTH) FEATURES (3 features)
    # ========================================================================

    distance_map = ndimage.distance_transform_edt(mask)
    widths = distance_map[skeleton > 0] * 2

    if len(widths) > 0:
        features['avg_vessel_width'] = float(np.mean(widths))
        features['vessel_width_std'] = float(np.std(widths))
        features['width_cv'] = float(features['vessel_width_std'] / (features['avg_vessel_width'] + 1e-8))
    else:
        features['avg_vessel_width'] = 0.0
        features['vessel_width_std'] = 0.0
        features['width_cv'] = 0.0

    # ========================================================================
    # 5. COMPLEXITY FEATURES (3 features)
    # ========================================================================

    # Fractal dimension (box-counting)
    def box_count(image, box_size):
        h_img, w_img = image.shape
        count = 0
        for i in range(0, h_img, box_size):
            for j in range(0, w_img, box_size):
                box = image[i:min(i + box_size, h_img), j:min(j + box_size, w_img)]
                if box.sum() > 0:
                    count += 1
        return count

    sizes = [4, 8, 16, 32, 64]
    counts = []
    for size in sizes:
        count = box_count(mask, size)
        if count > 0:
            counts.append(count)
        else:
            break

    if len(counts) >= 3:
        try:
            valid_sizes = sizes[:len(counts)]
            coeffs = np.polyfit(np.log(valid_sizes), np.log(counts), 1)
            features['fractal_dimension'] = float(-coeffs[0])
        except Exception:
            features['fractal_dimension'] = 0.0
    else:
        features['fractal_dimension'] = 0.0

    # Branching density (junctions in skeleton)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.float32), -1, kernel) - skeleton
    branch_points = int(((neighbor_count >= 3) & (skeleton == 1)).sum())

    skeleton_length = skeleton.sum()
    features['branching_density'] = float(branch_points / (skeleton_length + 1e-8))

    # Connectivity index (branch points / endpoints)
    endpoints = int(((neighbor_count == 1) & (skeleton == 1)).sum())
    features['connectivity_index'] = float(branch_points / (endpoints + 1e-8))

    # ========================================================================
    # 6. TEXTURE (1 feature)
    # ========================================================================

    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    mean_local = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    mean_sq_local = cv2.filter2D((mask ** 2).astype(np.float32), -1, kernel)
    variance_local = mean_sq_local - mean_local ** 2
    features['texture_variance'] = float(variance_local.mean())

    # ========================================================================
    # CREATE FEATURE VECTOR (15 features in order)
    # ========================================================================

    feature_vector = np.array([
        features['vessel_density'],           # 1
        features['peripheral_density'],       # 2
        features['density_gradient'],         # 3
        features['avg_vessel_thickness'],     # 4
        features['num_vessel_segments'],      # 5
        features['spatial_uniformity'],       # 6
        features['avg_tortuosity'],           # 7
        features['max_tortuosity'],           # 8
        features['avg_vessel_width'],         # 9
        features['vessel_width_std'],         # 10
        features['width_cv'],                 # 11
        features['fractal_dimension'],        # 12
        features['branching_density'],        # 13
        features['connectivity_index'],       # 14
        features['texture_variance']          # 15
    ], dtype=np.float32)

    return features, feature_vector
