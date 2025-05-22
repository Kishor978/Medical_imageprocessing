import numpy as np
from scipy.ndimage import distance_transform_edt

def get_voxel_spacing(affine):
    """
    Extract voxel spacing from the affine transformation matrix.
    
    Args:
        affine: 4x4 affine transformation matrix
        
    Returns:
        voxel_spacing: Array of voxel dimensions in mm [x, y, z]
    """
    # Extract the scaling factors from the affine matrix
    # The voxel spacing is the magnitude of the first 3 columns of the upper 3x3 matrix
    voxel_spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    return voxel_spacing

def mm_to_voxels(distance_mm, voxel_spacing):
    """
    Convert distance in millimeters to voxels for each dimension.
    
    Args:
        distance_mm: Distance in millimeters
        voxel_spacing: Array of voxel dimensions in mm [x, y, z]
        
    Returns:
        voxels: Array of distances in voxels [x, y, z]
    """
    voxels = distance_mm / voxel_spacing
    return voxels

def create_distance_field(mask, voxel_spacing):
    """
    Create a distance field from the mask boundary.
    
    Args:
        mask: Binary mask
        voxel_spacing: Array of voxel dimensions in mm [x, y, z]
        
    Returns:
        distance_field: Distance transform in millimeters
    """
    # Create distance transform (in voxels)
    distance_voxels = distance_transform_edt(mask)
    
    # Convert to millimeters using the minimum voxel spacing
    # This gives us an approximation of the real-world distance
    min_voxel_size = np.min(voxel_spacing)
    distance_mm = distance_voxels * min_voxel_size
    
    return distance_mm
