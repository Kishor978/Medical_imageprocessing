import os
import numpy as np
from scipy.ndimage import binary_dilation
from utils import load_ct_data, save_nifti, visualize_expansion_comparison, get_voxel_spacing, mm_to_voxels
from src import bone_segmentation

def create_spherical_structuring_element(radius_voxels):
    """
    Create a spherical structuring element for morphological operations.
    
    Args:
        radius_voxels: Array of radii in voxels [x, y, z]
        
    Returns:
        struct_element: 3D binary structuring element
    """
    # Calculate the size of the structuring element
    # We need to make it large enough to contain the sphere
    size_x = int(2 * np.ceil(radius_voxels[0])) + 1
    size_y = int(2 * np.ceil(radius_voxels[1])) + 1
    size_z = int(2 * np.ceil(radius_voxels[2])) + 1
    
    # Create coordinate grids
    x = np.arange(size_x) - size_x // 2
    y = np.arange(size_y) - size_y // 2
    z = np.arange(size_z) - size_z // 2
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create ellipsoid (sphere if all radii are equal)
    # Normalize by the respective radii
    ellipsoid = (X / radius_voxels[0])**2 + (Y / radius_voxels[1])**2 + (Z / radius_voxels[2])**2
    
    # Create binary structuring element
    struct_element = ellipsoid <= 1.0
    
    return struct_element

def expand_mask_uniform(mask, expansion_mm, voxel_spacing):
    """
    Expand a binary mask uniformly by a specified distance in millimeters.
    
    Args:
        mask: Binary mask to expand
        expansion_mm: Expansion distance in millimeters
        voxel_spacing: Array of voxel dimensions in mm [x, y, z]
        
    Returns:
        expanded_mask: Expanded binary mask
    """
    # Convert expansion distance to voxels
    expansion_voxels = mm_to_voxels(expansion_mm, voxel_spacing)
    
    print(f"Expanding mask by {expansion_mm}mm = {expansion_voxels} voxels")
    
    # Create spherical structuring element
    struct_element = create_spherical_structuring_element(expansion_voxels)
    
    print(f"Structuring element size: {struct_element.shape}")
    
    # Apply binary dilation
    expanded_mask = binary_dilation(mask, structure=struct_element)
    
    return expanded_mask

def contour_expansion(femur_mask, tibia_mask, affine, header, expansion_mm=2.0, output_dir="results/task1_2", visualize=True, original_data=None):
    """
    Main function for contour expansion.
    
    Args:
        femur_mask: Binary mask of the femur
        tibia_mask: Binary mask of the tibia
        affine: Affine transformation matrix
        header: NIfTI header
        expansion_mm: Expansion distance in millimeters
        output_dir: Directory to save the expanded masks
        visualize: Whether to visualize the expansion results
        original_data: Original CT data for visualization
        
    Returns:
        expanded_femur_mask: Expanded femur mask
        expanded_tibia_mask: Expanded tibia mask
    """
    print(f"Starting contour expansion by {expansion_mm}mm...")
    
    # Get voxel spacing from the affine matrix
    voxel_spacing = get_voxel_spacing(affine)
    print(f"Voxel spacing: {voxel_spacing} mm")
    
    # Expand the femur mask
    print("Expanding femur mask...")
    expanded_femur_mask = expand_mask_uniform(femur_mask, expansion_mm, voxel_spacing)
    
    # Expand the tibia mask
    print("Expanding tibia mask...")
    expanded_tibia_mask = expand_mask_uniform(tibia_mask, expansion_mm, voxel_spacing)
    
    # Create combined expanded mask
    combined_expanded_mask = np.zeros_like(expanded_femur_mask, dtype=np.int16)
    combined_expanded_mask[expanded_femur_mask] = 1  # Femur labeled as 1
    combined_expanded_mask[expanded_tibia_mask] = 2   # Tibia labeled as 2
    
    # Handle overlapping regions (give priority to the original assignment)
    overlap = expanded_femur_mask & expanded_tibia_mask
    if np.any(overlap):
        print(f"Found {np.sum(overlap)} overlapping voxels after expansion.")
        # For overlapping regions, check which mask they originally belonged to
        femur_overlap = overlap & femur_mask
        tibia_overlap = overlap & tibia_mask
        
        # Remove overlap from expanded masks
        expanded_femur_mask &= ~overlap
        expanded_tibia_mask &= ~overlap
        
        # Add back the original regions
        expanded_femur_mask |= femur_overlap
        expanded_tibia_mask |= tibia_overlap
        
        # Recreate combined mask
        combined_expanded_mask = np.zeros_like(expanded_femur_mask, dtype=np.int16)
        combined_expanded_mask[expanded_femur_mask] = 1
        combined_expanded_mask[expanded_tibia_mask] = 2
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the expanded masks
    print("Saving expanded segmentation results...")
    save_nifti(expanded_femur_mask.astype(np.int16), affine, header, 
               os.path.join(output_dir, f"femur_expanded_{expansion_mm}mm.nii.gz"))
    save_nifti(expanded_tibia_mask.astype(np.int16), affine, header, 
               os.path.join(output_dir, f"tibia_expanded_{expansion_mm}mm.nii.gz"))
    save_nifti(combined_expanded_mask, affine, header, 
               os.path.join(output_dir, f"combined_expanded_{expansion_mm}mm.nii.gz"))
    
    # Visualize the expansion results if requested
    if visualize and original_data is not None:
        print("Visualizing expansion results...")
        
        # Visualize femur expansion
        visualize_expansion_comparison(original_data, femur_mask, expanded_femur_mask, 
                                     "Femur", num_slices=5, axis=2)  # Axial slices
        visualize_expansion_comparison(original_data, femur_mask, expanded_femur_mask, 
                                     "Femur", num_slices=5, axis=1)  # Coronal slices
        
        # Visualize tibia expansion
        visualize_expansion_comparison(original_data, tibia_mask, expanded_tibia_mask, 
                                     "Tibia", num_slices=5, axis=2)  # Axial slices
        visualize_expansion_comparison(original_data, tibia_mask, expanded_tibia_mask, 
                                     "Tibia", num_slices=5, axis=1)  # Coronal slices
    
    print(f"Contour expansion by {expansion_mm}mm completed successfully.")
    
    return expanded_femur_mask, expanded_tibia_mask

def main():
    """
    Main function to run Task 1.1 (Bone Segmentation) followed by Task 1.2 (Contour Expansion).
    """
    # Parameters
    input_path = "data\\3702_left_knee.nii.gz"
    task1_1_output_dir = "results\\task1_1"
    task1_2_output_dir = "results\\task1_2"
    
    # Task 1.1 parameters
    bone_threshold = 200
    min_component_size = 10000
    morph_iterations = 1
    
    # Task 1.2 parameters
    expansion_mm = 2.0  # Expansion distance in millimeters (parameterized as required)
    
    # Run Task 1.1 - Bone Segmentation
    print("="*50)
    print("RUNNING TASK 1.1 - BONE SEGMENTATION")
    print("="*50)
    
    femur_mask, tibia_mask, combined_mask, affine, header = bone_segmentation(
        input_path, task1_1_output_dir, bone_threshold, min_component_size, 
        morph_iterations, visualize=False  # Skip visualization for now
    )
    
    # Load original data for visualization
    original_data, _, _ = load_ct_data(input_path)
    
    # Run Task 1.2 - Contour Expansion
    print("="*50)
    print("RUNNING TASK 1.2 - CONTOUR EXPANSION")
    print("="*50)
    
    expanded_femur_mask, expanded_tibia_mask = contour_expansion(
        femur_mask, tibia_mask, affine, header, expansion_mm, 
        task1_2_output_dir, visualize=True, original_data=original_data
    )
    
    print("="*50)
    print("TASKS 1.1 AND 1.2 COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Results saved in:")
    print(f"  Task 1.1: {task1_1_output_dir}")
    print(f"  Task 1.2: {task1_2_output_dir}")

if __name__ == "__main__":
    main()