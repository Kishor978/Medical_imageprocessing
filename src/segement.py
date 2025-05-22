import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from matplotlib.animation import FuncAnimation
from utils import load_ct_data, save_nifti,visualize_segmentation,create_segmentation_animation,apply_morphological_operations

def preprocess_volume(data, sigma=1.0):
    """
    Preprocess the CT volume to reduce noise.
    
    Args:
        data: CT scan data as a numpy array
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        preprocessed_data: Preprocessed CT scan data
    """
    # Apply Gaussian filter to reduce noise
    preprocessed_data = ndimage.gaussian_filter(data, sigma=sigma)
    
    return preprocessed_data

def threshold_bones(data, threshold=250):
    """
    Create a binary mask for bones using thresholding.
    
    Args:
        data: CT scan data as a numpy array
        threshold: Intensity threshold for bone segmentation
        
    Returns:
        bone_mask: Binary mask of bones
    """
    # Create a binary mask where voxels with intensity greater than the threshold are considered as bone
    bone_mask = data > threshold
    
    return bone_mask

def fill_3d_holes(binary_mask):
    filled = np.copy(binary_mask)
# Fill holes along each 2D axis slice
    for axis in range(3):
        for i in range(binary_mask.shape[axis]):
            slicer = [slice(None)] * 3
            slicer[axis] = i
            slice_2d = binary_mask[tuple(slicer)]
            filled_slice = binary_fill_holes(slice_2d)
            filled[tuple(slicer)] = filled_slice
    return filled


def separate_femur_tibia(bone_mask, min_size=1000):
    """
    Separate femur and tibia from the binary bone mask.
    
    Args:
        bone_mask: Binary mask of bones
        min_size: Minimum size of connected components to keep
        
    Returns:
        femur_mask: Binary mask of the femur
        tibia_mask: Binary mask of the tibia
    """
    # Label connected components
    struct = ndimage.generate_binary_structure(3, 3)
    labeled_mask, num_features = ndimage.label(bone_mask, structure=struct)
    component_sizes = np.bincount(labeled_mask.ravel())
    sorted_indices = np.argsort(-component_sizes)
    sorted_indices = sorted_indices[sorted_indices != 0]

    print(f"Found {len(sorted_indices)} components above background.")
    large_components = [i for i in sorted_indices if component_sizes[i] > min_size]

    if len(large_components) >= 2:
        # Use z-center of mass to assign femur/tibia
        idx1, idx2 = large_components[:2]
        comp1 = labeled_mask == idx1
        comp2 = labeled_mask == idx2

        z1 = np.mean(np.where(comp1)[2])
        z2 = np.mean(np.where(comp2)[2])

        if z1 < z2:
            femur_mask, tibia_mask = comp1, comp2
        else:
            femur_mask, tibia_mask = comp2, comp1

        print("Split based on two largest components using z-centroids.")
        return femur_mask, tibia_mask

    elif len(large_components) == 1:
        print("Only one component found. Using bounding-box-based z-split.")
        comp = labeled_mask == large_components[0]
        z_coords = np.any(np.any(comp, axis=0), axis=0)
        z_vals = np.where(z_coords)[0]

        if len(z_vals) < 2:
            print("Not enough slices to split.")
            return comp, np.zeros_like(comp)

        z_min, z_max = z_vals[0], z_vals[-1]
        z_split = int(z_min + 0.51 * (z_max - z_min))  # 45% height

        femur_mask = np.copy(comp)
        femur_mask[:, :, z_split:] = 0

        tibia_mask = np.copy(comp)
        tibia_mask[:, :, :z_split] = 0

        femur_mask = fill_3d_holes(femur_mask)
        tibia_mask = fill_3d_holes(tibia_mask)
        return femur_mask, tibia_mask

    else:
        print("No valid components found.")
        return np.zeros_like(bone_mask), np.zeros_like(bone_mask)

def bone_segmentation(input_path, output_dir, bone_threshold=300, min_component_size=10000, morph_iterations=2, visualize=True):
    """
    Main function for bone segmentation.
    
    Args:
        input_path: Path to the input CT scan file
        output_dir: Directory to save the segmentation results
        bone_threshold: Intensity threshold for bone segmentation
        min_component_size: Minimum size of connected components to keep
        morph_iterations: Number of iterations for morphological operations
        visualize: Whether to visualize the segmentation results
    """
    print("Starting bone segmentation...")
    
    # Load the CT scan data
    print(f"Loading CT scan data from: {input_path}")
    data, affine, header = load_ct_data(input_path)
    
    # Print information about the data
    print(f"CT scan data loaded successfully.")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data)}, {np.max(data)}]")
    
    # Preprocess the volume
    print("Preprocessing the volume...")
    preprocessed_data = preprocess_volume(data, sigma=1.0)
    
    # Threshold the bones
    print(f"Thresholding bones with threshold: {bone_threshold}...")
    bone_mask = threshold_bones(preprocessed_data, threshold=bone_threshold)
    
    # Apply initial morphological operations to clean up the mask
    print("Applying initial cleaning operations...")
    bone_mask = ndimage.binary_closing(bone_mask, iterations=2)  # Fill small holes first
    
    # Separate femur and tibia
    print("Separating femur and tibia...")
    femur_mask, tibia_mask = separate_femur_tibia(bone_mask, min_size=min_component_size)
    
    # Apply morphological operations to clean up the individual masks
    print("Applying morphological operations to clean up the masks...")
    femur_mask = apply_morphological_operations(femur_mask, iterations=morph_iterations)
    tibia_mask = apply_morphological_operations(tibia_mask, iterations=morph_iterations)
    
    # Make sure femur and tibia don't overlap (in case of any overlapping voxels)
    overlap = femur_mask & tibia_mask
    if np.any(overlap):
        print(f"Warning: Found {np.sum(overlap)} overlapping voxels between femur and tibia.")
        # Remove overlapping voxels from both masks
        femur_mask &= ~overlap
        tibia_mask &= ~overlap
    
    # Combine the masks for visualization and output
    combined_mask = np.zeros_like(bone_mask, dtype=np.int16)
    combined_mask[femur_mask] = 1  # Femur labeled as 1
    combined_mask[tibia_mask] = 2  # Tibia labeled as 2
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the segmentation results
    print("Saving segmentation results...")
    save_nifti(femur_mask.astype(np.int16), affine, header, os.path.join(output_dir, "femur_segmentation.nii.gz"))
    save_nifti(tibia_mask.astype(np.int16), affine, header, os.path.join(output_dir, "tibia_segmentation.nii.gz"))
    save_nifti(combined_mask, affine, header, os.path.join(output_dir, "combined_segmentation.nii.gz"))
    
    # Visualize the segmentation results if requested
    if visualize:
        print("Visualizing segmentation results...")
        visualize_segmentation(data, femur_mask, tibia_mask, num_slices=5, axis=2)  # Axial slices
        visualize_segmentation(data, femur_mask, tibia_mask, num_slices=5, axis=1)  # Coronal slices
        visualize_segmentation(data, femur_mask, tibia_mask, num_slices=5, axis=0)  # Sagittal slices
        create_segmentation_animation(data, femur_mask, tibia_mask, axis=1, interval=100, save_path=os.path.join(output_dir, "animation_axial.gif"))

    print("Bone segmentation completed successfully.")
    
    return femur_mask, tibia_mask, combined_mask, affine, header

if __name__ == "__main__":
    # Path to the CT scan file and output directory
    input_path ="data\\3702_left_knee.nii.gz"
    output_dir = "results\\task1_1"
    
    # Parameters for bone segmentation
    # Based on the histogram, a threshold of 200-300 HU should work well for bone
    bone_threshold = 200  # HU threshold for bone segmentation
    min_component_size = 10000  # Minimum size of connected components to keep
    morph_iterations = 1  # Number of iterations for morphological operations
    
    # Run the bone segmentation
    femur_mask, tibia_mask, combined_mask, affine, header = bone_segmentation(
        input_path, output_dir, bone_threshold, min_component_size, morph_iterations, visualize=True
    )