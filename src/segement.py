import os
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import load_ct_data, save_nifti

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
    # Label connected components in the bone mask
    # Use a structure that connects all neighboring voxels (26-connectivity in 3D)
    struct = ndimage.generate_binary_structure(3, 3)  # 3D, full connectivity
    labeled_mask, num_features = ndimage.label(bone_mask, structure=struct)
    
    # Calculate the size of each component
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Sort components by size (excluding background with index 0)
    sorted_indices = np.argsort(-component_sizes)
    # Background should be the first component (index 0)
    sorted_indices = sorted_indices[sorted_indices != 0]
    
    # Print info about the largest components for debugging
    print(f"Total number of components: {num_features}")
    for i, idx in enumerate(sorted_indices[:5]):  # Show top 5 components
        if idx > 0:  # Skip background (index 0)
            print(f"Component {i+1} (Label {idx}): Size = {component_sizes[idx]}")
    
    # Keep only components larger than the minimum size
    large_components = [i for i in sorted_indices if i > 0 and component_sizes[i] > min_size]
    
    if len(large_components) < 2:
        print(f"Warning: Found fewer than 2 large bone components. Adjust parameters.")
        if len(large_components) == 1:
            # If only one large component, try to split it based on location
            single_component = labeled_mask == large_components[0]
            # Find the midpoint along the z-axis
            z_midpoint = bone_mask.shape[2] // 2
            # Split into upper and lower parts
            tibia_mask = np.copy(single_component)
            tibia_mask[:, :, z_midpoint:] = False
            femur_mask = np.copy(single_component)
            femur_mask[:, :, :z_midpoint] = False
            return femur_mask, tibia_mask
        return bone_mask, np.zeros_like(bone_mask)
    
    # We expect the femur and tibia to be the two largest components
    # Create masks for the two largest components
    component1 = labeled_mask == large_components[0]
    component2 = labeled_mask == large_components[1]
    
    # Calculate centers of mass for each component along all axes
    indices = np.indices(bone_mask.shape)
    
    # For component 1
    if np.sum(component1) > 0:  # Avoid division by zero
        x_center1 = np.sum(component1 * indices[0]) / np.sum(component1)
        y_center1 = np.sum(component1 * indices[1]) / np.sum(component1)
        z_center1 = np.sum(component1 * indices[2]) / np.sum(component1)
        print(f"Component 1 center: ({x_center1:.1f}, {y_center1:.1f}, {z_center1:.1f})")
    else:
        z_center1 = 0
    
    # For component 2
    if np.sum(component2) > 0:  # Avoid division by zero
        x_center2 = np.sum(component2 * indices[0]) / np.sum(component2)
        y_center2 = np.sum(component2 * indices[1]) / np.sum(component2)
        z_center2 = np.sum(component2 * indices[2]) / np.sum(component2)
        print(f"Component 2 center: ({x_center2:.1f}, {y_center2:.1f}, {z_center2:.1f})")
    else:
        z_center2 = 0
    
    # Determine which one is femur and which one is tibia based on their position along z-axis
    # In knee CT, femur is generally located in the upper part and tibia in the lower part
    # Assuming z-axis increases from head to foot
    if z_center1 < z_center2:
        # Component 1 is higher (smaller z value), so it's likely the femur
        femur_mask = component1
        tibia_mask = component2
        print("Component 1 identified as femur, Component 2 as tibia")
    else:
        # Component 2 is higher, so it's likely the femur
        femur_mask = component2
        tibia_mask = component1
        print("Component 2 identified as femur, Component 1 as tibia")
    
    return femur_mask, tibia_mask

def apply_morphological_operations(mask, iterations=2):
    """
    Apply morphological operations to clean up the mask.
    
    Args:
        mask: Binary mask
        iterations: Number of iterations for morphological operations
        
    Returns:
        cleaned_mask: Cleaned binary mask
    """
    # Apply closing to fill small holes
    cleaned_mask = ndimage.binary_closing(mask, iterations=iterations)
    
    # Apply opening to remove small isolated regions
    cleaned_mask = ndimage.binary_opening(cleaned_mask, iterations=iterations)
    
    return cleaned_mask

def visualize_segmentation(original_data, femur_mask, tibia_mask, num_slices=5, axis=2):
    """
    Visualize the segmentation results.
    
    Args:
        original_data: Original CT scan data
        femur_mask: Binary mask of the femur
        tibia_mask: Binary mask of the tibia
        num_slices: Number of slices to visualize
        axis: Axis along which to take slices (0=sagittal, 1=coronal, 2=axial)
    """
    # Get the size of the data along the specified axis
    axis_size = original_data.shape[axis]
    
    # Calculate the indices of the slices to visualize
    indices = np.linspace(200, axis_size - 200, num_slices, dtype=int)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 9))
    
    # Axis labels
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Loop through the indices and visualize each slice
    for i, idx in enumerate(indices):
        # Take slices along the specified axis
        if axis == 0:
            original_slice = original_data[idx, :, :]
            femur_slice = femur_mask[idx, :, :]
            tibia_slice = tibia_mask[idx, :, :]
        elif axis == 1:
            original_slice = original_data[:, idx, :]
            femur_slice = femur_mask[:, idx, :]
            tibia_slice = tibia_mask[:, idx, :]
        else:  # axis == 2
            original_slice = original_data[:, :, idx]
            femur_slice = femur_mask[:, :, idx]
            tibia_slice = tibia_mask[:, :, idx]
        
        # Normalize original slice for better visualization
        norm_slice = np.clip(original_slice, -300, 1500)  # Clip to bone window
        print(f"Slice {idx} min: {np.min(norm_slice)}, max: {np.max(norm_slice)}")
        norm_slice = (norm_slice - np.min(norm_slice)) / (np.max(norm_slice) - np.min(norm_slice))
        
        # Display the original slice
        axes[0, i].imshow(norm_slice.T, cmap='bone', origin='lower')
        axes[0, i].set_title(f'{axis_names[axis]} Slice {idx} - Original')
        axes[0, i].axis('off')
        
        # Create overlay for femur
        overlay_femur = np.zeros_like(norm_slice.T, dtype=np.float32)
        overlay_femur[femur_slice.T] = 1.0  # Full opacity where femur is present
        
        # Display the femur segmentation
        axes[1, i].imshow(norm_slice.T, cmap='bone', origin='lower')
        femur_overlay = axes[1, i].imshow(overlay_femur, cmap='hot', alpha=0.5, origin='lower')
        axes[1, i].set_title(f'{axis_names[axis]} Slice {idx} - Femur')
        axes[1, i].axis('off')
        
        # Create overlay for tibia
        overlay_tibia = np.zeros_like(norm_slice.T, dtype=np.float32)
        overlay_tibia[tibia_slice.T] = 1.0  # Full opacity where tibia is present
        
        # Display the tibia segmentation
        axes[2, i].imshow(norm_slice.T, cmap='bone', origin='lower')
        tibia_overlay = axes[2, i].imshow(overlay_tibia, cmap='cool', alpha=0.5, origin='lower')
        axes[2, i].set_title(f'{axis_names[axis]} Slice {idx} - Tibia')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

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
    morph_iterations = 2  # Number of iterations for morphological operations
    
    # Run the bone segmentation
    femur_mask, tibia_mask, combined_mask, affine, header = bone_segmentation(
        input_path, output_dir, bone_threshold, min_component_size, morph_iterations, visualize=True
    )