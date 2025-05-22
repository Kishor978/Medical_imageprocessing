import os
import numpy as np
from scipy import ndimage
from utils import load_ct_data, save_nifti,get_voxel_spacing,visualize_randomized_comparison, create_distance_field,apply_morphological_operations
from src import bone_segmentation, contour_expansion

def create_randomized_mask(original_mask, expanded_mask, max_expansion_mm, voxel_spacing, 
                          random_seed=None, random_factor=1.0):
    """
    Create a randomized mask that lies between the original and expanded masks.
    
    Args:
        original_mask: Original binary mask
        expanded_mask: Expanded binary mask (at max_expansion_mm)
        max_expansion_mm: Maximum expansion distance in millimeters
        voxel_spacing: Array of voxel dimensions in mm [x, y, z]
        random_seed: Random seed for reproducibility
        random_factor: Factor to control randomness (0.0 = no randomness, 1.0 = full randomness)
        
    Returns:
        randomized_mask: Randomized binary mask
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Creating randomized mask with max expansion {max_expansion_mm}mm and random factor {random_factor}")
    
    # Create the expansion region (area between original and expanded masks)
    expansion_region = expanded_mask & (~original_mask)
    
    if not np.any(expansion_region):
        print("Warning: No expansion region found. Returning original mask.")
        return original_mask.copy()
    
    # Create distance field from the original mask boundary
    distance_field = create_distance_field(original_mask, voxel_spacing)
    
    # Create a random field with the same shape as the mask
    random_field = np.random.random(original_mask.shape)
    
    # Apply smoothing to the random field to create more coherent random patterns
    smoothing_sigma = 2.0  # Adjust this to control the smoothness of randomization
    smooth_random_field = ndimage.gaussian_filter(random_field, sigma=smoothing_sigma)
    
    # Normalize the smooth random field to [0, 1]
    smooth_random_field = (smooth_random_field - np.min(smooth_random_field)) / \
                         (np.max(smooth_random_field) - np.min(smooth_random_field))
    
    # Create the randomized mask
    randomized_mask = original_mask.copy()
    
    # For each voxel in the expansion region, decide whether to include it
    # based on its distance from the original boundary and the random field
    expansion_indices = np.where(expansion_region)
    
    for i in range(len(expansion_indices[0])):
        x, y, z = expansion_indices[0][i], expansion_indices[1][i], expansion_indices[2][i]
        
        # Get the distance of this voxel from the original boundary
        voxel_distance = distance_field[x, y, z]
        
        # Calculate the probability of inclusion based on distance
        # Closer to original boundary = higher probability
        # Closer to max expansion = lower probability
        if max_expansion_mm > 0:
            distance_factor = 1.0 - (voxel_distance / max_expansion_mm)
            distance_factor = np.clip(distance_factor, 0.0, 1.0)
        else:
            distance_factor = 1.0
        
        # Get the random value for this voxel
        random_value = smooth_random_field[x, y, z]
        
        # Combine distance factor and randomness
        # Higher random_factor means more randomness, lower means more distance-based
        combined_probability = (1.0 - random_factor) * distance_factor + \
                              random_factor * random_value
        
        # Decide whether to include this voxel (threshold at 0.5)
        if combined_probability > 0.5:
            randomized_mask[x, y, z] = True
    
    return randomized_mask


def randomized_contour_adjustment(original_femur_mask, original_tibia_mask, 
                                expanded_femur_mask, expanded_tibia_mask,
                                affine, header, max_expansion_mm=2.0, 
                                num_random_masks=2, output_dir="results/task1_3", 
                                visualize=True, original_data=None):
    """
    Main function for randomized contour adjustment.
    
    Args:
        original_femur_mask: Original femur binary mask
        original_tibia_mask: Original tibia binary mask
        expanded_femur_mask: Expanded femur binary mask
        expanded_tibia_mask: Expanded tibia binary mask
        affine: Affine transformation matrix
        header: NIfTI header
        max_expansion_mm: Maximum expansion distance in millimeters (parameterized)
        num_random_masks: Number of randomized masks to generate
        output_dir: Directory to save the randomized masks
        visualize: Whether to visualize the randomization results
        original_data: Original CT data for visualization
        
    Returns:
        randomized_femur_masks: List of randomized femur masks
        randomized_tibia_masks: List of randomized tibia masks
    """
    print(f"Starting randomized contour adjustment...")
    print(f"Max expansion: {max_expansion_mm}mm")
    print(f"Number of random masks to generate: {num_random_masks}")
    
    # Get voxel spacing from the affine matrix
    voxel_spacing = get_voxel_spacing(affine)
    print(f"Voxel spacing: {voxel_spacing} mm")
    
    # Lists to store randomized masks
    randomized_femur_masks = []
    randomized_tibia_masks = []
    
    # Generate randomized masks
    for i in range(num_random_masks):
        print(f"\nGenerating randomized mask set {i+1}...")
        
        # Create randomized femur mask
        random_seed_femur = 42 + i * 10  # Different seeds for different masks
        randomized_femur = create_randomized_mask(
            original_femur_mask, expanded_femur_mask, max_expansion_mm, 
            voxel_spacing, random_seed=random_seed_femur, random_factor=0.7
        )
        
        # Apply morphological smoothing
        randomized_femur = apply_morphological_operations(randomized_femur, iterations=1)
        
        # Create randomized tibia mask
        random_seed_tibia = 84 + i * 10  # Different seeds for different masks
        randomized_tibia = create_randomized_mask(
            original_tibia_mask, expanded_tibia_mask, max_expansion_mm, 
            voxel_spacing, random_seed=random_seed_tibia, random_factor=0.7
        )
        
        # Apply morphological smoothing
        randomized_tibia = apply_morphological_operations(randomized_tibia, iterations=1)
        
        # Store the masks
        randomized_femur_masks.append(randomized_femur)
        randomized_tibia_masks.append(randomized_tibia)
        
        # Create combined randomized mask
        combined_randomized_mask = np.zeros_like(randomized_femur, dtype=np.int16)
        combined_randomized_mask[randomized_femur] = 1  # Femur labeled as 1
        combined_randomized_mask[randomized_tibia] = 2   # Tibia labeled as 2
        
        # Handle overlapping regions
        overlap = randomized_femur & randomized_tibia
        if np.any(overlap):
            print(f"Found {np.sum(overlap)} overlapping voxels in randomized mask {i+1}.")
            # Resolve overlaps by checking original masks
            femur_priority = overlap & original_femur_mask
            tibia_priority = overlap & original_tibia_mask
            
            # Remove overlap
            randomized_femur &= ~overlap
            randomized_tibia &= ~overlap
            
            # Add back based on original priority
            randomized_femur |= femur_priority
            randomized_tibia |= tibia_priority
            
            # Update combined mask
            combined_randomized_mask = np.zeros_like(randomized_femur, dtype=np.int16)
            combined_randomized_mask[randomized_femur] = 1
            combined_randomized_mask[randomized_tibia] = 2
        
        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the randomized masks
        print(f"Saving randomized mask set {i+1}...")
        save_nifti(randomized_femur.astype(np.int16), affine, header, 
                   os.path.join(output_dir, f"femur_randomized_{i+1}.nii.gz"))
        save_nifti(randomized_tibia.astype(np.int16), affine, header, 
                   os.path.join(output_dir, f"tibia_randomized_{i+1}.nii.gz"))
        save_nifti(combined_randomized_mask, affine, header, 
                   os.path.join(output_dir, f"combined_randomized_{i+1}.nii.gz"))
    
    # Visualize the randomization results if requested
    if visualize and original_data is not None:
        print("Visualizing randomization results...")
        
        # Visualize femur randomization
        visualize_randomized_comparison(original_data, original_femur_mask, expanded_femur_mask,
                                      randomized_femur_masks, "Femur", num_slices=5, axis=2)
        visualize_randomized_comparison(original_data, original_femur_mask, expanded_femur_mask,
                                      randomized_femur_masks, "Femur", num_slices=5, axis=1)
        
        # Visualize tibia randomization
        visualize_randomized_comparison(original_data, original_tibia_mask, expanded_tibia_mask,
                                      randomized_tibia_masks, "Tibia", num_slices=5, axis=2)
        visualize_randomized_comparison(original_data, original_tibia_mask, expanded_tibia_mask,
                                      randomized_tibia_masks, "Tibia", num_slices=5, axis=1)
    
    print(f"Randomized contour adjustment completed successfully.")
    print(f"Generated {num_random_masks} randomized mask sets.")
    
    return randomized_femur_masks, randomized_tibia_masks

def main():
    """
    Main function to run Tasks 1.1, 1.2, and 1.3.
    """
    # Parameters
    input_path = "data\\3702_left_knee.nii.gz"
    task1_1_output_dir = "results\\task1_1"
    task1_2_output_dir = "results\\task1_2"
    task1_3_output_dir = "results\\task1_3"
    
    # Task 1.1 parameters
    bone_threshold = 200
    min_component_size = 10000
    morph_iterations = 1
    
    # Task 1.2 parameters
    expansion_mm = 2.0  # Parameterized as required
    
    # Task 1.3 parameters
    max_expansion_mm = 2.0  # Parameterized as required
    num_random_masks = 2  # Generate 2 randomized masks as mentioned in the task
    
    # Load original data for visualization
    original_data, _, _ = load_ct_data(input_path)
    
    # Run Task 1.1 - Bone Segmentation
    print("="*60)
    print("RUNNING TASK 1.1 - BONE SEGMENTATION")
    print("="*60)
    
    femur_mask, tibia_mask, combined_mask, affine, header = bone_segmentation(
        input_path, task1_1_output_dir, bone_threshold, min_component_size, 
        morph_iterations, visualize=False
    )
    
    # Run Task 1.2 - Contour Expansion
    print("="*60)
    print("RUNNING TASK 1.2 - CONTOUR EXPANSION")
    print("="*60)
    
    expanded_femur_mask, expanded_tibia_mask = contour_expansion(
        femur_mask, tibia_mask, affine, header, expansion_mm, 
        task1_2_output_dir, visualize=False, original_data=original_data
    )
    
    # Run Task 1.3 - Randomized Contour Adjustment
    print("="*60)
    print("RUNNING TASK 1.3 - RANDOMIZED CONTOUR ADJUSTMENT")
    print("="*60)
    
    randomized_femur_masks, randomized_tibia_masks = randomized_contour_adjustment(
        femur_mask, tibia_mask, expanded_femur_mask, expanded_tibia_mask,
        affine, header, max_expansion_mm, num_random_masks, 
        task1_3_output_dir, visualize=True, original_data=original_data
    )
    
    print("="*60)
    print("TASKS 1.1, 1.2, AND 1.3 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved in:")
    print(f"  Task 1.1: {task1_1_output_dir}")
    print(f"  Task 1.2: {task1_2_output_dir}")
    print(f"  Task 1.3: {task1_3_output_dir}")

if __name__ == "__main__":
    main()