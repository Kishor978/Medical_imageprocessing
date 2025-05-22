import os
import numpy as np
from src.segement import bone_segmentation
from src.contour_expansion import contour_expansion
from src.randomized_contour import randomized_contour_adjustment
from utils import save_nifti

def find_lowest_points(mask, affine):
    """
    Finds the medial and lateral lowest points on the tibial surface.
    
    Args:
        mask (np.ndarray): Input mask
        affine (np.ndarray): Affine matrix for coordinate transformation
    
    Returns:
        tuple: ((medial_x, medial_y, medial_z), (lateral_x, lateral_y, lateral_z))
    """
    # Get voxel coordinates of the mask
    points = np.where(mask > 0)
    
    # Create homogeneous coordinates (N x 4 matrix)
    homogeneous_coords = np.vstack([points[0], points[1], points[2], np.ones(len(points[0]))])
    
    # Transform to world coordinates (4 x N matrix)
    world_coords = np.dot(affine, homogeneous_coords)[:3]  # Take only x,y,z (drop homogeneous component)
    
    # Find points near the bottom (maximum z-coordinate)
    z_coords = world_coords[2]
    z_threshold = np.max(z_coords) - 5  # Consider points within 5mm of lowest
    lowest_points_mask = z_coords > z_threshold
    
    # Filter coordinates to only include lowest points
    lowest_x = world_coords[0][lowest_points_mask]
    lowest_y = world_coords[1][lowest_points_mask]
    lowest_z = world_coords[2][lowest_points_mask]
    
    # Separate medial and lateral based on x-coordinate
    median_x = np.median(lowest_x)
    medial_mask = lowest_x > median_x
    lateral_mask = lowest_x <= median_x
    
    # Find the lowest point in each region
    medial_z = lowest_z[medial_mask]
    lateral_z = lowest_z[lateral_mask]
    
    if len(medial_z) == 0 or len(lateral_z) == 0:
        raise ValueError("Could not find both medial and lateral points")
    
    medial_idx = np.argmax(medial_z)
    lateral_idx = np.argmax(lateral_z)
    
    # Get the final coordinates
    medial_point = (
        lowest_x[medial_mask][medial_idx],
        lowest_y[medial_mask][medial_idx],
        lowest_z[medial_mask][medial_idx]
    )
    
    lateral_point = (
        lowest_x[lateral_mask][lateral_idx],
        lowest_y[lateral_mask][lateral_idx],
        lowest_z[lateral_mask][lateral_idx]
    )
    
    return medial_point, lateral_point
def main():
    # Define paths
    input_path = "data/3702_left_knee.nii.gz"
    output_dir = "results/task1_4"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original masks from bone segmentation
    femur_mask, tibia_mask, combined_mask, affine, header = bone_segmentation(
        input_path=input_path,
        output_dir=output_dir,
        visualize=False
    )
    print("Bone segmentation completed successfully.")
    print("Generating masks...")
    print("generating expanded masks of 2mm")
    # Generate 2mm and 4mm expanded masks
    expanded_femur_mask_2mm, expanded_tibia_mask_2mm = contour_expansion(
        femur_mask, tibia_mask, affine, header, expansion_mm=2.0, visualize=False
    )
    
    print("generating expanded masks of 4mm")
    expanded_femur_mask_4mm, expanded_tibia_mask_4mm = contour_expansion(
        femur_mask,tibia_mask, affine, header, expansion_mm=4.0,visualize=False
    )
    print("Contour expansion completed successfully.")
    # Generate randomized masks
    _, randomized_tibia_masks_2mm = randomized_contour_adjustment(
        original_femur_mask=femur_mask,
        original_tibia_mask=tibia_mask,
        expanded_femur_mask=expanded_femur_mask_4mm,
        expanded_tibia_mask=expanded_tibia_mask_4mm,
        
        affine=affine,
        header=header,
        max_expansion_mm=2.0,
        num_random_masks=2,
        output_dir=output_dir,
        visualize=False
    )
    
    # Organize all masks
    masks = {
        'original': tibia_mask,
        'expanded_2mm': expanded_tibia_mask_2mm,
        'expanded_4mm': expanded_tibia_mask_4mm,
        'random_1': randomized_tibia_masks_2mm[0],
        'random_2': randomized_tibia_masks_2mm[1]
    }
    
    # Save all masks
    print("Saving masks...")
    for name, mask in masks.items():
        save_nifti(
            data=mask.astype(np.int16),
            affine=affine,
            header=header,
            output_path=os.path.join(output_dir, f"submissiion/tibia_{name}.nii.gz")
        )
    
    # Find and save landmarks
    print("Finding landmarks...")
    landmarks = {}
    for name, mask in masks.items():
        medial, lateral = find_lowest_points(mask, affine)
        landmarks[name] = {'medial': medial, 'lateral': lateral}
    
    # Save landmarks to file
    with open(os.path.join(output_dir, 'submissiion/landmarks.txt'), 'w') as f:
        for mask_name, points in landmarks.items():
            f.write(f"\n{mask_name.upper()} MASK:\n")
            f.write(f"Medial point: {points['medial']}\n")
            f.write(f"Lateral point: {points['lateral']}\n")
    
    print("Task 1.4 completed successfully.")

if __name__ == "__main__":
    main()