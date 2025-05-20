import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def load_ct_data(file_path):
    """
    Load CT scan data from a file.
    
    Args:
        file_path: Path to the CT scan file
        
    Returns:
        data: CT scan data as a numpy array
        affine: Affine transformation matrix
        header: Header information from the file
    """
    # Load the CT scan using nibabel
    ct_img = nib.load(file_path)
    
    # Get the data as a numpy array
    data = ct_img.get_fdata()
    
    # Get the affine transformation matrix
    affine = ct_img.affine
    
    # Get the header information
    header = ct_img.header
    
    return data, affine, header

def visualize_slices(data, num_slices=5, axis=2):
    """
    Visualize multiple slices of the CT scan along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        num_slices: Number of slices to visualize
        axis: Axis along which to take slices (0=sagittal, 1=coronal, 2=axial)
    """
    # Get the size of the data along the specified axis
    axis_size = data.shape[axis]
    
    # Calculate the indices of the slices to visualize
    indices = np.linspace(0, axis_size - 1, num_slices, dtype=int)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    
    # Axis labels
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Loop through the indices and visualize each slice
    for i, idx in enumerate(indices):
        # Take a slice along the specified axis
        if axis == 0:
            slice_data = data[idx, :, :]
        elif axis == 1:
            slice_data = data[:, idx, :]
        else:  # axis == 2
            slice_data = data[:, :, idx]
        
        # Display the slice
        axes[i].imshow(slice_data.T, cmap='bone', origin='lower')
        axes[i].set_title(f'{axis_names[axis]} Slice {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_3d_histograms(data, bins=50):
    """
    Visualize the histogram of voxel intensities in the CT scan.
    
    Args:
        data: CT scan data as a numpy array
        bins: Number of bins for the histogram
    """
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(data.flatten(), bins=bins)
    plt.title('Histogram of CT Scan Voxel Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add a vertical line at a potential threshold for bone segmentation
    # This is a starting point that will need to be adjusted
    threshold = np.percentile(data, 95)  # 95th percentile as an initial guess
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Potential Threshold ({threshold:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_mip_projection(data, axis=2):
    """
    Create a Maximum Intensity Projection (MIP) along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        axis: Axis along which to project (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        mip: Maximum Intensity Projection as a 2D numpy array
    """
    # Create the MIP by taking the maximum value along the specified axis
    mip = np.max(data, axis=axis)
    
    return mip

def visualize_mip(data):
    """
    Visualize Maximum Intensity Projections (MIPs) along all three axes.
    
    Args:
        data: CT scan data as a numpy array
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axis names
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Loop through the axes and create MIPs
    for i in range(3):
        # Create the MIP
        mip = create_mip_projection(data, axis=i)
        
        # Display the MIP
        axes[i].imshow(mip.T, cmap='bone', origin='lower')
        axes[i].set_title(f'{axis_names[i]} MIP')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_animation(data, axis=2):
    """
    Create an animation of slices along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        axis: Axis along which to animate (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        anim: Animation object
    """
    # Get the size of the data along the specified axis
    axis_size = data.shape[axis]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Axis names
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Take a slice along the specified axis
        if axis == 0:
            slice_data = data[frame, :, :]
        elif axis == 1:
            slice_data = data[:, frame, :]
        else:  # axis == 2
            slice_data = data[:, :, frame]
        
        # Display the slice
        ax.imshow(slice_data.T, cmap='bone', origin='lower')
        ax.set_title(f'{axis_names[axis]} Slice {frame}')
        ax.axis('off')
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=axis_size, interval=50)
    
    return anim

def main():
    # Path to the CT scan file
    file_path = 'data\\3702_left_knee.nii.gz'  
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please download the file from the provided link and adjust the path accordingly.")
        return
    
    # Load the CT scan data
    print("Loading CT scan data...")
    data, affine, header = load_ct_data(file_path)
    
    # Print information about the data
    print(f"CT scan data loaded successfully.")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data)}, {np.max(data)}]")
    
    # Visualize axial slices
    print("Visualizing axial slices...")
    visualize_slices(data, num_slices=5, axis=2)
    
    # Visualize coronal slices
    print("Visualizing coronal slices...")
    visualize_slices(data, num_slices=5, axis=1)
    
    # Visualize sagittal slices
    print("Visualizing sagittal slices...")
    visualize_slices(data, num_slices=5, axis=0)
        
    # Visualize the histogram
    print("Visualizing histogram...")
    visualize_3d_histograms(data)
    
    # Visualize MIPs
    print("Visualizing Maximum Intensity Projections...")
    visualize_mip(data)
    
    # Create an animation of axial slices
    print("Creating animation of axial slices...")
    anim = create_animation(data, axis=2)
    anim.save('ct_scan_animation.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation.gif'")
    anim.event_source.stop()  # Stop the animation event source
    
    anim = create_animation(data, axis=1)
    anim.save('ct_scan_animation_coronal.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation_coronal.gif'")
    anim.event_source.stop()  # Stop the animation event source
    anim = create_animation(data, axis=0)
    anim.save('ct_scan_animation_sagittal.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation_sagittal.gif'")
    anim.event_source.stop()  # Stop the animation event source
    
    # Display the animation (this will only work in notebooks or with plt.show())
    plt.show()

if __name__ == "__main__":
    main()