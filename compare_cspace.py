import numpy as np
import matplotlib.pyplot as plt

def visualize_dim_cspaces(cspace1, cspace2, title1='cspace_seq', title2='cspace_par'):
    assert cspace1.shape == cspace2.shape, "Both cspaces must have the same shape"
    assert len(cspace1.shape) >= 2, "Cspace dimensions must be at least 2"

    num_dims = len(cspace1.shape)
    fig, axs = plt.subplots(1, num_dims, figsize=(15, 5))
    
    for dim, ax in enumerate(axs):
        slice_idx1 = [slice(None) if i != dim else 0 for i in range(num_dims)]
        slice_idx2 = [slice(None) if i != dim else -1 for i in range(num_dims)]
        
        slice1 = cspace1[tuple(slice_idx1)]
        slice2 = cspace2[tuple(slice_idx2)]
        
        # Plotting
        im = ax.imshow(np.concatenate([slice1, np.zeros((slice1.shape[0], 1)), slice2], axis=1), cmap='viridis')
        ax.set_title(f"Dim {dim}: {title1} vs {title2}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

# Example usage
cspace_seq = np.load('cspace_seq.npy')
cspace_par = np.load('cspace_par.npy')

visualize_dim_cspaces(cspace_seq, cspace_par)

