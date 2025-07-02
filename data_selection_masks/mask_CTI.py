#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""


import numpy as np
from astropy.io import fits
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import argparse


from utils_plot import set_plot_style, set_axis_labels_aligned
set_plot_style()
import seaborn as sns


############################################
# 1) Load data and generate base mask
############################################

def load_data(fits_file, extension_name='CALIBRATED'):
    """
    Load data from a given extension in a FITS file.
    """
    
    with fits.open(fits_file) as hdul:
        return np.copy(hdul[extension_name].data)

def get_complete_mask_for_all_seeds(data, seed_threshold=50.0):
    """
    Identify seed pixels (>= seed_threshold) without including tails.
    Returns a base mask with only the seeds.
    """
    
    struct = generate_binary_structure(2, 2)
    seed_pixels = (data >= seed_threshold)
    labeled_seeds, _ = label(seed_pixels, structure=struct)

    base_mask = (labeled_seeds > 0)

    return base_mask, labeled_seeds 

############################################
# 2) Expansion functions
############################################

def expand_vcti(base_mask, iterations=1):
    """
    Vertical dilation to simulate vertical charge transfer inefficiency (VCTI).
    """
    if iterations == 0:
        return base_mask
    v_structure = np.array([[1],[1],[1]])  # Vertical 3x1 structure
    return binary_dilation(base_mask, structure=v_structure, iterations=iterations)

def expand_hcti(base_mask, iterations=1):
    """
    Horizontal dilation to simulate horizontal charge transfer inefficiency (HCTI).
    """
    if iterations == 0:
        return base_mask
    h_structure = np.array([[1,1,1]])  # Horizontal 3x1 structure
    return binary_dilation(base_mask, structure=h_structure, iterations=iterations)

def expand_halo(base_mask, iterations=1):
    """
    "Halo" expansion using a cross-shaped 3x3 structure.
    """
    if iterations == 0:
        return base_mask
    halo_structure = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])
    return binary_dilation(base_mask, structure=halo_structure, iterations=iterations)

def plot_heatmap(fits_file, final_mask_combined):
    """
    Display a heatmap of a 30x30 pixel region centered at (1410, 4368)
    from the 'CALIBRATED' extension of a FITS file.

    The color scale is capped at 20. Axis ticks show real coordinates
    with a spacing of 15 pixels. The color bar spans the height of the heatmap.
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits


    # Load data from FITS
    data = fits.getdata(fits_file, extname='CALIBRATED')

    center_y, center_x = 1410, 4368
    half_size = 51

    y_min = max(center_y - half_size, 0)
    y_max = min(center_y + half_size, data.shape[0])
    x_min = max(center_x - half_size, 0)
    x_max = min(center_x + half_size, data.shape[1])

    cropped_data = data[y_min:y_max, x_min:x_max]

    y_labels = np.arange(y_min, y_max)
    x_labels = np.arange(x_min, x_max)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cropped_data, cmap="cividis", cbar=True, square=True,
        xticklabels=False, yticklabels=False,
        cbar_kws={'pad': 0.01}  
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params()
    

    # Set ticks every 15 pixels with real coordinate labels
    xtick_pos = np.arange(0, cropped_data.shape[1], 15)
    ytick_pos = np.arange(0, cropped_data.shape[0], 15)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(x_labels[xtick_pos])  # Eje X en vertical
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(y_labels[ytick_pos])

    ax.set_xlabel("column")
    ax.set_ylabel("row")

    ax.tick_params(axis='both', which='major', labelsize=20)  

    ax.invert_yaxis()
    set_axis_labels_aligned(ax)
    plt.tight_layout()
    plt.show()
    
    # Plot the mask
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        final_mask_combined[y_min:y_max, x_min:x_max], cmap="gray", cbar=False, square=True,
        xticklabels=False, yticklabels=False,
        cbar_kws={'pad': 0.01}  
    )


    xtick_pos = np.arange(0, cropped_data.shape[1], 15)
    ytick_pos = np.arange(0, cropped_data.shape[0], 15)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(x_labels[xtick_pos])  # Eje X en vertical
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(y_labels[ytick_pos])

    ax.set_xlabel("column")
    ax.set_ylabel("row")

    ax.tick_params(axis='both', which='major', labelsize=20)  

    ax.invert_yaxis()
    set_axis_labels_aligned(ax)
    plt.tight_layout()
    plt.show()
    



############################################
# 3) Main processing logic
############################################
                
def main(args):
    """
    For each FITS file, we create base masks for two cases:
      - (50 <= seed < 100): VCTI = 10 and HCTI = 100
      - (seed >= 100): VCTI = 100 and HCTI = 6300

    Each expansion is applied separately (starting from the original seed),
    and vertical/horizontal dilations are combined using a logical OR.

    Finally, both cases are combined (Case1 OR Case2) and the resulting mask
    is saved in the FITS file.
    """

    for fits_file in args.files:
        print(f"\n[INFO] Processing FITS file: {fits_file}")
        data = load_data(fits_file, extension_name='CALIBRATED')

        # -- Case 1: base_mask_1 = 50 <= data < 100
        base_mask_1 = (data >= 50) & (data < 100)

        # Apply vertical and horizontal expansion separately
        mask_v1 = expand_vcti(base_mask_1, iterations=10)   # VCTI = 10
        mask_h1 = expand_hcti(base_mask_1, iterations=100)  # HCTI = 100

        # Combine vertical and horizontal masks
        final_mask_1 = mask_v1 | mask_h1

        # -- Case 2: base_mask_2 = data >= 100
        base_mask_2 = (data >= 100)

        # Apply vertical and horizontal expansion separately
        mask_v2 = expand_vcti(base_mask_2, iterations=100)    # VCTI = 100
        mask_h2 = expand_hcti(base_mask_2, iterations=6300)   # HCTI = 6300
        mask_halo2 = expand_halo(base_mask_2, iterations=10)  # HALO

        # Combine all masks
        final_mask_2 = mask_v2 | mask_h2 | mask_halo2

        # Combine both cases
        final_mask_combined = final_mask_1 | final_mask_2
        
        
        import matplotlib.pyplot as plt
        import os
        os.makedirs('.', exist_ok=True)
        original_values = data.flatten()
        
        #plot_heatmap(fits_file, final_mask_combined)

        # Masked data
        masked_values = np.where(final_mask_combined == 1, np.nan, data).flatten()
        valid_values = masked_values[~np.isnan(masked_values)]

        # Save result in FITS
        with fits.open(fits_file, mode='update') as hdul:
            print("\n--- Original extensions ---")
            hdul.info()

            # Remove any existing 'MASK_CTI' extensions
            mask_cti_indices = [i for i, hdu in enumerate(hdul) if hdu.name == "MASK_CTI"]
            if mask_cti_indices:
                for index in reversed(mask_cti_indices):  # Remove in reverse order
                    del hdul[index]
                print(f"[INFO] Removed {len(mask_cti_indices)} existing instances of 'MASK_CTI'.")

            # Save new CTI mask as uint8
            mask_hdu = fits.ImageHDU(data=final_mask_combined.astype(np.uint8), name='MASK_CTI')
            hdul.append(mask_hdu)

            # Save changes to file
            hdul.flush()

        print("[INFO] CTI mask has been saved to the FITS file.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a CTI mask with different expansion depending on the seed range.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help="FITS files with calibrated data.")
    args = parser.parse_args()
    main(args)
