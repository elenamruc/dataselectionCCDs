import numpy as np
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

############################################
# 1) Load data and generate base mask
############################################

def load_data(fits_file, extension_name='CALIBRATED'):
    """
    Load data from a given extension in a FITS file.
    """

    with fits.open(fits_file) as hdul:
        return np.copy(hdul[extension_name].data)

def load_mask_if_exists(hdul, ext_name):
    """
    Try to load the extension 'ext_name' as a boolean mask (True = masked).
    Returns None if the extension does not exist.
    """
    
    if ext_name in [hdu.name for hdu in hdul]:
        mask_data = hdul[ext_name].data
        return (mask_data != 0)  # Convertir 0/1 a bool
    return None


def build_partial_masks(fits_file, ordered_exts):
    """
    Reads a series of pre-existing extensions (CLSMASK, MASK_CROSSTALK, etc.)
    and builds a cumulative mask by combining them using logical OR.

    Returns:
        final_mask (bool array): True where pixels are masked.
    """
    
    with fits.open(fits_file) as hdul:
        
        # 1) Initialize empty mask
        shape_data = hdul['CALIBRATED'].data.shape
        total_pixels_full = np.prod(shape_data)
        partial_mask = np.zeros(shape_data, dtype=bool)
        mask_list = []
        
        valid_cols = slice(8, 6152)
        total_pixels = shape_data[0] * (valid_cols.stop - valid_cols.start)

        # Descriptive labels for each mask
        mask_labels = [
            "Clustering+AR", "Cluster Crosstalk", "Hot Columns", "Overdensity", 
            "Hight-Q Pixels", "Charge Multiplicity", "Isolated Columns"
        ]

        # First entry: no mask
        unmasked_pixels = np.sum(~partial_mask[:, valid_cols])
        masked_pixels = np.sum(partial_mask[:, valid_cols])
        unmasked_percentage = (unmasked_pixels / total_pixels) * 100
        
        unmasked_pixels_full = np.sum(~partial_mask)
        masked_pixels_full = np.sum(partial_mask)
        unmasked_percentage_full = (unmasked_pixels_full / total_pixels_full) * 100

        mask_list.append((partial_mask.copy(), "Unmasked Data", unmasked_percentage, masked_pixels, unmasked_percentage_full, masked_pixels_full))

        # 2) Apply masks cumulatively
        current_label = ""
        for i, ext_name in enumerate(ordered_exts):
            this_mask = load_mask_if_exists(hdul, ext_name)
            if this_mask is not None:
                partial_mask |= this_mask  # Logical OR
                current_label = mask_labels[i] if i < len(mask_labels) else ext_name

                unmasked_pixels = np.sum(~partial_mask[:, valid_cols])
                masked_pixels = np.sum(partial_mask[:, valid_cols])
                unmasked_percentage = (unmasked_pixels / total_pixels) * 100
                
                unmasked_pixels_full = np.sum(~partial_mask)
                masked_pixels_full = np.sum(partial_mask)
                unmasked_percentage_full = (unmasked_pixels_full / total_pixels_full) * 100

                mask_list.append((partial_mask.copy(), current_label, unmasked_percentage, masked_pixels, unmasked_percentage_full, masked_pixels_full))
                

    return mask_list

############################################
# 2) Function to mask isolated columns
############################################
    
def mask_isolated_columns(mask):
    """
    Returns a new mask containing only the isolated columns.
    A column is considered isolated if:
      - both its neighbors (left and right) are fully masked
      - the column itself is NOT fully masked
    """
    
    nrows, ncols = mask.shape
    new_mask = np.zeros_like(mask, dtype=bool)
    number_of_col = 0

    for j in range(1, ncols - 1):
        masked_left = np.all(mask[:, j - 1])
        masked_right = np.all(mask[:, j + 1])
        masked_self = np.all(mask[:, j])

        if masked_left and masked_right and not masked_self:
            new_mask[:, j] = True
            number_of_col += 1

    print(f"Number of Isolated Columns: {number_of_col}")
    return new_mask


############################################
# 3) Main
############################################
def main(args):

    for fits_file in args.files:
        print(f"\n[INFO] Processing: {fits_file}")

        # --- A) Load data ---
        data = load_data(fits_file, extension_name='CALIBRATED')
        print(f"[INFO] data.shape = {data.shape}, min={data.min()}, max={data.max()}")

        # --- B) Build list of cumulative partial masks ---
        mask_list = build_partial_masks(fits_file, ordered_exts = ['CLSMASK', 'MASK_CROSSTALK', 'MASK_HOTCOLUMNS', 'MASK_OVERDENSITY', 'MASK_CTI', 'MASK_MULTIPLICITY'])
        final_mask = mask_list[-1][0]  # Last accumulated mask
        
        # --- C) Apply the isolated columns mask ---
        isolated_columns_mask = mask_isolated_columns(final_mask)     

        # --- D) Save new masks into the FITS file ---
        with fits.open(fits_file, mode='update') as hdul:
            # **Remove previous extensions**
            for mask_name in ["MASK_ISOLATED_COLUMNS"]:
                indices = [i for i, hdu in enumerate(hdul) if hdu.name == mask_name]
                for index in reversed(indices):
                    del hdul[index]
                print(f"[INFO] Removed {len(indices)} previous instances of '{mask_name}'.")

            # **Append new masks**
            hdul.append(fits.ImageHDU(data=isolated_columns_mask.astype(np.uint8), name='MASK_ISOLATED_COLUMNS'))
            hdul.flush()

        print("[INFO] Masks saved successfully into the FITS file.")
        
        mask_list_final = build_partial_masks(fits_file, ordered_exts = ['CLSMASK', 'MASK_CROSSTALK', 'MASK_HOTCOLUMNS', 
        'MASK_OVERDENSITY', 'MASK_CTI', 'MASK_MULTIPLICITY', 'MASK_ISOLATED_COLUMNS'])
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply progressive masks including the isolated column mask.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help="FITS files to process.")
    args = parser.parse_args()
    main(args)
