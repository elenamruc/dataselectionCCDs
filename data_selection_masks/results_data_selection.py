#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""

import numpy as np
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_plot import set_plot_style, set_axis_labels_aligned
from scipy.special import factorial
from scipy.optimize import curve_fit
import math

set_plot_style()

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
        return (mask_data != 0)  # Convert 0/1 to boolean
    return None

def combine_masks(masks_dict):
    """
    Combines all available masks using a logical OR.
    
    Returns:
        A single boolean mask where any masked pixel in any input mask is masked.
    """
    combined_mask = None
    for mask in masks_dict.values():
        if mask is not None:
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask |= mask
    return combined_mask
    
############################################
# 2) Cumulative efficiencies
############################################

def build_partial_masks(fits_file, ordered_exts):
    """
    Builds both cumulative and individual masks from a FITS file.

    Returns:
        list of tuples: Cumulative masks with associated info (mask, label, %, count, full %, full count).
        list of tuples: Individual masks with stats for each extension.
    """
    
    with fits.open(fits_file) as hdul:
        shape_data = hdul['CALIBRATED'].data.shape
        total_pixels_full = np.prod(shape_data)
        partial_mask = np.zeros(shape_data, dtype=bool)
        mask_list = []
        individual_mask_list = []
        valid_cols = slice(8, 6152)
        total_pixels = shape_data[0] * (valid_cols.stop - valid_cols.start)
        
        mask_labels = [
            "Clustering+AR", "Cluster Crosstalk", "Hot Columns", "Overdensity",
            "Hight-Q Pixels", "Charge Multiplicity", "Isolated Columns"
        ]

        unmasked_pixels = np.sum(~partial_mask[:, valid_cols])
        masked_pixels = np.sum(partial_mask[:, valid_cols])
        unmasked_percentage = (unmasked_pixels / total_pixels) * 100
        unmasked_pixels_full = np.sum(~partial_mask)
        masked_pixels_full = np.sum(partial_mask)
        unmasked_percentage_full = (unmasked_pixels_full / total_pixels_full) * 100

        mask_list.append((partial_mask.copy(), "Unmasked Data", unmasked_percentage, masked_pixels, unmasked_percentage_full, masked_pixels_full))
        print(f"[INFO] 'Sin máscara' -> {unmasked_percentage:.2f}% ({unmasked_percentage_full:.2f}% en toda la matriz), {masked_pixels} enmascarados ({masked_pixels_full} en toda la matriz)")

        current_label = ""
        for i, ext_name in enumerate(ordered_exts):
            this_mask = load_mask_if_exists(hdul, ext_name)
            if this_mask is not None:
                individual_mask = this_mask.copy()  # Save individual mask
                masked_pixels_active = np.sum(individual_mask[:, valid_cols])
                masked_pixels_total = np.sum(individual_mask)
                individual_mask_list.append((individual_mask, mask_labels[i] if i < len(mask_labels) else ext_name, masked_pixels_active, masked_pixels_total))
                
                partial_mask |= this_mask  # Logical cumulative OR 
                current_label = mask_labels[i] if i < len(mask_labels) else ext_name

                unmasked_pixels = np.sum(~partial_mask[:, valid_cols])
                masked_pixels = np.sum(partial_mask[:, valid_cols])
                unmasked_percentage = (unmasked_pixels / total_pixels) * 100
                unmasked_pixels_full = np.sum(~partial_mask)
                masked_pixels_full = np.sum(partial_mask)
                unmasked_percentage_full = (unmasked_pixels_full / total_pixels_full) * 100

                mask_list.append((partial_mask.copy(), current_label, unmasked_percentage, masked_pixels, unmasked_percentage_full, masked_pixels_full))
                print(f"[INFO] '{current_label}' -> {unmasked_percentage:.2f}% ({unmasked_percentage_full:.2f}% in full matrix), {masked_pixels} masked AR ({masked_pixels_full} in full matrix)")
    return mask_list, individual_mask_list
    
    
############################################
# 3) Histograms and mask plots
############################################
    
def plot_histograms_log(data, mask_list, nbins=5000):
    """
    Plots log-scale histograms of unmasked pixel values after applying successive masks.
    """
    plt.figure(figsize=(12,10))
    
    for i, (mask, label, _, _, _, _) in enumerate(mask_list, start=1):
        unmasked_data = data[~mask] 
        counts, bins = np.histogram(unmasked_data, bins=nbins, range=(-2, 4))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        plt.step(bin_centers, counts, where='mid', label=label)

    plt.xlabel(r"charge (e$^{-}$)")
    plt.ylabel("counts")
    plt.xlim(-2, 4)
    plt.legend(loc='upper right', fontsize = 22)
    plt.yscale('log')
    plt.xticks()
    plt.yticks()
    ax = plt.gca()             
    ax.minorticks_on()         
    set_axis_labels_aligned(ax) 
    plt.tight_layout()
    plt.show()

def plot_final_mask(data, final_mask):
    """
    Plots a 2D image of the data, applying a mask (masked pixels shown as NaN).
    """
    data_to_show = data.copy().astype(float)
    data_to_show[final_mask] = np.nan  # Enmascarar los píxeles correspondientes

    plt.figure(figsize=(12, 8))
    img =plt.imshow(data_to_show, origin='lower', cmap='viridis', interpolation='nearest', aspect='equal', vmin=-2, vmax=4)
    plt.xlabel("column")
    plt.ylabel("row")
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    set_axis_labels_aligned()    
    plt.show()
    
    
############################################
# 4) Poisson + Gauss fit
############################################

def poisson_gauss_conv(x, A, lam, sigma, kmax=10):
    """
    Poisson-Gaussian convolution model:
    For each x[i], sum over k=0 to kmax:
        P(k; lambda) * Gaussian(x[i]; mean=k, sigma)

    Parameters:
        x (array-like): Input variable (charge axis).
        A (float): Global normalization factor.
        lam (float): Poisson mean (λ).
        sigma (float): Gaussian width.
        kmax (int): Maximum k to include in the sum.

    Returns:
        ndarray: Model evaluated at each x[i].
    """
    # Ensure input is an array
    x = np.asarray(x, dtype=float)

    # Pre-compute PMF(k; lambda)
    poisson_probs = [ (lam**k * math.exp(-lam) / math.factorial(k)) for k in range(kmax+1) ]

    # Output initialized with same shape as x
    out = np.zeros_like(x, dtype=float)

    # For each k, add contribution: Poisson(k) * Gauss(x - k)
    for k in range(kmax+1):
        pk = poisson_probs[k]
        gauss_k = np.exp(-(x - k)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
        out += pk * gauss_k

    return A * out

def fit_poisson_gauss(data, nbins=100, range_hist=(-2,3), kmax=10):
    """
    Fits the Poisson-Gaussian convolution model to the histogram of 'data'.

    Returns:
        tuple:
            - popt (array): Best-fit parameters [A, λ, σ].
            - pcov (2D array): Covariance matrix of the fit.
            - bin_centers (array): Centers of histogram bins.
            - counts (array): Counts per bin.
    """
    # 1) Build histogram
    counts, edges = np.histogram(data, bins=nbins, range=range_hist)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])


    # 2) Initial parameter guess: A, lambda, sigma
    #   - A ~ height of main peak
    #   - lambda ~ ~1 e- (e.g., 0.5 to 1.5)
    #   - sigma ~ 0.2–0.3 depending on noise
    guess = [np.max(counts), 1.0, 0.2]

    # 3) Fit model
    popt, pcov = curve_fit(
        lambda x, A, lam, sigma: poisson_gauss_conv(x, A, lam, sigma, kmax=kmax),
        bin_centers, counts, p0=guess
    )

    return popt, pcov, bin_centers, counts
    
        
def main(args):
    for fits_file in args.files:
        print(f"\n[INFO] Processing: {fits_file}")
        data = load_data(fits_file, extension_name='CALIBRATED')

        # Apply masking
        # Final mask is the last in the list of cumulative masks
        mask_list, individual_mask_list = build_partial_masks(fits_file, ordered_exts=[
            'CLSMASK', 'MASK_CROSSTALK', 'MASK_HOTCOLUMNS', 
            'MASK_OVERDENSITY', 'MASK_CTI', 'MASK_MULTIPLICITY', 'MASK_ISOLATED_COLUMNS'
        ])
        
        final_mask = mask_list[-1][0]
        
        print("\nTotal masked pixels after applying all masks:")
        for mask, label, _, masked_pixels, _, _ in mask_list:
            print(f"  - {label}: {masked_pixels} masked pixels")
            
            # Apply mask to data
            masked_data = data.copy()
            masked_data[mask] = 0     # Masked pixels set to zero (for display or analysis)
        
        valid_cols = slice(8, 6152)
        total_pixels = data.shape[0] * (valid_cols.stop - valid_cols.start)
        print("\nMasked pixels by individual masks:")
        for mask, label, masked_active, masked_total in individual_mask_list:    
            unmasked_percentage = ((total_pixels- masked_active) / total_pixels)
            print(f"  - {label}: {masked_active} in active region ({masked_total} total), percentage: {unmasked_percentage:.3f}")
            
         
        # C) Plot: histogram in log-scale for each partial mask
        plot_histograms_log(data, mask_list, nbins=300) 

        # D) Plot: 2D image after final cumulative mask
        plot_final_mask(data, final_mask)
        
        unmasked_data = data[~final_mask]
        counts, bins = np.histogram(unmasked_data, bins=500, range=(-2.5,3.5))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
    

        # Poisson-Gauss model fit
        popt, pcov, bin_centers, counts_g = fit_poisson_gauss(
            unmasked_data,
            nbins=500,
            range_hist=(-2.5,3.5),
            kmax=5
        )

        # Extract optimal parameters and errors
        A_opt, lam_opt, sigma_opt = popt
        err_A = np.sqrt(pcov[0,0])
        err_lam = np.sqrt(pcov[1,1])
        err_sigma = np.sqrt(pcov[2,2])
        
        # Unit conversion: lambda from e-/pix/image to e-/pix/day and then to e-/g/day
        lam_opt = lam_opt/100
        err_lam = err_lam/100
        lam_opt_day = lam_opt/0.0193088
        err_lam_day = err_lam/0.0193088
        
        lam_opt_day_g = lam_opt_day / ((15 * 15 * 670) * (10**-4)**3 * 2.33)
        err_lam_day_g = err_lam_day/ ((15 * 15 * 670) * (10**-4)**3 * 2.33)
        
 
        print(f"Poisson-Gauss fit:")
        print(f"  A      = {A_opt:.3f} ± {err_A:.3f}")
        print(f"  lambda (e/pix/img)= {lam_opt:.3e} ± {err_lam:.3e}")
        print(f"  lambda (e/pix/day)= {lam_opt_day:.3e} ± {err_lam_day:.3e}")
        print(f"  lambda (e/g/day)= {lam_opt_day_g:.3e} ± {err_lam_day_g:.3e}")
        print(f"  sigma  = {sigma_opt:.3f} ± {err_sigma:.3f}")
        
        # Evaluate model using best-fit parameters
        model_vals = poisson_gauss_conv(bin_centers, *popt, kmax=5)

        # Parameters values
        A_opt, lam_opt, sigma_opt = popt
        err_A = np.sqrt(pcov[0,0])
        err_lam = np.sqrt(pcov[1,1])
        err_sigma = np.sqrt(pcov[2,2])


       # Plot histogram and fitted curve
        plt.figure(figsize=(12,10))
        plt.step(bin_centers, counts, where='mid', color='black')
        plt.plot(bin_centers, model_vals, color='red', linewidth=2.5, label='Poisson-Gauss Fit')
        plt.yscale('log')
        plt.xlabel(r"charge (e$^{-}$)")
        plt.ylabel("counts")
        plt.xlim(-2.5, 3.5)
        plt.ylim(0.5, 1e7)
        plt.xticks()
        plt.yticks()
        ax = plt.gca()             
        ax.minorticks_on()         
        set_axis_labels_aligned(ax) 
        plt.grid(False)

        
        # Annotated box with fit results
        textstr = (
        f"{'Norm':<10}{A_opt:>8.0f} ± {err_A:<4.0f}\n"
        f"{'σ [e⁻]':<10}{sigma_opt:>8.3f} ± {err_sigma:<4.3f}\n"
        f"λ [e⁻/bin/img] ({lam_opt*1e4:>6.3f} ± {err_lam*1e4:<6.3f})$10^{{-4}}$"
        )

        # Compute coordinates for placing text box
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_pos = xlim[1] + 0.15 * (xlim[1] - xlim[0])
        y_pos = ylim[1] + 0.25 * (ylim[1] - ylim[0])

        plt.text(x_pos, y_pos, textstr,
                 fontsize=22, ha='right', va='top',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black',linewidth=2))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply all masks and show results")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help="Archivos FITS a procesar.")
    args = parser.parse_args()
    main(args)
