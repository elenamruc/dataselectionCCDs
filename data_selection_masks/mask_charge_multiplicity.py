#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""

import math, numpy as np
from astropy.io import fits
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import poisson, norm
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats

from utils_plot import set_plot_style, set_axis_labels_aligned
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

def build_partial_masks(fits_file):
    """
    Reads a series of pre-existing extensions (CLSMASK, MASK_CROSSTALK, etc.)
    and builds a cumulative mask by combining them using logical OR.

    Returns:
        final_mask (bool array): True where pixels are masked.
    """
    
    ordered_exts = [
        'CLSMASK',
        'MASK_CROSSTALK',
        'MASK_HOTCOLUMNS',
        'MASK_OVERDENSITY',
        'MASK_CTI'
    ]
    
    with fits.open(fits_file) as hdul:
        shape_data = hdul['CALIBRATED'].data.shape
        partial_mask = np.zeros(shape_data, dtype=bool)

        for ext_name in ordered_exts:
            this_mask = load_mask_if_exists(hdul, ext_name)
            if this_mask is not None:
                partial_mask |= this_mask # Logical OR

    return partial_mask

############################################
# 2) Robust and classical statistics
############################################

def median_absolute_deviation(arr):
    """
    Computes MAD as median(|arr - median(arr)|).
    Ignores NaNs.
    """
    
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def poisson_gauss_conv(x, A, lam, sigma, kmax=10):
    """
    Convolution of a Poisson(λ) with a Gaussian(σ).
    """
    
    x   = np.asarray(x, float)
    pk  = [(lam**k * math.exp(-lam) / math.factorial(k)) for k in range(kmax+1)]
    out = np.zeros_like(x)
    for k in range(kmax+1):
        g = np.exp(-(x-k)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
        out += pk[k]*g
    return A*out

def fit_poisson_gauss(charges, nbins=100, rng=(-2, 3), kmax=10):
    """
    Fit a histogram of charges to a Poisson convolved with Gaussian.

    Returns:
        popt: Fitted parameters (A, lambda, sigma)
        pcov: Covariance matrix
        centres: Bin centers
        h: Histogram values
    """
    
    h, e = np.histogram(charges, bins=nbins, range=rng)
    centres = 0.5*(e[1:] + e[:-1])

    p0 = (h.max(), 1.0, 0.20)  # Initial guess: A, λ, σ
    popt, pcov = curve_fit(
        lambda x, A, lam, s: poisson_gauss_conv(x, A, lam, s, kmax),
        centres, h, p0=p0,
        sigma=np.sqrt(h+1), absolute_sigma=True
    )
    return popt, pcov, centres, h

    
############################################
# 3) Normalized row/column multiplicities and masking
############################################

def compute_row_multiplicities(data, unmasked_mask, electron):
    """
    For each row i, compute:
        multiplicity[i] = (# of pixels with data > electron and unmasked)
                          divided by (# of unmasked pixels in that row).

    Returns:
        1D array with the multiplicity per row.
    """
    
    nrows, ncols = data.shape
    row_mult = np.zeros(nrows, dtype=float)

    for i in range(nrows):
        row_unmasked = unmasked_mask[i, :]
        total_unmasked_in_row = np.sum(row_unmasked)
        if total_unmasked_in_row == 0:
            row_mult[i] = np.nan  # No valid pixels
            continue

        # Pixels >1 e- and without mask
        row_data = data[i, :]
        n_gt1 = np.sum((row_data > electron) & row_unmasked)

        row_mult[i] = n_gt1
    return row_mult

def compute_col_multiplicities(data, unmasked_mask, electron):
    """
    Same as compute_row_multiplicities, but for each column j:
        multiplicity[j] = (# of pixels > electron and unmasked) /
                          (# of unmasked pixels in column j)
    """
    
    nrows, ncols = data.shape
    col_mult = np.zeros(ncols, dtype=float)

    for j in range(ncols):
        col_unmasked = unmasked_mask[:, j]
        total_unmasked_in_col = np.sum(col_unmasked)
        if total_unmasked_in_col == 0:
            col_mult[j] = np.nan
            continue

        col_data = data[:, j]
        n_gt1 = np.sum((col_data > electron) & col_unmasked)

        col_mult[j] = n_gt1

    return col_mult

def mask_bad_rows_and_cols_from_multiplicity(row_mult, col_mult, row_threshold, col_threshold):
    """
    Given row and column multiplicities and their thresholds,
    build a 2D boolean mask that flags entire rows or columns
    whose values exceed the threshold.

    Returns:
        mask (2D bool array): True for masked rows/columns.
    """
    
    nrows = row_mult.size
    ncols = col_mult.size

    bad_rows = np.where(row_mult > row_threshold)[0]
    bad_cols = np.where(col_mult > col_threshold)[0]

    print(f"[INFO] Bad rows (multiplicity > {row_threshold:.5f}): {bad_rows}")
    print(f"[INFO] Bad columns (multiplicity > {col_threshold:.5f}): {bad_cols}")

    rowcol_mask = np.zeros((nrows, ncols), dtype=bool)
    for br in bad_rows:
        rowcol_mask[br, :] = True
    for bc in bad_cols:
        rowcol_mask[:, bc] = True

    return rowcol_mask

############################################
# 4) Plot multiplicities with thresholds
############################################

def plot_row_multiplicities(row_mult, thr_per, name):
    """
    Plot row multiplicity on Y vs. row index on X.

    Parameters:
        row_mult (array): Row-wise multiplicities.
        thr_per (float): Custom threshold to draw (e.g. percentile).
        name (str): Label for the threshold.
    """
    
    x = np.arange(len(row_mult))    
    y = row_mult.copy()

    valid = ~np.isnan(y)
    y_valid = y[valid]

    # Robust statistics
    med = np.median(y_valid)
    mad_val = median_absolute_deviation(y_valid)
    thr_robust = med + 3.0 * mad_val

    # Classical statistics
    mu = np.mean(y_valid)
    sigma = np.std(y_valid)
    thr_classic = mu + 3.0 * sigma
    
    # IQR-based threshold
    q1 = np.percentile(y_valid, 25)
    q3 = np.percentile(y_valid, 75)
    iqr = q3 - q1
    cutoff = q3 + 3 * iqr

    # Plot
    plt.figure(figsize=(14, 8))
    plt.xticks()
    plt.yticks()
    plt.plot(x, y, '-', color='black')
    

    #plt.axhline(med, linestyle='-', label=f"Median={med:.5f}")
    #plt.axhline(thr_robust, linestyle='--', label=f"Median+3*MAD={thr_robust:.5f}")
    #plt.axhline(thr_classic, linestyle=':', label=f"Media+3σ={thr_classic:.5f}")
    #plt.axhline(thr_classic, linestyle='-.', label=f"IQR={cutoff:.5f}")
    plt.axhline(thr_per, linestyle='-', label=f"{name}", color='red', linewidth=2.5)


    plt.xlabel("row")
    plt.ylabel("counts")
    plt.legend()
    ax = plt.gca()             
    ax.minorticks_on()         
    set_axis_labels_aligned(ax)     
    plt.tight_layout()
    plt.show()

def plot_col_multiplicities(col_mult, thr_per, name):
    """
    Plot col multiplicity on Y vs. col index on X.

    Parameters:
        col_mult (array): Col-wise multiplicities.
        thr_per (float): Custom threshold to draw (e.g. percentile).
        name (str): Label for the threshold.
    """
    
    x = np.arange(len(col_mult))
    y = col_mult.copy()

    valid = ~np.isnan(y)
    y_valid = y[valid]

    # Robust statistics
    med = np.median(y_valid)
    mad_val = median_absolute_deviation(y_valid)
    thr_robust = med + 3.0 * mad_val

    # Classical statistics
    mu = np.mean(y_valid)
    sigma = np.std(y_valid)
    thr_classic = mu + 3.0 * sigma
    
    # IQR-based threshold
    q1 = np.percentile(y_valid, 25)
    q3 = np.percentile(y_valid, 75)
    iqr = q3 - q1
    cutoff = q3 + 3 * iqr

    # Plot
    plt.figure(figsize=(14, 8))
    plt.xticks()
    plt.yticks()
    plt.plot(x, y, '-', color='black')
    
    #plt.axhline(med, linestyle='-', label=f"Median={med:.5f}")
    #plt.axhline(thr_robust, linestyle='--', label=f"Median+3*MAD={thr_robust:.5f}")
    #plt.axhline(thr_classic, linestyle=':', label=f"Media+3σ={thr_classic:.5f}")
    #plt.axhline(thr_classic, linestyle='-.', label=f"IQR={cutoff:.5f}")
    plt.axhline(thr_per, linestyle='-', label=f"{name}", color='red', linewidth=2.5)


    plt.xlabel("column")
    plt.ylabel("counts")
    plt.legend()
    ax = plt.gca()             
    ax.minorticks_on()         
    set_axis_labels_aligned(ax)     
    plt.tight_layout()
    plt.show()
    
def mask_rows_cols_multiple(data, unmasked_mask, more_electron, mask_2e=1):
    """
    Masks rows/columns that contain more than 'mask_2e' pixels
    with charge above the defined threshold (e.g., ~2e−).

    Returns:
        new_mask (2D bool array): True for masked pixels.
    """
    
    # Define charge condition (e.g., >1.7e−)
    mask_more2e = (data>=more_electron) & unmasked_mask

    nrows, ncols = data.shape

    # Count how many suspicious pixels per row
    row_count_2e = np.sum(mask_more2e, axis=1)

    # Identify rows exceeding the threshold
    row_suspicious = (row_count_2e > mask_2e)
    n_bad_rows = np.sum(row_suspicious)

    # Count per column
    col_count_2e = np.sum(mask_more2e, axis=0)

    col_suspicious = (col_count_2e > mask_2e)
    n_bad_cols = np.sum(col_suspicious)

    print(f"[INFO] Suspicious rows: {n_bad_rows} / {nrows}")
    print(f"[INFO] Suspicious columns: {n_bad_cols} / {ncols}")

    # Build combined row/column mask
    new_mask = np.zeros((nrows, ncols), dtype=bool)
    for i in range(nrows):
        if row_suspicious[i]:
            new_mask[i,:] = True
    for j in range(ncols):
        if col_suspicious[j]:
            new_mask[:,j] = True

    return new_mask


############################################
# 3) Gaussian fit + thresholding by deviation
############################################

def gaussian_func(x, A, mu, sigma):
    """
    1D Gaussian function.
    A = amplitude (peak height),
    mu = mean,
    sigma = standard deviation.
    """
    
    return A * np.exp(-0.5 * ((x - mu)/sigma)**2)


def mask_rows_cols_by_cutoff(row_mult, col_mult, row_cutoff, col_cutoff):
    """
    Builds a mask that flags rows where row_mult > row_cutoff
    and columns where col_mult > col_cutoff.

    Returns:
        rowcol_mask (2D bool array): True for masked rows/columns.
    """
    
    nrows = len(row_mult)
    ncols = len(col_mult)
    rowcol_mask = np.zeros((nrows, ncols), dtype=bool)

    if row_cutoff is not None:
        bad_rows = np.where(row_mult > row_cutoff)[0]
        for r in bad_rows:
            rowcol_mask[r, :] = True

    if col_cutoff is not None:
        bad_cols = np.where(col_mult > col_cutoff)[0]
        for c in bad_cols:
            rowcol_mask[:, c] = True

    return rowcol_mask
    
def outliers_percentile(values, p=0.9):
    """
    Returns indices of elements in 'values' (1D) that are above the p-th percentile.

    Returns:
        bad_indices (array): Indices of outliers.
        cutoff (float): Value corresponding to the percentile threshold.
    """
    
    vals_clean = values[~np.isnan(values)]
    cutoff = np.percentile(vals_clean, p)
    bad_indices = np.where(values > cutoff)[0]
    
    return bad_indices, cutoff
    

############################################
# 4) Hit probability and λ estimation from threshold exceedance
############################################

def p_hit(lam, threshold, sigma_read, k_max=50):
    """
    Computes P(pixel > threshold) for a Poisson(λ) signal
    plus Gaussian noise (σ = sigma_read).

    Uses convolution:
        P = sum_k [ Poisson(λ = lam) * P(Gauss > threshold - k) ]

    Returns:
        float: Probability of exceeding threshold.
    """
    
    if lam <= 0:
        # Only noise, no Poisson signal
        return norm.sf(threshold, loc=0.0, scale=sigma_read)
    ks = np.arange(0, k_max+1)
    # pmf Poisson
    pk = poisson.pmf(ks, lam)
    # P(ε > threshold - k)
    surv = norm.sf(threshold - ks, loc=0.0, scale=sigma_read)
    return np.sum(pk * surv)

def estimate_lambda_from_fraction(p_obs, threshold, sigma_read,
                                  k_max=50, lam_min=0.0, lam_max=1.0):
    """
    Estimates the value of λ such that:
        p_hit(λ) = p_obs

    Returns:
        float or None:
            - Estimated λ if solution found.
            - 0.0 if p_obs ≤ p_hit(0).
            - None if p_obs > p_hit(lam_max) or no solution found.
    """
    
    p0 = p_hit(0.0, threshold, sigma_read, k_max=k_max)
    if p_obs <= p0:
        return 0.0
    p_max = p_hit(lam_max, threshold, sigma_read, k_max=k_max)
    if p_obs > p_max:
        return None

    def f(lam):
        return p_hit(lam, threshold, sigma_read, k_max=k_max) - p_obs
    try:
        lam_est = brentq(f, lam_min, lam_max)
        return lam_est
    except ValueError:
        return None


# ------------------------------------------------------------------
#  Monte-Carlo that honours the live-mask *and* the λ(x) profile
# ------------------------------------------------------------------
def mc_multiplicity_with_profile(live_mask,
                                 lambda_profile_col,          # 1-D (ncols)
                                 sigma_lambda_per_col=0.0,    # scalar or 1-D
                                 n_images=500,
                                 charge_cut=0.7,
                                 read_noise=0.15,
                                 axis=0,                      # 0→cols, 1→rows
                                 rng=None):
    """
    Return a   (n_images , n_cols)   or   (n_images , n_rows)  array
    with the multiplicity in each col / row for every simulated frame.

      live_mask            bool[rows,cols]  (True = pixel usable)
      lambda_profile_col   expected λ per *column* (e⁻/pix/exposure)
      sigma_lambda_per_col σ(λ) column-wise (same length or scalar)
    """
    if rng is None:
        rng = np.random.default_rng()

    nrows, ncols = live_mask.shape
    
    # Broadcast σλ
    if np.isscalar(sigma_lambda_per_col):
        sigma_lambda_per_col = np.full(ncols, sigma_lambda_per_col, float)

    valid = live_mask.any(axis=axis) 
    out = np.full((n_images, valid.size), np.nan, dtype=float)

    for i in range(n_images):
    
        # 1) sample one λ for every column (clip at 0)
        lam_cols = rng.normal(lambda_profile_col,0)
        lam_cols = np.clip(lam_cols, 0.0, None)

        # 2) build a full λ matrix (rows × cols)
        lam_mat = np.broadcast_to(lam_cols, (nrows, ncols))

        # 3) Poisson DC + read noise
        dark = rng.poisson(lam=lam_mat).astype(float)
        dark += rng.normal(0.0, read_noise, size=dark.shape)

        # 4) honour the live-mask
        dark[~live_mask] = -np.inf
        hits = (dark > charge_cut)

        # 5) count hits
        counts = hits.sum(axis=axis).astype(float)
        counts[~valid]  = np.nan
        out[i] = counts

    return out

# ------------------------------------------------------------------
#  Fit-and-compare routine (Gaussian fit + profile-MC)
# ------------------------------------------------------------------
def find_cutoff_from_gaussian_fit(values,                # 1-D multiplicities
                                  live_mask,             # bool 2-D
                                  thr_per,               # 99th-percentile line
                                  lambda_profile_col,    # λ(x) profile
                                  sigma_lambda_per_col,  # jitter
                                  n_mc=500,
                                  read_noise=0.15,
                                  charge_cut=0.7,
                                  axis=0, MC=False):
    """
    Plot the data histogram, Gaussian tail fit, *and* profile-aware MC.
    Returns the data→model deviation cutoff (or None).
    """
    # Histogram of the data
    vals     = values[~np.isnan(values)]
    max_val  = int(np.max(vals))
    bins     = np.arange(max_val + 2) - 0.5
    hist, be = np.histogram(vals, bins=bins)
    bc       = 0.5 * (be[1:] + be[:-1])
    
    cum = np.cumsum(hist)
    total = cum[-1]
    q60_bin  = bc[np.searchsorted(cum, 0.60 * total)] 
    fit_mask = (bc >= q60_bin) & (hist > 0)

    # Gaussian tail fit
    valid = hist > 0   
    if valid.sum() >= 5:          
        p0 = [hist[fit_mask].max(), bc[fit_mask].mean(), np.std(vals)]

        popt, _ = curve_fit(
        gaussian_func,
        bc[fit_mask], hist[fit_mask], 
        p0=p0,
        sigma=np.sqrt(hist[fit_mask]+1),   
        absolute_sigma=True
        )
        gauss_ok   = True
        x_fit = np.linspace(bins[0], bins[-1], 400)
        y_fit = gaussian_func(x_fit, *popt) 
        
        q60_bin = bc[np.searchsorted(cum, 0.60 * total)]
        mask_tail = x_fit >= q60_bin
    else:
        gauss_ok = False

    plt.figure(figsize=(12, 8))
    plt.bar(bc, hist, width=1.0, alpha=0.3, color='gray', align='center', label='Data bins')
    plt.errorbar(bc, hist, yerr=np.sqrt(hist), fmt='o', ms=7, label='Data (≥0.7 e⁻)', color="black")
    
    # Monte-Carlo with profile
    if MC:
        mc = mc_multiplicity_with_profile(live_mask,
                                      lambda_profile_col,
                                      sigma_lambda_per_col,
                                      n_images=n_mc,
                                      charge_cut=charge_cut,
                                      read_noise=read_noise,
                                      axis=axis).ravel()
        h_mc, _ = np.histogram(mc, bins=bins)
        h_mc    = h_mc.astype(float) 
        scale   = hist.sum() / h_mc.sum() if h_mc.sum() else 1.0
        h_mc   *= scale
        err_mc  = np.sqrt(h_mc)


        plt.errorbar(bc, h_mc, yerr=err_mc, fmt='s', mfc='royalblue', mec='royalblue', ms=7, capsize=2, lw=1.2, label=fr"MC DC")
        
    if gauss_ok:
        plt.plot(x_fit[mask_tail], y_fit[mask_tail], '-', color='red',
             label=f'Gauss fit', linewidth=2.5)
    thr_per = thr_per+0.5
    plt.axvline(thr_per, linestyle='-', label=f"99th Percentile", linewidth=2.5, color='orange')
    plt.xlabel("multiplicity")
    plt.ylabel("counts")  
    plt.ylim(bottom=0)
    plt.legend()
    plt.xticks()
    plt.yticks()
    set_axis_labels_aligned()
    plt.tight_layout()
    plt.show() 
        
    p90 = np.percentile(vals, 90)       
    cutoff_val = None
    for x, d, m in zip(bc, hist, y_fit):
        if x > popt[1] and d - m > 3*np.sqrt(d):
            if x >= p90 and x<=thr_per:                 
                cutoff_val = x
            break

    return cutoff_val


    
############################################
# 5) MAIN
############################################

def main(args):
    for fits_file in args.files:
        print(f"\n=== Procesando {fits_file} ===")

        # A) data + pre-existing masks
        data = load_data(fits_file)
        partial_mask = build_partial_masks(fits_file)  # True = masked
        unmasked_mask = ~partial_mask                  # True = available pixel

        # B) Compute row/column multiplicities
        row_mult = compute_row_multiplicities(data, unmasked_mask, args.charge1)
        col_mult = compute_col_multiplicities(data, unmasked_mask, args.charge1)

        
        bad_rows, row_cutoff_p = outliers_percentile(row_mult, p=99)
        bad_cols, col_cutoff_p = outliers_percentile(col_mult, p=99)
        
        # Fit Poisson × Gaussian to unmasked pixel values
        pix = data[unmasked_mask].ravel()
        pix = pix[~np.isnan(pix)]           # remove NaNs
        popt, pcov, xc, hc = fit_poisson_gauss(pix)
        A_fit, lam_fit, sigma_fit = popt 
        err_A,  err_lam, err_sigma = np.sqrt(np.diag(pcov)) 
        sigma_lambda = err_lam
        
        print(f"[INFO]   Poisson x Gauss fit →  λ = {lam_fit:.3e}  σ = {sigma_fit:.3f} e⁻")
        
        # Estimate per-column λ using inversion of p_hit
        nrows, ncols = data.shape
        threshold = args.charge1
        sigma_read = sigma_fit 
        lam_max = 1.0

        lambda_raw = np.zeros(ncols, dtype=float)
        lambda_low = np.zeros(ncols, dtype=float)
        lambda_high = np.zeros(ncols, dtype=float)

        for j in range(ncols):
            mask_col = unmasked_mask[:, j]
            n_j = np.sum(mask_col)
            if n_j <= 0:
                lambda_raw[j] = lambda_low[j] = lambda_high[j] = 0.0
                continue

            k_j = col_mult[j]
            p_obs = k_j / n_j

            # Estimate lambda_j
            lam_j = estimate_lambda_from_fraction(p_obs, threshold, sigma_read,
                                          k_max=50, lam_min=0.0, lam_max=lam_max)
            if lam_j is None:
                lambda_raw[j] = np.nan
            else:
                lambda_raw[j] = lam_j

          
            sigma_p = np.sqrt(p_obs*(1-p_obs)/n_j) if n_j>0 else 0.0
            p_low  = max(0.0, p_obs - sigma_p)
            p_high = min(1.0, p_obs + sigma_p)
            lam_low = estimate_lambda_from_fraction(p_low, threshold, sigma_read,
                                            k_max=50, lam_min=0.0, lam_max=lam_max)
            lam_high = estimate_lambda_from_fraction(p_high, threshold, sigma_read,
                                             k_max=50, lam_min=0.0, lam_max=lam_max)
            lambda_low[j] = lam_low if lam_low is not None else np.nan
            lambda_high[j] = lam_high if lam_high is not None else np.nan

        # Interpolate missing values
        mask_nan = np.isnan(lambda_raw)
        lambda_raw[mask_nan] = np.interp(np.flatnonzero(mask_nan),
                                   np.flatnonzero(~mask_nan),
                                   lambda_raw[~mask_nan])
        lambda_low[np.isnan(lambda_low)] = 0.0
        lambda_high[np.isnan(lambda_high)] = lambda_raw[np.isnan(lambda_high)]

        # Smooth the profile
        window = 201
        if window % 2 == 0:
            window += 1
        if window >= ncols:
            window = ncols-1 if (ncols-1)%2==1 else ncols-2
        if window >= 3:
            try:
                lambda_smooth = savgol_filter(lambda_raw, window_length=window,
                                      polyorder=2, mode='nearest')
            except Exception:
                lambda_smooth = lambda_raw.copy()
        else:
            lambda_smooth = lambda_raw.copy()
            
        lambda_smooth = gaussian_filter1d(lambda_smooth, sigma=50, mode='nearest')
        lambda_smooth = np.clip(lambda_smooth, 0.0, None)

        # Fit 2nd-degree polynomial to smoothed profile
        x = np.arange(ncols)        
        xm = x - (ncols-1)/2
        good = ~np.isnan(lambda_smooth)
        if np.sum(good) >= 3:
            coef2 = np.polyfit(xm[good], lambda_smooth[good], deg=2)
            lambda_poly = np.clip(np.polyval(coef2, xm), 0.0, None)
        else:
            lambda_poly = lambda_smooth.copy()

       
        mean_lambda = np.nanmean(lambda_raw)
        std_lambda  = np.nanstd(lambda_raw)
        print(f"Mean λ over columns: {mean_lambda:.3e}, σ across columns: {std_lambda:.3e}")

        # Graficar perfil de lambda con barras de error ± incertidumbre:
        #fig, ax = plt.subplots(figsize=(12,4))
        #ax.errorbar(x, lambda_raw, yerr=[lambda_raw - lambda_low, lambda_high - lambda_raw],
        #    fmt='.', ms=3, alpha=0.5, label='λ bruto (inversión p_hit)')
        #ax.plot(x, lambda_smooth, '-', lw=1.5, label='λ suavizada')
        #ax.plot(x, lambda_poly, '-', lw=2, label='λ polinomio deg2')
        # Líneas de media ± std
        #ax.axhline(mean_lambda, color='k', linestyle='--', label='media λ')
        #ax.axhline(mean_lambda + std_lambda, color='gray', linestyle=':', label='media ± 1σ')
        #ax.axhline(mean_lambda - std_lambda, color='gray', linestyle=':')
        #ax.set_xlabel("Índice de columna")
        #ax.set_ylabel("λ estimada [e⁻/pix]")
        #ax.set_title("Perfil de corriente oscura λ por columna")
        #ax.legend(ncol=2, fontsize='small')
        #plt.tight_layout()
        #plt.show()
        
        # Extract dark current estimate from FITS headers
        with fits.open(fits_file) as hdul:
            lambda_dark_adu = hdul['CALIBRATED'].header.get('MEDCC', 3.2e-4)
            lambda_dark_adu_e = hdul['CALIBRATED'].header.get('MEGAINC', 1.5)
            sigma = hdul['CALIBRATED'].header.get('PSSIGA', 0.15)
            lambda_dark = lambda_dark_adu/lambda_dark_adu_e
       
        lambda_prof = lambda_poly

        # C) Compute Gaussian-based cutoffs
        col_cutoff = find_cutoff_from_gaussian_fit(
            col_mult, live_mask=unmasked_mask, thr_per=col_cutoff_p,
            lambda_profile_col=lambda_prof, sigma_lambda_per_col=std_lambda,
            n_mc=500, read_noise=sigma_fit, axis=0, MC=False)
      
        row_cutoff = find_cutoff_from_gaussian_fit(
            row_mult, live_mask=unmasked_mask, thr_per=row_cutoff_p,
            lambda_profile_col=lambda_prof, sigma_lambda_per_col=std_lambda,
            n_mc=500, read_noise=sigma_fit, axis=1, MC=False)
            
        # D) Plot and apply row/col masking based on cutoff
        if row_cutoff is not None and row_cutoff !=0.0:
            print(f"[INFO] Computed row_cutoff = {row_cutoff}")
            plot_row_multiplicities(row_mult, row_cutoff, "Gauss fit tail")  
            
        elif row_cutoff == row_cutoff_p:
            print(f"[INFO] Using percentile-based row_cutoff = {row_cutoff}")
            plot_row_multiplicities(row_mult, row_cutoff, "Gauss fit tail")
        else: 
            row_cutoff = row_cutoff_p
            print(f"[INFO] Using percentile-based row_cutoff = {row_cutoff}")
            plot_row_multiplicities(row_mult, row_cutoff, "99th Percentile")        


        if col_cutoff is not None and col_cutoff != 0.0:
            print(f"[INFO] Computed col_cutoff = {col_cutoff}")
            plot_col_multiplicities(col_mult, col_cutoff, "Gauss fit tail")
        elif col_cutoff == col_cutoff_p:
            print(f"[INFO] Using percentile-based col_cutoff = {col_cutoff}")
            plot_col_multiplicities(col_mult, col_cutoff, "Gauss fit tail")
        else:
            col_cutoff = col_cutoff_p
            print(f"[INFO] Using percentile-based col_cutoff = {col_cutoff}")
            plot_col_multiplicities(col_mult, col_cutoff, "99th Percentile")
        

        # D) Maskrows/columns above threshold
        rowcol_mask = mask_rows_cols_by_cutoff(row_mult, col_mult, row_cutoff, col_cutoff)
        rowscols_mask_multiple = mask_rows_cols_multiple(data, unmasked_mask, more_electron=1.7, mask_2e=1)
        

        # E) Combine masks and save result
        final_new_mask = partial_mask | rowcol_mask | rowscols_mask_multiple
        masked_pixels = np.sum(final_new_mask)
        total_pixels = final_new_mask.size
        print(f"[INFO] Total masked pixels: {masked_pixels} / {total_pixels} "
              f"({100. * masked_pixels / total_pixels:.2f}%)")
         
        multiplicity_mask = rowcol_mask | rowscols_mask_multiple
        masked_pixels = np.sum(multiplicity_mask)
        total_pixels = multiplicity_mask.size
        print(f"[INFO] Masked pixels (multiplicity only): {masked_pixels} / {total_pixels} "
              f"({100. * masked_pixels / total_pixels:.2f}%)")
              
        with fits.open(fits_file, mode='update') as hdul:
            ext_names = [hdu.name for hdu in hdul]
            if 'MASK_MULTIPLICITY' in ext_names:
                idx = ext_names.index('MASK_MULTIPLICITY')
                del hdul[idx]
       
            mask_hdu = fits.ImageHDU(data=multiplicity_mask.astype(np.uint8), name='MASK_MULTIPLICITY')
            hdul.append(mask_hdu)
            hdul.flush()

        print("[INFO] MASK_MULTIPLICITY saved in FITS file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Computes normalized multiplicity (pixels > 1e−) per row and column, "
        "and masks full rows/columns exceeding the threshold (median + 3×MAD or Gaussian-based)."
    ))
    parser.add_argument('-f', '--files', nargs='+', required=True, help="List of FITS files to process.")
    parser.add_argument('-q', '--charge1', type=float, required=True,
                        help="Charge threshold for a single electron (in e−).")
    args = parser.parse_args()
    main(args)
