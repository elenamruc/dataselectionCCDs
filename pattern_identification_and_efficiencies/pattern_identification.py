#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""

import numpy as np
from numba import njit, prange, types
from math import erf, sqrt, log
from astropy.io import fits
import argparse
from numba.typed import List, Dict
from itertools import product
from collections import defaultdict


############################################
# 1) Load data and generate base mask
############################################


# List of mask extensions to apply
ALL_MASK_EXTS = [
    'CLSMASK', 'MASK_CROSSTALK', 'MASK_HOTCOLUMNS',
    'MASK_OVERDENSITY', 'MASK_CTI', 'MASK_CORRELATED_NOISE',
    'MASK_MULTIPLICITY', 'MASK_ISOLATED_COLUMNS'
]

def load_mask_if_exists(hdul, ext_name):
    """
    Attempts to load the FITS extension named `ext_name` as a boolean mask (True = masked).
    Returns None if the extension is not found.
    """
    if ext_name in [hdu.name for hdu in hdul]:
        mask_data = hdul[ext_name].data
        return (mask_data != 0)  # Convert 0/1 to boolean
    return None

def combine_all_masks_and_apply(fits_path):
    """
    Loads the calibrated FITS image and applies all available masks,
    setting NaN where the mask indicates (True).
    """
    
    with fits.open(fits_path) as hdul:
        pedsub_data = hdul['CALIBRATED'].data.astype(np.float64).copy()
        shape_data = pedsub_data.shape

        final_mask = np.zeros(shape_data, dtype=bool)
        for ext in ALL_MASK_EXTS:
            this_mask = load_mask_if_exists(hdul, ext)
            if this_mask is not None:
                final_mask |= this_mask

    pedsub_data[final_mask] = np.nan
    return pedsub_data, final_mask


############################################
# 2) Creation of pattern combos
############################################
def make_pattern_order(n):
    """
    Generates all combinations of `n` integers in [1..3] whose sum is <= 5.
    Returns them grouped by total sum, in increasing order.
    """
    grouped_combos = {}
    for c in product(range(1, 4), repeat=n):
        if sum(c) <= 5:
            s_ = sum(c)
            if s_ not in grouped_combos:
                grouped_combos[s_] = []
            grouped_combos[s_].append(np.array(c, dtype=np.int64))
    return [grouped_combos[key] for key in sorted(grouped_combos.keys())]
    

def pythonlist_to_numba(pylist_of_lists):
    """
    Converts a list of Python lists to a numba.typed.List of typed Lists,
    enabling fast numba-compatible iteration.
    """
    nb_list = List()
    for sublist in pylist_of_lists:
        sub_nb_list = List()
        for tup in sublist:
            sub_nb_list.append(tup)
        nb_list.append(sub_nb_list)
    return nb_list

############################################
# 3) Calculate CDF values
############################################

@njit
def cdf_value(q, m, sigma=0.15):
    """
    Evaluates the CDF value assuming a Gaussian model centered at `m`.
    Clamped to avoid exact 0 or 1 values.
    """
    if np.isnan(q):
        return 1e-20
    z = (q - m) / (sigma * sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    return max(1e-20, min(cdf, 1.0 - 1e-20))


@njit
def precompute_neglog_cdf_matrices(data_matrix, sigma=0.15):
    """
    Precomputes:
    - neglogcdf_matrix[i, j, m] = -log(CDF(data[i,j], m, sigma)) for m in 0..3
    - neglog1mcdf0_matrix[i, j] = -log(CDF(data[i,j], 1, sigma))
    Used for efficient pattern scoring and isolation checks.
    """
    nrows, ncols = data_matrix.shape
    neglogcdf_matrix = np.empty((nrows, ncols, 4), dtype=np.float64)
    neglog1mcdf0_matrix = np.empty((nrows, ncols), dtype=np.float64)

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(data_matrix[i,j]):
                for m in range(4):
                    neglogcdf_matrix[i,j,m] = 1e10
                neglog1mcdf0_matrix[i,j] = 1e10
            else:
                for m in range(4):
                    val_cdf = cdf_value(data_matrix[i,j], m, sigma)
                    neglogcdf_matrix[i,j,m] = -log(val_cdf)

                val_cdf0 = cdf_value(data_matrix[i,j], 1, sigma)
                neglog1mcdf0_matrix[i,j] = -log(val_cdf0)

    return neglogcdf_matrix, neglog1mcdf0_matrix

@njit
def cascade_method(coords, combos_nb, neglogcdf_matrix, threshold):
    """
    Evaluates all candidate pattern combinations and selects the one
    with the lowest total score (sum of -log(CDF) values).
    It stops early if no group contains any valid combination below the threshold.
    """
    n = len(coords)
    best_combo = np.full(n, -1, dtype=np.int64)   # Initialize best combo with -1 
    best_score = 1e12                             # Start with a very high score

    for combo_group in combos_nb:
        at_least_one_valid = False
        group_best_combo = None    # Best combination within this group
        group_best_score = 1e12    # Best score within this group

        for combo in combo_group:
            score = 0.0
            for i_cord in range(n):
                r, c = coords[i_cord]
                m = combo[i_cord]
                score += neglogcdf_matrix[r, c, m]
              
                # Skip evaluation if partial score already exceeds threshold
                if score > threshold:
                    break 
                    
            # Select the best combination within the group
            if score <= threshold:
                if score < group_best_score:
                    group_best_combo = combo.copy()
                    group_best_score = score
                at_least_one_valid = True

        # If the group has at least one valid combination, update best result
        if at_least_one_valid:
            best_combo[:] = group_best_combo 
            best_score = group_best_score

        else:
            break  # Stop evaluating if the group has no valid combinations

    return best_combo, best_score
    
    
@njit
def is_isolated(coords, neglog1mcdf0_matrix, data_matrix, threshold):
    """
    Checks whether the given pixel coordinates are isolated:
    i.e., none of their immediate neighbors are charged or masked.
    """
    nrows, ncols = neglog1mcdf0_matrix.shape
    for (ix, jx) in coords:
        for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0),(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = ix + di, jx + dj
            if 0 <= rr < nrows and 0 <= cc < ncols:
                if (rr, cc) not in coords:
                    if np.isnan(data_matrix[rr, cc]): 
                        return False
                    if neglog1mcdf0_matrix[rr, cc] < threshold:
                        return False
    return True

@njit
def detect_patterns(data_matrix, combos1_nb, combos2_nb, combos3_nb,
                    neglogcdf_matrix, neglog1mcdf0_matrix, threshold1=3.5, threshold2=4.0, threshold3=5.5):
    """
    Detects patterns in the data matrix while avoiding duplicate pixel usage.
    For each pixel and direction, it keeps only the last valid pattern that passes the selection.
    """
    nrows, ncols = data_matrix.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    dir_names = ["Row", "Col", "diag_r", "diag_l"]

    # Boolean matrix (uint8) to track used pixels and prevent overlaps
    used_pixels = np.zeros((nrows, ncols), dtype=np.uint8)
    patterns = []

    for i in range(nrows):
        for j in range(ncols):
            if used_pixels[i, j] == 1:
                continue  # Skip if pixel is already used by a previous pattern

            # Try all directions
            for d_idx, (di, dj) in enumerate(directions):
                # Temporary variable to store the last valid pattern found
                last_valid_pattern = (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), -1.0, "", np.empty(0, dtype=np.float64))

                # Iterate over pattern lengths: 1, 2, and 3
                for length in (1, 2, 3):
                    coords = []
                    out_of_bounds = False

                    # Build coordinates of the candidate pattern
                    for k in range(length):
                        xx = i + k * di
                        yy = j + k * dj
                        if xx < 0 or xx >= nrows or yy < 0 or yy >= ncols:
                            out_of_bounds = True
                            break
                        coords.append((xx, yy))

                    if out_of_bounds:
                        continue

                    # Check for conflicts with already used pixels
                    conflict = False
                    for (r, c) in coords:
                        if used_pixels[r, c] == 1:
                            conflict = True
                            break
                    if conflict:
                        continue

                    # Choose appropriate threshold and combinations
                    if length == 1:
                        thr = threshold1
                        combos_nb_ = combos1_nb
                    elif length == 2:
                        thr = threshold2
                        combos_nb_ = combos2_nb
                    else:
                        thr = threshold3
                        combos_nb_ = combos3_nb

                    # Apply cascade_method to evaluate the pattern
                    best_combo, best_score = cascade_method(
                        coords, combos_nb_, neglogcdf_matrix, thr
                    )
                    
                    
                    # If a valid pattern is found and it's isolated, store it temporarily
                    if best_combo[0] != -1 and is_isolated(coords, neglog1mcdf0_matrix, data_matrix, threshold1):
                        last_valid_pattern = (np.array(coords, dtype=np.int64), best_combo.copy(), best_score, dir_names[d_idx],
                                              np.array([data_matrix[r, c] for (r, c) in coords]))


                # After testing all lengths in this direction, keep the last valid one
                if last_valid_pattern[0].shape[0] > 0:  # Comprobamos si se detectó algo
                    patterns.append(last_valid_pattern)
                    # Mark pixels as used
                    for (r, c) in last_valid_pattern[0]:  
                        used_pixels[r, c] = 1

    return patterns
        
def main():
    parser = argparse.ArgumentParser(description="Optimized detection of charge patterns in FITS files.")
    parser.add_argument('-f', '--files', nargs='+', required=True, help="Paths to FITS files.")
    args = parser.parse_args()
    
    # Generate combinations for 1-, 2- and 3-pixel patterns
    combos1 = pythonlist_to_numba(make_pattern_order(1))
    combos2 = pythonlist_to_numba(make_pattern_order(2))
    combos3 = pythonlist_to_numba(make_pattern_order(3)) 
    
    # Calibration parameters for each CCD quadrant (PA08 103)
    # Uncomment the block for the dataset being processed
    
    # 14/10/2024
#    lambdas_103 = [0.000282, 0.0002266, 0.0002592, 0.000312]
#    sigma_e_103 = [0.155, 0.159, 0.162, 0.167]
    
    # 28/10/2024
#    lambdas_103 = [0.0002806, 0.0002269, 0.0002708, 0.0003117]
#    sigma_e_103 = [0.153, 0.158, 0.161, 0.164]
    
    # 19/11/2024
    lambdas_103 = [0.000291, 0.0002322, 0.0002892, 0.0003229]
    sigma_e_103 = [0.153, 0.157, 0.160, 0.164]

    # 17/12/2024
#    lambdas_103 = [0.000289, 0.0002194, 0.0002745, 0.0003217]
#    sigma_e_103 = [0.153, 0.155, 0.158, 0.164]

    
    files = args.files
    import pandas as pd
    detected_11_patterns = {f: [] for f in files}
    
    for i, f in enumerate(args.files):
        print("\n" + "="*60)
        print(f"Processing file #{i+1}: {f}")
        
        if i < len(lambdas_103):
            current_lambda = lambdas_103[i]
            current_sigma_e = sigma_e_103[i]
       
        # Load and mask data
        pedsub_data, masks = combine_all_masks_and_apply(f)
        pedsub_data = pedsub_data.astype(np.float64)    
     
         # Precompute -log(CDF) matrices for all values m = 0..3
        neglogcdf_matrix, neglog1mcdf0_matrix = precompute_neglog_cdf_matrices(pedsub_data, sigma=current_sigma_e)
        
         
        # Detect patterns using the cascade algorithm
        found = detect_patterns(pedsub_data,
                                combos1, combos2, combos3,
                                neglogcdf_matrix, neglog1mcdf0_matrix,
                                threshold1=3.5, threshold2=4.0, threshold3=5.5)
        
        print(f"Total detected patterns: {len(found)}")

        # Count total frequency of detected patterns (ignores direction)
        from collections import Counter
        c = Counter([str(p[1]) for p in found])
        print("Pattern histogram:", c)

        # Count frequency of patterns grouped by direction
        c_by_direction = Counter([(str(p[1]), p[3]) for p in found])
        print("\nPattern histogram by direction:")
        for (pattern, direction), count in c_by_direction.items():
            print(f"  - Pattern {pattern} in direction {direction}: {count} times")
                                 
        # Show details of each pattern found (excluding isolated 1-pixel patterns)
        print("\nDetected patterns:")
        for pattern in found:
            coords, pattern_type, score, direction, values = pattern
            if not np.array_equal(pattern_type, np.array([1])):
                print(f"  - coords={coords}, type={pattern_type}, "
                  f"score={score:.2f}, direction={direction}, values={values}")

     
if __name__ == "__main__":
    main()
