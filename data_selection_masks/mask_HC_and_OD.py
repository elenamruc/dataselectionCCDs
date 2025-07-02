#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""
# Simulation to detect hot columns in a CCD image
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from utils_plot import set_plot_style, set_axis_labels_aligned
from scipy.signal import savgol_filter


def combine_clsmask(list_of_files):
    """
    Reads the CLSMASK extension from each of the provided files and constructs
    a single combined mask, where a pixel is 1 if it is 1 in any of the files
    (CLSMASK == 1), and 0 otherwise.

    Returns:
        numpy.ndarray: Combined mask array.
    """
    # Read the first file to initialize
    with fits.open(list_of_files[0]) as f0:
        clsmask_data_0 = f0['CLSMASK'].data.astype(int)

    # Create the combined mask (initially using the first file)
    combined_clsmask = (clsmask_data_0 == 1).astype(int)

    # Perform logical OR for the remaining files
    for filename in list_of_files[1:]:
        with fits.open(filename) as ff:
            clsmask_data = ff['CLSMASK'].data.astype(int)
           
        # Set to 1 if already 1 in the combined mask or in the current file
        combined_clsmask = np.where((combined_clsmask == 1) | (clsmask_data == 1), 1, 0)

    return combined_clsmask 

def extract_data(combined_cls, fits_path, charge_range):
    """
    Processes the FITS file and extracts necessary data.

    Returns:
        tuple of numpy.ndarray: 
        - non_masked_data_all: Non-masked pixels (all values).
        - non_masked_data_ranged: Non-masked pixels within the charge range.
    """
    with fits.open(fits_path) as fits_file:
        pedsubtracted_data = fits_file['CALIBRATED'].data

    clsmask_data = combined_cls
    in_range = (pedsubtracted_data >= charge_range[0]) & (pedsubtracted_data <= charge_range[1])
    not_masked = clsmask_data == 0

    non_masked_data_all = np.where(not_masked, pedsubtracted_data, np.nan)
    non_masked_data_ranged = np.where(in_range & not_masked, pedsubtracted_data, np.nan)

    return non_masked_data_all, non_masked_data_ranged
    

def calculate_normalized_values(non_masked_data_ranged, non_masked_data_all):
    """
    Calculates normalized values for each column.

    Returns:
    - normalized_values: Normalized proportion of pixels in range per column.
    - pixels_in_range_per_column: Number of pixels in range per column.
    """
    num_rows, total_columns = non_masked_data_ranged.shape
    pixels_in_range_per_column = np.nansum(~np.isnan(non_masked_data_ranged), axis=0)
    total_non_masked_per_column = np.nansum(~np.isnan(non_masked_data_all), axis=0)

    # Avoid division by zero
    normalized_values = np.full(total_columns, np.nan, dtype=float)
    valid_columns = total_non_masked_per_column > 0
    normalized_values[valid_columns] = pixels_in_range_per_column[valid_columns] / total_non_masked_per_column[valid_columns] *num_rows

    return normalized_values, pixels_in_range_per_column, total_non_masked_per_column


def detect_hot_columns_iterative(normalized_values):
    """
    Detects hot columns by iteratively identifying columns with significant differences in normalized values.

    Returns:
        tuple:
            - hot_columns_detected (list): Indices of detected hot columns.
            - threshold_values (list): Thresholds used for detection.
            - threshold_final_value (float): The final threshold value.
            - all_differences (list): Differences calculated in each iteration.
    """
    remaining_events = np.copy(normalized_values).astype(float)
    hot_columns_detected = []
    threshold_values = []
    all_differences = []  # To store differences at each iteration    
    while True:
        valid_events = remaining_events[~np.isnan(remaining_events)]
        sorted_counts = np.sort(valid_events)
        differences = sorted_counts[1:] - sorted_counts[:-1]
        all_differences.append(differences)  # Store differences        
        if len(differences) == 0 or np.nanmax(differences) <= 0:
            break        
        
        max_diff_index = np.nanargmax(differences)
        threshold_value = sorted_counts[max_diff_index + 1]
        mean_count = np.nanmean(normalized_values)   
        
        
        p        = 60 # % of columns considered as clear background
        cut_idx  = int(len(sorted_counts) * p / 100)
        mu_bck   = np.mean(sorted_counts[:cut_idx])
        sigma_bck= np.std(sorted_counts[:cut_idx])
        k_sigma  = 3.0
             
        
        if threshold_value >= mu_bck + k_sigma * sigma_bck:
            current_hot_columns = np.where(remaining_events >= threshold_value)[0]
            hot_columns_detected.extend(current_hot_columns)
            threshold_values.append(threshold_value)
        else:
            break        
        
        remaining_events[current_hot_columns] = np.nan
        
    threshold_final_value = np.nanmin(threshold_values) if threshold_values else np.nan
    return hot_columns_detected, threshold_values, threshold_final_value, all_differences
    
    
def analyze_blocks(
    normalized_values, hot_columns, threshold_final_value, neighbor_range=10, delta_threshold=5, required_matches=2
):
    """
    Analyzes and expands blocks of hot columns based on nearby values and majority voting.

    Returns:
        list: Compressed representation of expanded blocks.
        dict: Information about each block.
    """
    # Group hot columns into consecutive blocks
    hot_columns = sorted(hot_columns)
    blocks = []
    current_block = [hot_columns[0]]
    for i in range(1, len(hot_columns)):
        if hot_columns[i] - hot_columns[i - 1] == 1:  # Consecutive
            current_block.append(hot_columns[i])
        else:
            blocks.append(current_block)
            current_block = [hot_columns[i]]
    blocks.append(current_block)

    final_blocks = []
    block_info = {}

    for block in blocks:
        if len(block) == 1:  # Skip individual hot columns
            final_blocks.append(block)
            block_info[tuple(block)] = {
                "original_block": block,
                "expanded": False,
                "peak": normalized_values[block[0]],
                "min_value": normalized_values[block[0]],
                "expansion_threshold": normalized_values[block[0]] - delta_threshold,
                "original_size": 1,
                "expanded_size": 1,
            }
            continue

        # Analyze the block
        block_values = normalized_values[block]
        peak_value = np.max(block_values)
        min_value = np.min(block_values)
        

        # Define the threshold for expansion
        #expansion_threshold = threshold_final_value - delta_threshold
        expansion_threshold = threshold_final_value - delta_threshold

        # Expand block by checking neighbors with majority voting
        extended_block = block[:]

        # Check left range
        left_range = range(max(0, block[0] - neighbor_range), block[0])
        left_candidates = [
            idx for idx in left_range if normalized_values[idx] >= expansion_threshold
        ]
        if len(left_candidates) >= required_matches:
            extended_block = list(range(left_candidates[0], extended_block[-1] + 1))

        # Check right range
        right_range = range(block[-1] + 1, min(len(normalized_values), block[-1] + neighbor_range + 1))
        right_candidates = [
            idx for idx in right_range if normalized_values[idx] >= expansion_threshold
        ]
        if len(right_candidates) >= required_matches:
            extended_block = list(range(extended_block[0], right_candidates[-1] + 1))

        # Check if the block expanded
        expanded = len(extended_block) > len(block)

        # Store the extended block and its statistics
        final_blocks.append(extended_block)
        block_info[tuple(extended_block)] = {
            "original_block": block,
            "expanded": expanded,
            "peak": peak_value,
            "min_value": min_value,
            "expansion_threshold": expansion_threshold,
            "original_size": len(block),
            "expanded_size": len(extended_block),
        }

    # Merge overlapping or adjacent blocks
    merged_blocks = []
    for block in sorted(final_blocks, key=lambda x: x[0]):
        if not merged_blocks or block[0] > merged_blocks[-1][-1] + 1:
            merged_blocks.append(block)  # Add as a new block
        else:
            # Merge with the previous block
            merged_blocks[-1] = list(range(merged_blocks[-1][0], max(merged_blocks[-1][-1], block[-1]) + 1))

    # Compress the merged blocks representation
    compressed_blocks = []
    for block in merged_blocks:
        if len(block) == 1:
            compressed_blocks.append(block[0])  # Single value as an integer
        else:
            compressed_blocks.append([block[0], block[-1]])  # Range as [start, end]

    return compressed_blocks, block_info
    
def plot_sorted_normalized(normalized_values, filename="sorted_normalized_values.png",
                            *, add_scatter=True):
    """
    Plots the sorted normalized column values with log scale on Y-axis,
    automatically filtering NaNs and non-positive values.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Filter valid data (remove NaNs and values ≤ 0)
    valid = normalized_values[~np.isnan(normalized_values) & (normalized_values > 0)]
    if valid.size == 0:
        raise ValueError("No hay datos válidos (sin NaN y > 0) para representar.")

    # 2) Sort values and prepare X axis
    sorted_vals = np.sort(valid)
    x = np.arange(sorted_vals.size)

    # 3) Plot the sorted values
    plt.figure(figsize=(14, 8))
    plt.vlines(x, 1e-6, sorted_vals, color='black', linewidth=1.2)

    if add_scatter:
        plt.scatter(x, sorted_vals, s=20, color='black', zorder=3)

    #plt.yscale('log')
    plt.ylabel(r"$M_{\mathrm{col}}$")
    plt.xlabel("sorted column")

    ax = plt.gca()
    ax.set_ylim(bottom=1)
    ax.minorticks_on()
    set_axis_labels_aligned(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

    
def plot_pixels(normalized_values, pixels_in_range_per_column, hot_columns=None, pixel_count_filename="pixel_count_vs_columns.png"):

    # Plot Normalized Values per Column
    plt.figure(figsize=(14, 8))
    plt.xticks()
    plt.yticks()
    plt.plot(range(len(normalized_values)), normalized_values, linestyle='-', color='black', linewidth = 2)
    plt.yscale('log')
    plt.ylabel(r"$M_{\mathrm{col}}$")
    plt.xlabel("column")
    
    ax = plt.gca()             
    ax.minorticks_on()         
    set_axis_labels_aligned(ax) 
    #plt.grid(alpha=0.7)
    
    # Mark hot columns with vertical lines
    if hot_columns and len(hot_columns) > 0:  # Ensure hot_columns has elements
        for col in hot_columns:
            plt.vlines(x=col,
               ymin=0,
               ymax=normalized_values[col],
               color='red',
               linestyle='--',
               linewidth=2.5,
               label='Hot Column' if col == hot_columns[0] else "")


    plt.tight_layout()
    plt.show()


    
def group_consecutives(data):
    result = []
    temp = []
    for i in range(len(data)):
        if i == 0 or data[i] == data[i-1] + 1:
            temp.append(data[i])
        else:
            if len(temp) > 1:
                result.append([temp[0], temp[-1]])
            else:
                result.extend(temp)
            temp = [data[i]]
    if len(temp) > 1:
        result.append([temp[0], temp[-1]])
    else:
        result.extend(temp)

    return result
    
    

from scipy.stats import chisquare
from scipy.optimize import curve_fit

def goodness_of_flat_dc(data, errors=None):
    """
    Evaluates the goodness-of-fit of the charge distribution to a flat distribution, including errors.

    Returns:
        dict: Contains chi-square statistic, reduced chi-square, and other metrics.
    """
    # Flatten and remove NaN values
    valid_mask = ~np.isnan(data)
    flat_data = data[valid_mask]
    if errors is not None:
        flat_errors = errors[valid_mask]
    else:
        flat_errors = np.sqrt(flat_data)  # Default Poisson errors

    if len(flat_data) < 2:
        raise ValueError("Insufficient data points for goodness-of-fit evaluation.")

    # Expected value for a flat distribution
    expected = np.full_like(flat_data, flat_data.mean())

    # Adjust chi-square calculation to include errors
    chi_squared = np.sum(((flat_data - expected) / flat_errors) ** 2)
    degrees_of_freedom = len(flat_data) - 1
    reduced_chi_squared = chi_squared / degrees_of_freedom

    return {
        "chi2_stat": chi_squared,
        "reduced_chi2": reduced_chi_squared,
        "mean": np.nanmean(flat_data),
        "std": np.std(flat_data),
        "coeff_var": np.std(flat_data) / np.nanmean(flat_data) if np.nanmean(flat_data) != 0 else np.nan,
    }


def detect_overdensity_regions(profile,
                               W          = 100,
                               Wsub       = 10,       
                               FACT       = 1.50,     
                               FACTsub    = 1.25,     
                               min_size   = 100,     
                               gap_max    = 30):     
    """
    Detects over-density regions in a column-wise profile using a moving window method.

    Args:
        profile (array-like): Column-wise multiplicity values (NaN in masked regions).
        W (int): Width of the main window.
        Wsub (int): Width of the sub-window used for edge profiling.
        FACT (float): Threshold multiplier for the main window over the global median.
        FACTsub (float): Threshold multiplier for the sub-window.
        min_size (int): Minimum number of columns to accept a detected band.
        gap_max (int): Maximum gap (in columns) allowed to merge two candidate bands.

    Returns:
        list of tuples: Each tuple (start, end) represents a detected region of over-density.
    """

    prof = np.asarray(profile, float)
    μ_global = np.nanmedian(prof)

    # 1) Main window sweep
    cand = []
    for i in range(len(prof) - W + 1):
        win = prof[i:i+W]
        if np.all(np.isnan(win)):
            continue
        if np.nanmedian(win) > FACT * μ_global:
            # 2) Sub-window sweep to refine edges
            for j in range(W - Wsub + 1):
                sub   = win[j:j+Wsub]
                if np.all(np.isnan(sub)):
                    continue
                if np.nanmedian(sub) > FACTsub * μ_global:
                    s = i + j
                    e = s + Wsub - 1
                    cand.append((s, e))

    if not cand:
        return []

    # 3) Merge overlapping or adjacent segments
    cand.sort()
    merged = [list(cand[0])]
    for s, e in cand[1:]:
        ls, le = merged[-1]
        if s <= le + 1:               
            merged[-1][1] = max(le, e)
        else:
            merged.append([s, e])

    # 4) Fill small gaps between regions (gap ≤ gap_max)
    filled = [merged[0]]
    for s, e in merged[1:]:
        ls, le = filled[-1]
        if s - le - 1 <= gap_max:
            filled[-1][1] = e
        else:
            filled.append([s, e])

    # 5) Filter out regions smaller than min_size
    final = [(s, e) for s, e in filled if (e - s + 1) >= min_size]
    return final


        
# Define the binning function with error calculations
def bin_data_with_errors(data, bin_size):
    """
    Reduces the data resolution by grouping it into bins of specified size,
    and calculates the errors for each bin, handling empty bins.

    Args:
        data (numpy.ndarray): 1D array of data to bin.
        bin_size (int): Number of points per bin.

    Returns:
        tuple:
            - binned_data (numpy.ndarray): Binned data (averaged values per bin).
            - binned_errors (numpy.ndarray): Errors for each bin (standard error of the mean).
    """
    # Ensure data length is a multiple of bin_size
    trimmed_length = len(data) // bin_size * bin_size
    reshaped_data = data[:trimmed_length].reshape(-1, bin_size)

    # Initialize arrays for binned data and errors
    binned_data = np.full(reshaped_data.shape[0], np.nan)  # Default to NaN for empty bins
    binned_errors = np.full(reshaped_data.shape[0], np.nan)  # Default to NaN for errors in empty bins

    # Loop through each bin to calculate mean and errors manually
    for i in range(reshaped_data.shape[0]):
        bin_values = reshaped_data[i, :]
        valid_values = bin_values[~np.isnan(bin_values)]  # Remove NaNs
        if valid_values.size > 0:  # Only process non-empty bins
            binned_data[i] = np.mean(valid_values)
            binned_errors[i] = np.std(valid_values) / np.sqrt(valid_values.size)  # Standard error of the mean

    return binned_data, binned_errors
    

def plot_cluster_heatmap(fits_file):
    """
    Displays a heatmap of a 30x30 pixel crop centered at (2618, 2402)
    from the 'CALIBRATED' extension of a FITS file.

    The color scale is capped at 20. Axis ticks reflect actual coordinates
    of the cropped region, with ticks every 5 pixels. The colorbar is
    vertically aligned and spans the full height of the heatmap.
    """

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    # Load FITS data
    data = fits.getdata(fits_file, extname='CALIBRATED')

    center_y, center_x = 2618, 2402
    half_size = 15

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
        cbar_kws={'pad': 0.01}   # Set colorbar without shrinking it vertically
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # Set ticks every 5 pixels with real coordinates
    xtick_pos = np.arange(0, cropped_data.shape[1], 5)
    ytick_pos = np.arange(0, cropped_data.shape[0], 5)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(x_labels[xtick_pos])
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(y_labels[ytick_pos])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.tick_params(axis='both', which='major', labelsize=20)  
    ax.invert_yaxis()
    set_axis_labels_aligned(ax)
    plt.tight_layout()
    plt.show()

    
import numpy as np

def remove_baseline(counts, baseline, zero_mean=True):
    """
    Removes a baseline trend from a 1D profile.

    Args:
        counts (array-like): 1D array with the column profile (e.g., M_col).
        baseline (array-like or list): Either:
            - A 1D array of the same size as `counts`, used as-is.
            - A list or array of polynomial coefficients [a, b, c], which will be 
              evaluated as a second-degree polynomial: a*x² + b*x + c.
        zero_mean (bool): If True, forces the mean of the residual to be zero.

    Returns:
        tuple:
            - residual (np.ndarray): Difference `counts - baseline`, mean-centered if `zero_mean=True`.
            - base (np.ndarray): Evaluated baseline (useful if polynomial coefficients were provided).
    """
    counts = np.asarray(counts, float)

   
    if np.ndim(baseline) == 1 and len(baseline) == len(counts):
        base = baseline
    else:                         
        x = np.arange(len(counts))
        base = np.polyval(baseline, x)

    resid = counts - base
    if zero_mean:
        resid -= np.nanmean(resid)

    return resid, base

def main(args):

    for i, fits_file in enumerate(args.files):
        # ---------------------------------------------------------------------
        # Generate 2D masks to be saved as FITS extensions
        # ---------------------------------------------------------------------   
            
        with fits.open(fits_file, ignore_missing_simple=True,checksum=False, do_not_scale_image_data=True) as hdul_check:
            hdul_check.info()
            
            
    # Combine CLSMASK
    combined_clsmask = combine_clsmask(args.files)
    
     # Process each file separately, but using the same 'combined_clsmask'
    for i, fits_file in enumerate(args.files):
    
        print("\n" + "="*60)
        print(f"Processing file #{i+1}: {fits_file}")
        #plot_cluster_heatmap(fits_file)
        
        
        set_plot_style()
      
        # Extract processed data
        non_masked_all, non_masked_ranged = extract_data(combined_clsmask, fits_path=fits_file, charge_range=args.charge_range)

        # Calculate normalized values
        normalized_values, pixels_in_range_per_column, total_non_masked_per_column = calculate_normalized_values(non_masked_data_ranged=non_masked_ranged, non_masked_data_all=non_masked_all)

        # Detect hot columns
        hot_columns_detected, threshold_values, threshold_final_value, all_differences = detect_hot_columns_iterative(normalized_values)
    

        print("\n--- Detected Hot Columns first method---")

        # Ensure the hot columns are sorted
        hot_columns_sorted = sorted(hot_columns_detected)

        # Group consecutive columns
        grouped_hot_columns = group_consecutives(hot_columns_sorted)

        print(f"{grouped_hot_columns}")
        print(f"Final threshold value: {threshold_final_value:.4f}")
    
        
   
        # Exclude hot columns: replace them with NaN instead of dropping
        normalized_values_filtered_without_hc = np.copy(normalized_values).astype(float)
        pixels_in_range_filtered_without_hc = np.copy(pixels_in_range_per_column).astype(float)
        non_masked_all_filtered_without_hc = np.copy(non_masked_all).astype(float)
        
        
        for column in hot_columns_detected:
            normalized_values_filtered_without_hc[column] = np.nan
            pixels_in_range_filtered_without_hc[column] = np.nan
            non_masked_all_filtered_without_hc[:, column] = np.nan

        

        if not grouped_hot_columns==[]:
            final_blocks, block_info = analyze_blocks(normalized_values, hot_columns_detected, threshold_final_value, neighbor_range=10, delta_threshold=5, required_matches=2)


            print("\n--- Detected Hot Columns ---") 
            print(f"{final_blocks}")               
    

            # Flatten the indices from final_blocks
            hot_columns_from_blocks = []
            for block in final_blocks:
                if isinstance(block, list):  
                    hot_columns_from_blocks.extend(range(block[0], block[-1] + 1))
                else: 
                    hot_columns_from_blocks.append(block)

            # Replace detected hot columns with NaN
            for column in hot_columns_from_blocks:
                normalized_values_filtered_without_hc[column] = np.nan
                pixels_in_range_filtered_without_hc[column] = np.nan
                non_masked_all_filtered_without_hc[:, column] = np.nan
        
        
     # Plot distributions
        #plot_sorted_normalized(normalized_values, filename="sorted_normalized_values.png")
        plot_pixels(normalized_values=normalized_values, pixels_in_range_per_column=pixels_in_range_per_column, hot_columns=hot_columns_detected, pixel_count_filename="pixel_count_vs_columns.png")
                
                                             
        # Binning setup
        bin_size = 10
        binned_data, binned_errors = bin_data_with_errors(normalized_values_filtered_without_hc, bin_size)                                              

        flatness_metrics = goodness_of_flat_dc(binned_data, errors=binned_errors)
        print("\n--- Goodness of Fit for Flat Distribution ---")
        print(f"Mean: {flatness_metrics['mean']:.4f}")
        print(f"Standard Deviation: {flatness_metrics['std']:.4f}")
        print(f"Coefficient of Variation: {flatness_metrics['coeff_var']:.4f}")
        print(f"Chi-squared: {flatness_metrics['chi2_stat']:.4f}")    
    
        overdensity_regions = detect_overdensity_regions(normalized_values_filtered_without_hc, W=100, Wsub=10, FACT = 1.7, FACTsub= 1.2, min_size=150, gap_max=0)
  
        
        # Display results in the terminal for OD:
        print("\n--- Over-Density Regions---")
        if overdensity_regions:
            print(f"Regions with significant over-density: {overdensity_regions}")
         
        
        # Plot results with OD regions:
        plt.figure(figsize=(14, 8))
        plt.plot(normalized_values_filtered_without_hc, color="black")
        
        smooth = savgol_filter(normalized_values_filtered_without_hc, 201, 2)
        plt.plot(smooth, color = "blue")
        
        # Fit a second-degree polynomial to the smoothed profile
        x = np.arange(len(normalized_values_filtered_without_hc))
        good = ~np.isnan(normalized_values_filtered_without_hc)
        coef = np.polyfit(x[good], normalized_values_filtered_without_hc[good], deg=2)   # [a, b, c]
        poly_fit = np.polyval(coef, x)
        
        residual, baseline = remove_baseline(normalized_values_filtered_without_hc,
                                     poly_fit, zero_mean=True)

        plt.plot(x, poly_fit, color="red", label=r"$a\,x^{2}+b\,x+c$", linewidth=2.5)
        

        # Highlight detected regions    
        for region in overdensity_regions:
            plt.axvspan(region[0], region[1], color="orange", alpha=0.3, label="Detected Region")

        plt.xlabel("column")
        plt.ylabel(r"$M_{\mathrm{col}}$")
        plt.xticks()
        plt.yticks()
        plt.tight_layout()
        set_axis_labels_aligned()
        plt.show()
    
        if overdensity_regions:
 
            normalized_values_filtered_with_overdensity_removed = np.copy(normalized_values_filtered_without_hc)
            non_masked_all_filtered_with_overdensity_removed = np.copy(non_masked_all)

            for start, end in overdensity_regions:
                normalized_values_filtered_with_overdensity_removed[start:end + 1] = np.nan  # Normalized
                non_masked_all_filtered_with_overdensity_removed[:, start:end + 1] = np.nan  # No normalized (matriz)

            print("\n--- Over-Density Regions Removed ---")
        else:
            print("No significant over-density regions detected.")    
            


        with fits.open(fits_file, mode='update') as hdul:
            print("\n--- Original extensions ---")
            hdul.info()

            # Remove previous versions if they exist to avoid duplication
            for ext_name in ["MASK_CROSSTALK", "MASK_HOTCOLUMNS", "MASK_OVERDENSITY"]:
                if ext_name in hdul:
                    hdul.remove(hdul[ext_name]) 

            # Generate the new 2D masks
            mask_crosstalk = combined_clsmask.astype(np.uint8)
            mask_hotcolumns = np.zeros_like(non_masked_all, dtype=np.uint8)
            for c in hot_columns_detected:
                mask_hotcolumns[:, c] = 1  # Mark all rows in the hot column
            if not grouped_hot_columns==[]:
                for  c in hot_columns_from_blocks:
                    mask_hotcolumns[:, c] = 1 
            
            mask_overdensity = np.zeros_like(non_masked_all, dtype=np.uint8)
            for (start, end) in overdensity_regions:
                mask_overdensity[:, start:end+1] = 1  # Mark the entire detected region

            # Append the new extensions to the FITS file
            hdul.append(fits.ImageHDU(data=mask_crosstalk, name='MASK_CROSSTALK'))
            hdul.append(fits.ImageHDU(data=mask_hotcolumns, name='MASK_HOTCOLUMNS'))
            hdul.append(fits.ImageHDU(data=mask_overdensity, name='MASK_OVERDENSITY'))

            # Save changes without closing the file abruptly
            hdul.flush()

        print(f"\nMasks added in {fits_file}")
    
        # Verify that extensions were saved correctly
        with fits.open(fits_file) as hdul:
            print("\n--- Final Extensions ---")
            hdul.info() 

if __name__ == "__main__":
    import argparse
    import os
    from astropy.io import fits
    import numpy as np

    parser = argparse.ArgumentParser(description="Process 4 FITS files and calculate distances to clusters.")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help="Paths to the 4 FITS files (4 required).")
    parser.add_argument('-q', '--charge_range', type=float, nargs=2, default=(0.77, 3.77), metavar=('MIN_VALUE', 'MAX_VALUE'),
                        help="Charge value range (default 0.77 to 3.77).")
    parser.add_argument('-m', '--module_name', type=int, required=True,
                        help="Name of the module, 103 or 104")
                        
    parser.add_argument('-o', '--output', type=str, default="cluster_crosstalk_masks.png",
                        help="Output image filename (default: cluster_crosstalk_masks.png)")
    args = parser.parse_args()
    main(args)
