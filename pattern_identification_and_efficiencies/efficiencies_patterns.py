#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""

import numpy as np
from math import erf, sqrt, log
from itertools import product
from collections import defaultdict
from numba import njit
from numba.typed import List
import matplotlib.pyplot as plt
import pandas as pd


from utils_plot import set_plot_style, set_axis_labels_aligned

set_plot_style()

#########################################
# 1) Base functions for CDFs and patterns
#########################################

@njit
def cdf_value(q, m, sigma=0.15):
    """
    Computes the CDF of a Gaussian with mean `m` and standard deviation `sigma`
    evaluated at value `q`. Clamps the result to avoid log(0) errors later.
    """
    if np.isnan(q):
        return 1e-20
    z = (q - m) / (sigma * sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    return max(1e-20, min(cdf, 1.0 - 1e-20))

def make_pattern_order(n):
    """
    Generates all combinations of `n` values in [1..5] whose sum is <= 5.
    Returns them grouped by total sum, sorted in increasing order of the sum.
    """
    grouped_combos = {}
    for combo in product(range(1, 6), repeat=n):
        total = sum(combo)
        if total <= 5:
            if total not in grouped_combos:
                grouped_combos[total] = []
            grouped_combos[total].append(np.array(combo, dtype=np.int64))
    return [grouped_combos[key] for key in sorted(grouped_combos.keys())]

def pythonlist_to_numba(pylist_of_lists):
    """
    Converts a Python list of lists into a numba.typed.List of typed Lists
    to enable numba-compatible iteration.
    """
    nb_list = List()
    for sublist in pylist_of_lists:
        sub_nb_list = List()
        for item in sublist:
            sub_nb_list.append(item)
        nb_list.append(sub_nb_list)
    return nb_list
   
@njit
def precompute_neglog_cdf_matrices(data_matrix, sigma=0.15):
    """
    Precomputes:
    - neglogcdf_matrix[i, j, m] = -log(CDF(data[i,j], m, sigma)) for m in 0..5
    - neglog1mcdf0_matrix[i, j] = -log(1 - CDF(data[i,j], 0))
    Used later for efficient pattern scoring.
    """
    nrows, ncols = data_matrix.shape
    neglogcdf_matrix = np.empty((nrows, ncols, 6), dtype=np.float64)
    neglog1mcdf0_matrix = np.empty((nrows, ncols), dtype=np.float64)

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(data_matrix[i, j]):
                for m in range(6):
                    neglogcdf_matrix[i, j, m] = 1e10
                neglog1mcdf0_matrix[i, j] = 1e10
            else:
                for m in range(6):
                    val_cdf = cdf_value(data_matrix[i, j], m, sigma)
                    neglogcdf_matrix[i, j, m] = -log(val_cdf)
                val_cdf0 = cdf_value(data_matrix[i,j], 1, sigma)
                neglog1mcdf0_matrix[i, j] = -log(val_cdf0)

    return neglogcdf_matrix, neglog1mcdf0_matrix


#############################################
# 2) Pattern detection with thresholds
#############################################

@njit
def cascade_article_improved(coords, combos_nb, neglogcdf_matrix, threshold):
    """
    Evaluates each group of candidate combinations to find the one with the
    lowest score (sum of -log CDF values), stopping early if no group meets the threshold.
    """
    n = len(coords)
    best_combo = np.full(n, -1, dtype=np.int64)
    best_score = 1e12

    for combo_group in combos_nb:
        group_best_combo = None
        group_best_score = 1e12

        for combo in combo_group:
            score = 0.0
            for i in range(n):
                r, c = coords[i]
                m = combo[i]
                score += neglogcdf_matrix[r, c, m]
                if score > threshold:
                    break

            if score <= threshold and score < group_best_score:
                group_best_combo = combo.copy()
                group_best_score = score

        if group_best_combo is not None:
            best_combo[:] = group_best_combo
            best_score = group_best_score
        else:
            break

    return best_combo, best_score

@njit
def detect_patterns_diff_thresholds(data_matrix,
                                    combos1_nb, combos2_nb, combos3_nb,
                                    neglogcdf_matrix, neglog1mcdf0_matrix,
                                    threshold1=8.0, threshold2=8.0, threshold3=8.0):
    """
    Applies pattern detection across the data_matrix using length-dependent thresholds.
    Only patterns aligned along columns (left to right) are evaluated.

    Returns a list of valid patterns found, including:
        - pixel coordinates
        - best-matching charge combination
        - score (sum of -log CDFs)
        - direction (always 'Col' here)
        - list of measured charges
    """
    nrows, ncols = data_matrix.shape
    directions = [(0, 1)]  # horizontal (left to right)
    dir_names = ["Col"]
    used_pixels = np.zeros((nrows, ncols), dtype=np.uint8)
    patterns = []

    for i in range(nrows):
        for j in range(ncols):
            if used_pixels[i, j] == 1:
                continue  # Skip used pixels

            for d_idx, (di, dj) in enumerate(directions):
                last_valid_pattern = (np.empty((0, 2), dtype=np.int64), 
                                      np.empty(0, dtype=np.int64), 
                                      -1.0, "", 
                                      np.empty(0, dtype=np.float64))

                for length in (1, 2, 3):
                    if length == 1 and (j == 0 or j == ncols - 1):
                        continue
                    if length == 2 and (j == ncols - 1):
                        continue

                    coords = []
                    out_of_bounds = False
                    for k in range(length):
                        xx = i + k * di
                        yy = j + k * dj
                        if xx < 0 or xx >= nrows or yy < 0 or yy >= ncols:
                            out_of_bounds = True
                            break
                        coords.append((xx, yy))

                    if out_of_bounds:
                        continue

                    conflict = False
                    for r, c in coords:
                        if used_pixels[r, c] == 1:
                            conflict = True
                            break
                    if conflict:
                        continue

                    if length == 1:
                        thr = threshold1
                        combos_nb_ = combos1_nb
                    elif length == 2:
                        thr = threshold2
                        combos_nb_ = combos2_nb
                    else:
                        thr = threshold3
                        combos_nb_ = combos3_nb

                    best_combo, best_score = cascade_article_improved(
                        coords, combos_nb_, neglogcdf_matrix, thr
                    )

                    if best_combo[0] != -1:
                        last_valid_pattern = (np.array(coords, dtype=np.int64),
                                              best_combo.copy(),
                                              best_score,
                                              dir_names[d_idx],
                                              np.array([data_matrix[r, c] for (r, c) in coords]))

                if last_valid_pattern[0].shape[0] > 0:
                    patterns.append(last_valid_pattern)
                    for r, c in last_valid_pattern[0]:
                        used_pixels[r, c] = 1

    return patterns


######################################
# 3) Simulated dataset generation
######################################
def generate_data_2d_for_patterns(n_samples_per=500, sigma=0.15, Nx=5, Ny=5, return_with_L=False):
    """
    Each pattern (e.g., (1,), (2,1)) is embedded into the center row and padded with zeros.

    Returns:
        Dictionary mapping real pattern -> list of noisy 2D matrices.
    """
    rng = np.random.default_rng()
    all_pats = [(0,)]  # background only

    for size in [1, 2, 3]:
        for combo in product(range(1, 6), repeat=size):
            if sum(combo) <= 5:
                all_pats.append(combo)

    center_row = Nx // 2
    data_2d = {}

    for pat in all_pats:
        mats = []
        L = len(pat)
        for _ in range(n_samples_per):
            mat = np.zeros((Nx, Ny), dtype=np.float64)
            if L > 0:
                start_col = (Ny - L) // 2
                mat[center_row, start_col:start_col + L] = pat
            noise = rng.normal(0, sigma, size=(Nx, Ny))
            mat += noise
            if return_with_L:
                mats.append({"matrix": mat, "L": L})
            else:
                mats.append(mat)
        data_2d[pat] = mats

    return data_2d


######################################
# 4) Theoretical pattern weights
######################################
def build_pattern_weights(data_2d, lam):
    """
    Computes weights for each pattern based on the ratio:
        w[pat] = P_real(pat) / alpha_pat
    where:
        - alpha_pat is the fraction of total simulated events with that pattern
        - P_real is the expected probability assuming a Poisson distribution

    Args:
        data_2d: dict from pattern to list of matrices
        lam: Poisson mean (lambda)

    Returns:
        dict of pattern -> weight
    """
    from math import exp, factorial

    def poisson(k, lam):
        return exp(-lam) * (lam ** k) / factorial(k)

    total_mats = sum(len(mats) for mats in data_2d.values())
    weights = {}

    for pat, mats in data_2d.items():
        alpha_pat = len(mats) / total_mats
        p_real = 1.0
        for ki in pat:
            p_real *= poisson(ki, lam)
        p_real *= (poisson(0, lam)) ** (3 - len(pat))

        weights[pat] = p_real / alpha_pat if alpha_pat > 0 else 0.0

    return weights

######################################
# 5) Classification and confusion matrix
######################################
def classify_2d_and_build_confusion(data_2d, 
                                 combos1_nb, combos2_nb, combos3_nb,
                                 threshold1, threshold2, threshold3,
                                 sigma, weights=None):
    """
    Classifies simulated matrices using the pattern detection logic,
    builds a weighted confusion matrix: real pattern -> detected pattern.

    If no pattern is detected, assigns (0,) as the detected pattern.
    """
    confusion = defaultdict(lambda: defaultdict(float))
    confusion_sq = defaultdict(lambda: defaultdict(float))
    real_count = defaultdict(float)
    real_count_sq = defaultdict(float)
    detect_count = defaultdict(float)
    detect_count_sq = defaultdict(float)
    
    results_list = []
    correct = 0.0
    total = 0.0
  

    for real_pat, mats_list in data_2d.items():
        weight = weights.get(real_pat, 1.0) if weights else 1.0

        for mat in mats_list:
            neglogcdf_m, neglog1mcdf0_m = precompute_neglog_cdf_matrices(mat, sigma)
            found = detect_patterns_diff_thresholds(
                mat,
                combos1_nb, combos2_nb, combos3_nb,
                neglogcdf_m, neglog1mcdf0_m,
                threshold1, threshold2, threshold3
            )
            

            if len(found) == 0:
                detected_pat = (0,)
                pixel_charges = []
            else:
                coords, best_combo, score, direction, qvals = found[0]
                detected_pat  = tuple(best_combo)
                pixel_charges = qvals

            confusion[real_pat][detected_pat] += weight
            confusion_sq[real_pat][detected_pat] += weight**2
            
            real_count[real_pat] += weight
            real_count_sq[real_pat] += weight**2
            
            detect_count[detected_pat] += weight
            detect_count_sq[detected_pat] += weight**2
            
            total += weight
            if detected_pat == real_pat:
                correct += weight
            
            results_list.append((real_pat, detected_pat, pixel_charges))

    return (confusion, confusion_sq,
            real_count, real_count_sq,
            detect_count, detect_count_sq,
            correct, total, results_list)


def plot_confusion_matrix_global(confusion, title="Global Confusion Matrix"):
    """
    Plots a single confusion matrix heatmap for all real/detected patterns:
      - rows = real patterns (D)
      - cols = detected patterns (E)
    Normalizes each row to sum to 1, so each cell is p(D->E).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect all unique patterns
    all_pats = set(confusion.keys())
    for D in confusion:
        all_pats.update(confusion[D].keys())
    all_pats = sorted(all_pats)

    pat_to_idx = {p: i for i, p in enumerate(all_pats)}
    n = len(all_pats)

    mat = np.zeros((n, n), dtype=np.float64)

    # Fill the matrix, normalizing each row to sum to 1
    for D in confusion:
        row_sum = sum(confusion[D].values())
        if row_sum > 0:
            for E in confusion[D]:
                mat[pat_to_idx[D], pat_to_idx[E]] = confusion[D][E] / row_sum

    # Confusion plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(mat, origin='upper', cmap='Blues', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(p) for p in all_pats], rotation=90)
    ax.set_yticklabels([str(p) for p in all_pats])
    ax.set_xlabel("Detected Pattern (E)")
    ax.set_ylabel("Real Pattern (D)")
    cbar = plt.colorbar(cax, ax=ax, pad=0.01)
    cbar.ax.tick_params()
    cbar.set_label("p(D→E)")
    set_axis_labels_aligned(ax)

    plt.tight_layout()
    plt.show()


######################################
# 6) Metric computation utilities
######################################
def compute_recall_misid_score(conf_s):
    """
    Computes global performance metrics from a given confusion matrix:
        - recall: correct detections / total real instances
        - precision: correct detections / total detections
        - misID: 1 - precision
        - score: recall - misID

    Returns:
        recall, misID, score, hits, total_detected
    """
    hits = 0.0
    total_real = 0.0

    for D in conf_s:
        hits += conf_s[D].get(D, 0.0)
        total_real += sum(conf_s[D].values())

    recall = hits / total_real if total_real > 0 else 0.0

    total_detected = sum(
        conf_s[D][E] for D in conf_s for E in conf_s[D]
    )
    precision = hits / total_detected if total_detected > 0 else 1.0
    misID = 1.0 - precision
    score = recall - misID

    return recall, misID, score, hits, total_detected

def compute_pattern_metrics(confusion):
    """
    Computes recall and misidentification rate for each pattern `X`:
        recall[X] = #(X -> X) / #(X -> anything)
        misID[X] = #(not X -> X) / #(anything -> X)
    """
    recall_p = {}
    misid_p = {}
    all_pats = set(confusion.keys())
    for d in confusion:
        all_pats.update(confusion[d].keys())

    for X in all_pats:
        total_real_X = sum(confusion.get(X, {}).values())
        correct_X = confusion.get(X, {}).get(X, 0.0)
        recallX = correct_X / total_real_X if total_real_X > 0 else float('nan')

        detect_X = sum(confusion[d2].get(X, 0.0) for d2 in confusion)
        precisionX = correct_X / detect_X if detect_X > 0 else float('nan')
        misX = 1 - precisionX if detect_X > 0 else float('nan')

        recall_p[X] = recallX
        misid_p[X] = misX

    return recall_p, misid_p


######################################
# 7) Exporting efficiencies
######################################
def save_eficiencies(confusion_s, path_out="results_efficiencies.txt"):
    """
    Saves recall, misID and transition probabilities p(D->E) for each pattern
    into a text file.
    """
    recall_dict, misid_dict = compute_pattern_metrics(confusion_s)
    with open(path_out, "w", encoding="utf-8") as f:
        f.write("RealPattern\tDetectedPattern\tp(D->E)\tRecallD\tMisID_D\n")
        for D in sorted(confusion_s.keys()):
            totalD = sum(confusion_s[D].values())
            recallD = recall_dict.get(D, float('nan'))
            misD = misid_dict.get(D, float('nan'))
            for E in sorted(confusion_s[D].keys()):
                pDE = confusion_s[D][E] / totalD if totalD > 0 else 0.0
                f.write(f"{D}\t{E}\t{pDE:.6f}\t{recallD:.6f}\t{misD:.6f}\n")
    print(f"File saved: {path_out}")
    

######################################
# 8) Filter confusion matrix by pattern size
######################################
def filter_confusion_by_size(confusion_dict, s):
    """
    Filters the confusion matrix to retain only entries where the real pattern
    has a specific size (number of elements).
    """
    filtered = {}
    for D in confusion_dict:
        if len(D) == s:
            filtered[D] = confusion_dict[D]
    return filtered



######################################
# 9) Pattern plots
######################################

def plot_scores_lengthsTogether(scores1, scores2, scores3,
                                thresholds=(3.5, 4.0, 5.5),
                                fname="scores_lengths_1_2_3.pdf", print_examples=True, print_scores_only=False):
    """
    This function plots score distributions for patterns of length 1, 2, and 3:
    - Example: (1), (1,1), and (1,1,1) in a single figure
    - Inputs are three score arrays and their respective thresholds
    """
    import matplotlib.pyplot as plt
    import numpy as np, os
    import numpy as np, os, pprint, random
    

    colors = ['black', 'blue', 'purple']
    labels = [r"$\Lambda_{1}$",
              r"$\Lambda_{1,1}$",
              r"$\Lambda_{1,1,1}$"]

    fig, ax = plt.subplots(figsize=(12, 10))

    for vals, thr, lab, col in zip([scores1, scores2, scores3],
                                   thresholds, labels, colors):
        if len(vals) == 0:
            continue
        ax.hist(vals,
                bins=100,
                histtype='step',
                color=col,
                label=f"{lab}", lw=2.5)
        ax.axvline(thr, color='red', linestyle='--', linewidth=2.5)

    ax.set_xlabel(r"$\Lambda_{m,n,l}$")
    ax.set_ylabel("counts")
    ax.set_yscale("log")
    ax.tick_params(axis='both')
    ax.grid(False)
    ax.legend(loc="upper right", fontsize = 27)
    plt.xticks()
    plt.yticks()
    set_axis_labels_aligned(ax)
    plt.tight_layout()

    os.makedirs("eff_pattern_unblinded", exist_ok=True)
    path = f"eff_pattern_unblinded/{fname}"
    plt.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close()    

def plot_extra_debug_info(detailed_list, data_2d, sigma=0.16):
    """
    This function visualizes debugging information for detected charge patterns:
    (A) 2D scatter or density plots of charge pairs (q[i], q[i+1]) for each detected pattern.
    (B) 2D density plot using Gaussian KDE for patterns of length 2.
    (C) 1D histograms of the charges q[i], q[i+1], and q[i+2] per pattern.

    Parameters:
    - detailed_list: list of tuples (real_pattern, detected_pattern, [q[i], q[i+1], ...])
    - data_2d: not used in this function (placeholder for compatibility)
    - sigma: (not used) kept for consistency in signature
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from math import log
    from scipy.stats import gaussian_kde
    from utils_plot import set_axis_labels_aligned
    import numpy as np, os, pprint, random
    
    # Global figure size
    plt.rcParams["figure.figsize"] = (12, 10)

    scores = []
    third_charges = []
    pattern_points = defaultdict(list)
    pattern_charges = defaultdict(lambda: ([], [], [])) # q[i], q[i+1], q[i+2]
    
    # Extract charge values for each detected pattern
    for real_pat, detected_pat, qvals in detailed_list:
        if len(qvals) >= 2:
            pattern_points[detected_pat].append((qvals[0], qvals[1]))
            pattern_charges[detected_pat][0].append(qvals[0])
            pattern_charges[detected_pat][1].append(qvals[1])
            if len(qvals) >= 3:
                pattern_charges[detected_pat][2].append(qvals[2])

    # ------------------------------------------------------------------------
    # (A) 2D Scatter / Density with third charge as color (if any)
    # ------------------------------------------------------------------------
    for pat in sorted(pattern_points.keys()):
        pts = pattern_points[pat]
        if len(pts) == 0:
            continue
        
        pts = np.array(pts)  
        x, y = pts[:, 0], pts[:, 1]
        xy = np.vstack([x, y])
        dens_xy = gaussian_kde(xy)(xy)
        idx = dens_xy.argsort()
        x, y, dens = x[idx], y[idx], dens_xy[idx]

        # Plot
        fig, ax = plt.subplots(figsize=(12,10))
        
        if len(pat) == 2:
            # For 2-pixel patterns: simple scatter
            ax.scatter(x, y, facecolors='white', edgecolors='black', s=20, marker='o', linewidths=1, label=f"Pattern {pat}")
     
        elif len(pat) == 3:
            # For 3-pixel patterns: use q[i+2] as color
            c = pattern_charges[pat][2]  
            z = np.array(c)  
            sc = ax.scatter(x, y, c=z, cmap='viridis', s=20, marker='o', label=f"Pattern {pat}")
            cb = plt.colorbar(sc, ax=ax)
            cb.ax.tick_params()
            cb.set_label(r"$q[i+2]$", fontsize=32)
        
        # Reference lines (thresholds)
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2.5)
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2.5)
        ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2.5)
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=2.5)

        ax.set_xlabel("q[i]", fontsize = 32)
        ax.set_ylabel("q[i+1]", fontsize = 32)
        ax.tick_params(axis='both')
        ax.grid(False)
        ax.legend(loc='upper right')
        ax.set_xlim(left=0.5)
        ax.set_ylim(bottom=0)
        plt.xticks()
        plt.yticks()
        set_axis_labels_aligned(ax)
        plt.tight_layout()
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)
        pattern_str = "_".join(map(str, pat))
        plt.savefig(f"eff_pattern_unblinded/pattern_{pattern_str}.png", dpi=300)
        print(f"Save pattern_{pattern_str}.pdf")
        #plt.show()
        
    # ------------------------------------------------------------------------
    # (B) 2D Density visualization with colorbar (only for len=2)
    # ------------------------------------------------------------------------
    for pat in sorted(pattern_points.keys()):
        pts = pattern_points[pat]
        if len(pts) == 0:
            continue
        
        pts = np.array(pts) 
        x, y = pts[:, 0], pts[:, 1]
        xy = np.vstack([x, y])
        dens_xy = gaussian_kde(xy)(xy)
        idx = dens_xy.argsort()
        x, y, dens = x[idx], y[idx], dens_xy[idx]
        
        fig, ax = plt.subplots(figsize=(12,10))
        if len(pat) == 2:
            sc = ax.scatter(x, y, c=dens, cmap='viridis', s=20, marker='o', label=f"Pattern {pat}")
            cb = plt.colorbar(sc, ax=ax, pad=0.01)
            cb.ax.tick_params()
            cb.set_label("Density", fontsize = 32)
        
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2.5)
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2.5)
        ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2.5)
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=2.5)

        ax.set_xlabel("q[i]", fontsize = 32)
        ax.set_ylabel("q[i+1]", fontsize = 32)
        ax.tick_params(axis='both')
        ax.grid(False)
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.xticks()
        plt.yticks()
        set_axis_labels_aligned(ax)
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)        
        plt.tight_layout()
        pattern_str = "_".join(map(str, pat))
        plt.savefig(f"eff_pattern/pattern_{pattern_str}_density.png", dpi=300)
        print(f"Save pattern_{pattern_str}_density.pdf")
        plt.show()
        
    # ------------------------------------------------------------------------
    # (C) 1D Histograms of q[i], q[i+1], q[i+2] per detected pattern
    # ------------------------------------------------------------------------
    for pat in sorted(pattern_charges.keys()):
        q0, q1, q2 = pattern_charges[pat]

        fig, ax = plt.subplots(figsize=(12,10))
        if len(q0) > 0:
            ax.hist(q0, bins=100, histtype='step', color='black', label='q[i]', linewidth=2.5)
        if len(q1) > 0:
            ax.hist(q1, bins=100, histtype='step', color='blue', label='q[i+1]', linewidth=2.5)
        if len(q2) > 0:
            ax.hist(q2, bins=100, histtype='step', color='purple', label='q[i+2]', linewidth=2.5)
        
        ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2.5)
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2.5)
        ax.set_xlabel(r"charge (e$^{-}$)", fontsize=32)
        ax.set_ylabel("counts", fontsize=32)
        ax.tick_params(axis='both')
        ax.legend(loc='upper right')
        ax.grid(False)
        ax.set_yscale("log")
        ax.set_xlim(left=0.5)
        plt.xticks()
        plt.yticks()
        set_axis_labels_aligned(ax)
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)
        plt.tight_layout()
        
        pattern_str = "_".join(map(str, pat))
        plt.savefig(f"eff_pattern_unblinded/pattern_{pattern_str}_charge_counts.pdf", dpi=300)
        print(f"Save pattern_{pattern_str}_charge_counts.pdf")

def plot_scores_patterns(data_2d, combos1_py, combos2_py, combos3_py, sigma=0.16, Ny=3):
    """
    Plots score distributions for pattern templates of length 1, 2, and 3.

    For each pattern combination:
      · Iterates over all candidate samples of the same length
      · Computes the cumulative distribution score
      · Draws histogram of the score with threshold line

    Parameters:
    - data_2d: dict mapping real patterns to 2D matrices or samples
    - combosX_py: list of pattern combinations of length X
    - sigma: standard deviation for Gaussian CDF
    - Ny: number of rows (used to slice matrix sample)
    """
    import matplotlib.pyplot as plt
    from math import log
    from collections import defaultdict
    from utils_plot import set_axis_labels_aligned
    
    import matplotlib.pyplot as plt
    import numpy as np, os
    import numpy as np, os, pprint, random

    # Define score thresholds for L = 1, 2, 3
    thr_map = {1: 3.5, 2: 4.0, 3: 5.5}

    # Step 1: Extract all valid samples of lengths 1–3
    all_samples = {1: [], 2: [], 3: []}
    for real_pat, mats in data_2d.items():
        L = len(real_pat)
        if L < 1 or L > 3 or real_pat == (0,):
            continue
        start = (Ny - L)//2
        for sample in mats:
            if isinstance(sample, dict):
                mat = sample["matrix"]
            elif isinstance(sample, (tuple,list)) and hasattr(sample[0], "shape"):
                mat = sample[0]
            else:
                mat = sample
            qvals = mat.ravel()[start:start+L]
            if len(qvals) == L:
                all_samples[L].append(qvals)

    # Step 2: Compute scores and plot per pattern combination
    for L, combos in ((1, combos1_py), (2, combos2_py), (3, combos3_py)):
        for pat in combos:
            scores = []
            for qvals in all_samples[L]:
                s = 0.0
                for q_meas, q_exp in zip(qvals, pat):
                    p = max(cdf_value(q_meas, q_exp, sigma), 1e-12)
                    s += -log(p)
                scores.append(s)

            # Step 3: Plot histogram
            plt.figure(figsize=(12, 10))
            plt.hist(scores, bins=100, histtype='step', lw=2.5, color='black')
            plt.axvline(thr_map[L], color='red', linestyle='--', lw=2.5)
            plt.yscale('log')
            pattern_label = ",".join(str(x) for x in pat)
            plt.xlabel(rf"$\Lambda_{{{pattern_label}}}$", fontsize = 32)
            plt.ylabel("counts", fontsize = 32)
            ax = plt.gca()
            set_axis_labels_aligned(ax)
            ax.tick_params(axis='x', labelsize=32)
            ax.tick_params(axis='y', labelsize=32)
            
            fname = f"scores_pattern_L{L}_{'_'.join(map(str,pat))}.pdf"
            os.makedirs("eff_pattern_unblinded", exist_ok=True)
            path = f"eff_pattern_unblinded/{fname}"
            plt.savefig(path, dpi=300)
            print(f"Saved {path}")
            plt.close() 
      
 
def plot_example_1_11_111(data_2d, sigma=0.16, Ny=3):
    """
    Generates score histograms for patterns of lengths 1, 2, and 3 (e.g., (1,), (1,1), (1,1,1)).
    Displays up to 10 example patterns with scores above the threshold.
    """
    import matplotlib.pyplot as plt
    import numpy as np, random, pprint
    from math import log
    from collections import defaultdict
    from utils_plot import set_axis_labels_aligned

    # Thresholds for each pattern length
    thr_map = {1: 3.5, 2: 4.0, 3: 5.5}
    scores_len = {1: [], 2: [], 3: []}  # score lists by length
    examples   = {1: [], 2: [], 3: []}  # examples above threshold

    # ------------ 1. Scan all patterns in dataset -----------------
    for pat, mats in data_2d.items():
        if pat == (0,):         # skip pattern (0,)
            continue
        L = len(pat)
        if L == 0 or L > 3:
            continue
        start_col = (Ny - L)//2  # center pattern in Ny vector

        for sample in mats:
            # Unwrap matrix
            if isinstance(sample, dict):
                mat = sample["matrix"]
            elif isinstance(sample, (tuple, list)) and hasattr(sample[0], "shape"):
                mat = sample[0]
            else:
                mat = sample

            qvals = mat.ravel()[start_col:start_col+L]  # extract relevant q values

            # Calculate score using CDF (lower is better match)
            score = sum(-log(max(cdf_value(q, 1, sigma), 1e-12)) for q in qvals)
            scores_len[L].append(score)

            # Save up to 10 examples above threshold
            if score > thr_map[L] and len(examples[L]) < 10:
                examples[L].append((round(score, 6),
                                    [round(float(q), 3) for q in qvals]))

    # ------------ 2. Print examples above thresholds --------------
    for L in (1, 2, 3):
        print(f"\n### Examples with score Λ > {thr_map[L]}  (L={L}) ###")
        pprint.pp(examples[L], width=60, compact=True)

    # ------------ 3. Plot unified histograms ----------------------
    plot_scores_lengthsTogether(
        scores_len[1], scores_len[2], scores_len[3],
        thresholds=(thr_map[1], thr_map[2], thr_map[3]),
        fname="scores_lengths_1_2_3_f.pdf"
    )

    
    
######################################
# 10) Uncertainties and saving
######################################    
    
import numpy as np
from collections import defaultdict


def compute_pattern_uncertainties(confusion, confusion_sq,
                                  real_count, real_count_sq,
                                  detect_count, detect_count_sq):
    """
    Computes recall, precision, misID and associated errors for each pattern.

    Arguments:
        confusion:        real -> detected pattern weight sums
        confusion_sq:     real -> detected pattern squared weights
        real_count:       real pattern -> total weight
        real_count_sq:    real pattern -> total squared weight
        detect_count:     detected pattern -> total weight
        detect_count_sq:  detected pattern -> total squared weight

    Returns:
        recall, recall_err
        precision, precision_err
        misID, misID_err
        pDE_err: dict (real -> detected) -> error in p(D->E)
    """
    recall = {}
    recall_err = {}
    precision = {}
    precision_err = {}
    misID = {}
    misID_err = {}
    pDE_err = {}  # error in each transition probability

    # Collect all patterns that appear anywhere
    all_pats = set(real_count.keys()) | set(detect_count.keys()) | set(confusion.keys())
    for d in confusion:
        all_pats |= set(confusion[d].keys())

    for pat in all_pats:
        total_real = real_count.get(pat, 0.0)
        sum_w2_real = real_count_sq.get(pat, 0.0)
        correct = confusion.get(pat, {}).get(pat, 0.0)

        # Recall = correct / real, error by effective N
        if total_real > 0 and sum_w2_real > 0:
            p_rec = correct / total_real
            # N_eff para real: (sum w)^2 / sum(w^2)
            N_eff_real = (total_real ** 2) / sum_w2_real
            recall_err_val = np.sqrt(p_rec * (1 - p_rec) / N_eff_real)
        elif total_real > 0:
            p_rec = correct / total_real
            recall_err_val = np.nan
        else:
            p_rec = np.nan
            recall_err_val = np.nan

        # Precision = correct / detected, error by effective N
        total_det = detect_count.get(pat, 0.0)
        sum_w2_det = detect_count_sq.get(pat, 0.0)
        if total_det > 0 and sum_w2_det > 0:
            p_prec = correct / total_det
            N_eff_det = (total_det ** 2) / sum_w2_det
            precision_err_val = np.sqrt(p_prec * (1 - p_prec) / N_eff_det)
        elif total_det > 0:
            p_prec = correct / total_det
            precision_err_val = np.nan
        else:
            p_prec = np.nan
            precision_err_val = np.nan

        recall[pat] = p_rec
        recall_err[pat] = recall_err_val
        precision[pat] = p_prec
        precision_err[pat] = precision_err_val
        misID[pat] = 1 - p_prec if not np.isnan(p_prec) else np.nan
        misID_err[pat] = precision_err_val
    
    # Error in transition probabilities p(D->E)
    for D, inner in confusion.items():
        totalD = real_count.get(D, 0.0)
        sum_w2_totalD = real_count_sq.get(D, 0.0)
        if totalD > 0 and sum_w2_totalD > 0:
            N_eff_D = (totalD ** 2) / sum_w2_totalD
            for E, sum_w_DE in inner.items():
                pDE = sum_w_DE / totalD
                pDE = min(max(pDE, 0.0), 1.0) # Clamp to [0,1]
                # binomial:
                err = np.sqrt(pDE * (1.0 - pDE) / N_eff_D)
                pDE_err[(D, E)] = err
        else:
            for E in inner.keys():
                pDE_err[(D, E)] = np.nan

    return recall, recall_err, precision, precision_err, misID, misID_err, pDE_err

def save_eficiencies_with_errors(
        confusion, confusion_sq,
        real_count, real_count_sq,
        detect_count, detect_count_sq,
        path_out="results_efficiencies_with_errors.txt"):
    """
    Computes and saves pattern identification metrics and uncertainties to a tab-separated file.

    Outputs a table with the following columns:
    RealPattern, DetectedPattern, p(D->E), Err_p(D->E),
    RecallD, Err_RecallD, MisID_D, Err_MisID_D
    """
    
    # First compute the metrics and their uncertainties
    recall, recall_err, precision, precision_err, misID, misID_err, pDE_err = compute_pattern_uncertainties(
        confusion, confusion_sq,
        real_count, real_count_sq,
        detect_count, detect_count_sq
    )

    # Construct rows for the output table
    rows = []
    # Sort patterns by length and then lexicographically
    patterns = sorted(confusion.keys(), key=lambda x: (len(x), x))
    for D in patterns:
        totalD = real_count.get(D, 0.0)
        recallD = recall.get(D, np.nan)
        err_rec = recall_err.get(D, np.nan)
        misD = misID.get(D, np.nan)
        err_mis = misID_err.get(D, np.nan)
        inner = confusion[D]
        for E in sorted(inner.keys(), key=lambda x: (len(x), x)):
            pDE = inner[E] / totalD if totalD > 0 else np.nan
            err_pDE = pDE_err.get((D, E), np.nan)
            rows.append({
                "RealPattern": str(D),
                "DetectedPattern": str(E),
                "p(D->E)": pDE,
                "Err_p(D->E)": err_pDE,
                "RecallD": recallD,
                "Err_RecallD": err_rec,
                "MisID_D": misD,
                "Err_MisID_D": err_mis
            })

    df = pd.DataFrame(rows)
    # Save the results to a tab-separated file
    df.to_csv(path_out, sep="\t", index=False, float_format="%.6f")
    print(f"File saved: {path_out}")

def flatten_all_pattern_arrays(pattern_list):
    """
    Flattens a list of lists of arrays into a list of tuples.

    Example:
    Input:  [[array([1, 2])], [array([2, 1]), array([1, 2])]]
    Output: [(1, 2), (2, 1), (1, 2)]
    """
    out = []
    for sublist in pattern_list:
        for arr in sublist:
            out.append(tuple(int(x) for x in arr))
    return out  

######################################
# 9) Main runner for simulation + classification
######################################
def main_individual_different_thresholds(
    sigma=0.16,
    sigma_name="_example",
    lam=3.2e-4,
    lam_name="",
    thr1=3.5,
    thr2=4.0,
    thr3=5.5
):
    """
    1) Generate combos for pattern lengths 1,2,3.
    2) Generate the dataset (all patterns).
    3) Classify with different thresholds for each length
       but produce a single global confusion matrix.
    4) Compute and print global metrics.
    5) Save p(D->E), recall(D), misID(D) for all patterns to one file.
    6) Plot a single global confusion matrix (row = real pattern, col = detected pattern).
    """

    # 1) Prepare combos for each pattern length
    combos1 = pythonlist_to_numba(make_pattern_order(1))
    combos2 = pythonlist_to_numba(make_pattern_order(2))
    combos3 = pythonlist_to_numba(make_pattern_order(3))


    combos1_py = flatten_all_pattern_arrays(make_pattern_order(1))
    combos2_py = flatten_all_pattern_arrays(make_pattern_order(2))
    combos3_py = flatten_all_pattern_arrays(make_pattern_order(3))
    
    # 2) Generate dataset and build weights
    return_with_L = False
    
    data_2d = generate_data_2d_for_patterns(
        n_samples_per=10000,
        sigma=sigma,
        Nx=1,
        Ny=3, return_with_L=return_with_L
    )
    plot_example_1_11_111(data_2d)
    plot_scores_patterns(data_2d, combos1_py, combos2_py, combos3_py)
    
    # 2) Generate dataset and build weights
    
    return_with_L = True
    
    data_2d = generate_data_2d_for_patterns(
        n_samples_per=1000000,
        sigma=sigma,
        Nx=1,
        Ny=3, return_with_L=return_with_L
    )
    weights_dict = build_pattern_weights(data_2d, lam)

    # 3) Classify all events in a single pass
    #         detect_patterns_diff_thresholds internally uses thr1 for length=1,
    #         thr2 for length=2, thr3 for length=3. This yields ONE dictionary:
    #         confusion_full[D][E] = weighted count of (D->E).
    (confusion_full, confusion_sq_full,
     real_count_full, real_count_sq_full,
     detect_count_full, detect_count_sq_full,
     correct_full, total_full, detailed_list) = \
    classify_2d_and_build_confusion(
        data_2d,
        combos1, combos2, combos3,
        threshold1=thr1,
        threshold2=thr2,
        threshold3=thr3,
        sigma=sigma,
        weights=weights_dict
    )
    
    import pandas as pd

    # 4) Compute GLOBAL metrics from confusion_full
    recall_global, misID_global, score_global, hits_global, total_detected_global = \
        compute_recall_misid_score(confusion_full)

    print("\n=== GLOBAL Classification Summary (All Patterns) ===")
    print(f"  Thresholds used => length=1: {thr1}, length=2: {thr2}, length=3: {thr3}")
    print(f"  Real patterns (sum) = {sum(real_count_full.values()):.1f}")
    print(f"  Detected patterns (sum) = {sum(detect_count_full.values()):.1f}")
    print(f"  Global recall = {recall_global:.3f}")
    print(f"  Global misID  = {misID_global:.3f}")
    print(f"  Global score  = {score_global:.3f}")

    # 5) Save per-pattern p(D->E), recall(D), misID(D) into ONE text file
    import os

    # Define folder:
    folder = "results_efficiency_blinded"
    #folder = "results_efficiency_unblinded"
    os.makedirs(folder, exist_ok=True)  # Crea la carpeta si no existe
    outfile = os.path.join(folder, f"results_efficiencies_full_{lam_name}_{sigma_name}.txt")

    # Call eficiencies function
    #save_eficiencies(confusion_full, path_out=outfile)
    save_eficiencies_with_errors(
        confusion_full, confusion_sq_full,
        real_count_full, real_count_sq_full,
        detect_count_full, detect_count_sq_full,
        path_out=outfile
    )

    # 6) Plot single global confusion matrix
    plot_confusion_matrix_global(confusion_full, title="Global Confusion Matrix")

    plot_extra_debug_info(detailed_list, data_2d)

def run_efficiency_sim():
    thr1=3.5
    thr2=4.0
    thr3=5.5
    sigma = [0.155, 0.159, 0.162, 0.167, 0.153, 0.158, 0.161, 0.164, 0.153, 0.157, 0.160, 0.164, 0.153, 0.155, 0.158, 0.164]
    sigma_name = ["0.155", "0.159", "0.162", "0.167", "0.153", "0.158", "0.161", "0.164", "0.153", "0.157", "0.160", "0.164", "0.153", "0.155", "0.158", "0.164"]
    
    lam= [0.0002815, 0.0002345, 0.0002592, 0.0003086, 0.0002805, 0.0002216, 0.0002709, 0.0003114, 0.000291, 0.0002317, 0.0002894, 0.0003211, 0.0002906, 0.0002321, 0.0002744, 0.0003217]
    lam_name= ["0.0002815", "0.0002345", "0.0002592", "0.0003086", "0.0002805", "0.0002216", "0.0002709", "0.0003114", "0.000291", "0.0002317", "0.0002894", "0.0003211", "0.0002906", "0.0002321", "0.0002744", "0.0003217"]
    
    # Unblinded data
    #sigma = [0.152, 0.156, 0.159, 0.163]
    #sigma_name = ["0.152", "0.156", "0.159", "0.163"]
    
    #lam= [0.0003602, 0.0003013, 0.0003203, 0.0004006]
    #lam_name= ["0.0003602", "0.0003013", "0.0003203", "0.0004006"]
    
    for i in range(len(sigma)):
        print(f"\n--- Running for sigma={sigma[i]}, lambda={lam[i]} ---")
        main_individual_different_thresholds(
            sigma=sigma[i],
            sigma_name=sigma_name[i],
            lam=lam[i],
            lam_name=lam_name[i],
            thr1=thr1,
            thr2=thr2,
            thr3=thr3
        )
        
        
if __name__ == "__main__":
    #run_efficiency_sim()
    main_individual_different_thresholds()

