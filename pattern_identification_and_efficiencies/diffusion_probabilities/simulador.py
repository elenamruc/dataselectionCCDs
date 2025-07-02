#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz

This script simulates electron diffusion in a CCD and models the probability 
that a deposited charge q gives rise to a detectable pattern p. It includes:
- Charge generation based on precomputed pair-creation probabilities.
- Monte Carlo simulation of charge diffusion and noise application.
- Pattern detection using negative log-CDF scoring and isolation criteria.
- Estimation of transition statistics before and after diffusion.
- Computation and optional saving of p_diff(q → p) values.
- Visualization of charge histograms and pattern probabilities.

"""



import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, log
from itertools import product
from matplotlib.ticker import FuncFormatter
import json
from pathlib import Path

from numba import njit
from numba.typed import List
from scipy.stats import poisson

from charge_ionization import Probability_Neh_E
from diffusion import generate_event, diffuse_event
from collections import defaultdict

from utils_plot import set_plot_style, set_axis_labels_aligned


set_plot_style()

PIX_SIZE = 15.0  # Tamaño del píxel en micras
N_cols = 6300
N_rows = 6080 * 100


#####################
# UTILS Y DETECTION #
#####################

@njit
def neighbors_8(i, j):
    """
    Returns the 8-connected neighborhood of pixel (i, j).
    """
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            out.append((i + di, j + dj))
    return out

@njit
def cdf_value(q, m, sigma=0.15):
    """
    Computes the cumulative distribution function (CDF) of a Gaussian
    with mean m and standard deviation sigma at value q.
    Returns values clamped between 1e-20 and 1 - 1e-20 to avoid log(0).
    """
    if np.isnan(q):
        return 1e-20
    z = (q - m) / (sigma * sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    # Clampeamos para evitar 0 o 1 exactos.
    return max(1e-20, min(cdf, 1.0 - 1e-20))
    
def make_pattern_order(n):
    """
    Generates pattern combinations of length n grouped by total charge sum.
    Only combinations with total sum ≤ 5 are included.
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
    Converts a nested Python list into a numba.typed.List of typed lists.
    Useful for numba-accelerated functions.
    """
    nb_list = List()
    for sublist in pylist_of_lists:
        sub_nb_list = List()
        for tup in sublist:
            sub_nb_list.append(tup)
        nb_list.append(sub_nb_list)
    return nb_list


@njit
def precompute_neglog_cdf_matrices(data_matrix, sigma=0.15):
    """
    Precomputes:
    - neglogcdf_matrix[i, j, m] = -log( CDF(data[i,j], m, sigma) ) for m=0..3
    - neglog1mcdf0_matrix[i, j] = -log( CDF(data[i,j], m=1) )
    
    NaN pixels are penalized with high log-probability values (1e10).
    """
    nrows, ncols = data_matrix.shape
    neglogcdf_matrix = np.empty((nrows, ncols, 4), dtype=np.float64)
    neglog1mcdf0_matrix = np.empty((nrows, ncols), dtype=np.float64)

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(data_matrix[i,j]):
                # Penalize NaNs with high score
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
def cascade(coords, combos_nb, neglogcdf_matrix, threshold):
    """
    Evaluates combinations of expected charges at given coordinates, ordered by total charge.
    Returns the best-scoring combination (lowest -log score), if any under the threshold.
    """
    n = len(coords)
    best_combo = np.full(n, -1, dtype=np.int64)  # Initialize best combo with -1
    best_score = 1e12 # Start with a high score

    for combo_group in combos_nb:
        at_least_one_valid = False
        group_best_combo = None  
        group_best_score = 1e12  

        for combo in combo_group:
            score = 0.0
            for i_cord in range(n):
                r, c = coords[i_cord]
                m = combo[i_cord]
                score += neglogcdf_matrix[r, c, m]
              
                
                if score > threshold:
                    break  # Skip if score already exceeds threshold

            if score <= threshold:
                if score < group_best_score:  
                    group_best_combo = combo.copy()
                    group_best_score = score
                at_least_one_valid = True

        # Replace global best if this group had valid combination(s)
        if at_least_one_valid:
            best_combo[:] = group_best_combo  # Reemplazamos el combo final
            best_score = group_best_score

        else:
            break  # Stop evaluation if all combos in group failed

    return best_combo, best_score
    
    
@njit
def is_isolated(coords, neglog1mcdf0_matrix, threshold):
    """
    Checks that no immediate neighbor (in 8-connected directions) is
    significantly charged, i.e., -log(CDF(m=1)) < threshold.

    If any neighbor has -log(CDF(m=1)) < threshold,
    the pattern is considered NOT isolated.

    Returns True if isolated, False otherwise.
    """
    nrows, ncols = neglog1mcdf0_matrix.shape
    for (ix, jx) in coords:
        for di, dj in [(0, -1), (0, 1), (-1, 0), (1, 0),(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = ix + di, jx + dj
            if 0 <= rr < nrows and 0 <= cc < ncols:
                if (rr, cc) not in coords:
                    if neglog1mcdf0_matrix[rr, cc] < threshold:
                        return False
    return True


@njit
def detect_patterns(data_matrix, combos1_nb, combos2_nb, combos3_nb,
                    neglogcdf_matrix, neglog1mcdf0_matrix, threshold1=7.5, threshold2=8.5, threshold3=10):
    """
    Detects isolated patterns in the data matrix.
    Avoids double-counting pixels by using a mask of used pixels.
    Retains the last valid pattern found in each direction.
    """
    nrows, ncols = data_matrix.shape
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    dir_names = ["Row", "Col", "diag_r", "diag_l"]

    used_pixels = np.zeros((nrows, ncols), dtype=np.uint8)
    patterns = []

    for i in range(nrows):
        for j in range(ncols):
            if used_pixels[i, j] == 1:
                continue  # Already used by a previous pattern

            for d_idx, (di, dj) in enumerate(directions):
                last_valid_pattern = (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), -1.0, "", np.empty(0, dtype=np.float64))

                for length in (1, 2, 3):
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
                    for (r, c) in coords:
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

                    best_combo, best_score = cascade(
                        coords, combos_nb_, neglogcdf_matrix, thr
                    )
                    
       
                    val_isolated = is_isolated(coords, neglog1mcdf0_matrix, threshold1)
                    if best_combo[0] != -1 and is_isolated(coords, neglog1mcdf0_matrix, threshold1):
                        last_valid_pattern = (np.array(coords, dtype=np.int64), best_combo.copy(), best_score, dir_names[d_idx],
                                              np.array([data_matrix[r, c] for (r, c) in coords]))


                if last_valid_pattern[0].shape[0] > 0: 
                    patterns.append(last_valid_pattern)
                    for (r, c) in last_valid_pattern[0]:  
                        used_pixels[r, c] = 1

    return patterns


#######################
# SIMULATION CODE
#######################

def get_pixel_indices(x, y, pix_size=PIX_SIZE):
    row = int(np.floor(y / pix_size))
    col = int(np.floor(x / pix_size))
    return row, col

def apply_noise(n_e, sigma=0.15):
    noisy = np.random.normal(n_e, sigma)
    return noisy

def bin_pixels_vertically(pixel_counts, bin_size=100):
    binned_counts = {}
    for (row, col), count in pixel_counts.items():
        binned_row = row // bin_size
        binned_counts[(binned_row, col)] = binned_counts.get((binned_row, col), 0) + count
    return binned_counts


def simulate_events_with_diffusion(E, combos1_nb, combos2_nb, combos3_nb, N_events=1000, sigma_noise=0.15):
    """
    Simulates electron diffusion and detects the resulting patterns.

    - Tracks how many times each deposited charge q results in a given pattern p.
    - Computes the empirical probability p_diff(q → p) from simulation statistics.
    - Applies Gaussian noise to charges and vertical binning to simulate readout.
    - Collects histograms of pre- and post-diffusion charge values and transitions.
    
    Returns:
    - pre_counts: charge histogram before diffusion
    - post_counts: charge histogram after diffusion
    - changed: number of events where pre- and post- distributions differ
    - transition_counts: mapping from (q_before, q_after) to counts
    - pattern_stats: how often each (q, p) pair occurred
    - deposit_q: number of events simulated for each q
    """
    pattern_stats = defaultdict(int)
    deposit_q = defaultdict(int) 
    p_n = Probability_Neh_E(E, Niter=100000)
    n_bins = np.arange(1, len(p_n)+1)
    rng = np.random.default_rng()

    pre_counts = np.zeros(16, dtype=int)
    post_counts = np.zeros(16, dtype=int)
    changed = 0
    transition_counts = {}
    
    for _ in range(N_events):
        n_e = rng.choice(n_bins, p=p_n)  
        deposit_q[n_e] += 1 

        y0, x0, z0, _, _, _ = generate_event(E)
        sigma_xy = diffuse_event(z0, E)

        pre_pixel_counts = {}
        post_pixel_counts = {}
        post_pixel_counts_noise = {}
        pre_pixel = {}
        post_pixel_pre_dif = {}
        
        row, col = get_pixel_indices(x0, y0, PIX_SIZE)
        pre_pixel[(row, col)] = n_e
        post_pixel_pre_dif = bin_pixels_vertically(pre_pixel, bin_size=100)
        
        # Simulación de difusión
        for __ in range(n_e):
            xF = np.random.normal(x0, sigma_xy)
            yF = np.random.normal(y0, sigma_xy)
            row, col = get_pixel_indices(xF, yF, PIX_SIZE)
            pre_pixel_counts[(row, col)] = pre_pixel_counts.get((row, col), 0) + 1

        # Aplicar binning vertical
        post_pixel_counts = bin_pixels_vertically(pre_pixel_counts, bin_size=100)

        # Aplicar ruido y clasificación
        for pix, count in post_pixel_counts.items():
            post_pixel_counts_noise[pix] = apply_noise(count, sigma_noise)

        pre_pixel_distrib = np.zeros(16, dtype=int)
        post_pixel_distrib = np.zeros(16, dtype=int)
        
        # Contar valores antes y después de la difusión
        for count in post_pixel_pre_dif.values():
            pre_pixel_distrib[min(count, 15)] += 1
        for count in post_pixel_counts.values():
            post_pixel_distrib[min(count, 15)] += 1
            
        # Acumular conteos globales
        pre_counts += pre_pixel_distrib
        post_counts += post_pixel_distrib

        if not np.array_equal(pre_pixel_distrib, post_pixel_distrib):
            changed += 1

        # Contabilizar transiciones
        for pre_val, post_val in zip(pre_pixel_distrib, post_pixel_distrib):
            transition_counts[(pre_val, post_val)] = transition_counts.get((pre_val, post_val), 0) + 1
            
        # Construcción de matriz para detección de patrones
        if len(post_pixel_counts) == 0:
            continue  # No hay electrones, saltamos

        rows_all = [rc[0] for rc in post_pixel_counts_noise.keys()]
        cols_all = [rc[1] for rc in post_pixel_counts_noise.keys()]
        rmin, rmax = min(rows_all), max(rows_all)
        cmin, cmax = min(cols_all), max(cols_all)
        nrows_box = rmax - rmin + 1
        ncols_box = cmax - cmin + 1

        data_matrix = np.full((nrows_box, ncols_box), np.nan, dtype=np.float64)
        for (r, c), val in post_pixel_counts_noise.items():
            data_matrix[r - rmin, c - cmin] = val
            
        neglogcdf_matrix, neglog1mcdf0_matrix = precompute_neglog_cdf_matrices(data_matrix, sigma=sigma_noise)

        # Detectar patrones en la matriz resultante
        patterns = found = detect_patterns(data_matrix,
                                combos1_nb, combos2_nb, combos3_nb,
                                neglogcdf_matrix, neglog1mcdf0_matrix,
                                threshold1=3.5, threshold2=4.0, threshold3=5.5)

        # Contar los patrones detectados
        for (_, last_ok, _, _, _) in patterns:
            pat_str = "".join(str(x) for x in last_ok if x > 0)  # Convertir en string
            pattern_stats[(n_e, pat_str)] += 1

    return pre_counts, post_counts, changed, transition_counts, pattern_stats, deposit_q


def simulate_for_energy_range(energies, combos1_nb, combos2_nb, combos3_nb, N_events=1000, sigma_noise=0.15):
    """
    Simula sobre un rango de energías y acumula estadísticas.
    """
    total_pre = np.zeros(16, dtype=int)
    total_post = np.zeros(16, dtype=int)
    total_changed = 0
    total_transitions = {}
    
    total_pattern_stats = defaultdict(int)
    total_deposit_q = defaultdict(int)

    for E in energies:
        pre, post, changed, transitions, pattern_stats, deposit_q = simulate_events_with_diffusion(
            E, combos1_nb, combos2_nb, combos3_nb, N_events, sigma_noise)
        total_pre += pre
        total_post += post
        total_changed += changed
        for key, value in transitions.items():
            total_transitions[key] = total_transitions.get(key, 0) + value
            
        for key, val in pattern_stats.items():
            total_pattern_stats[key] += val
        for q, count in deposit_q.items():
            total_deposit_q[q] += count

    return total_pre, total_post, total_changed, total_transitions, total_pattern_stats, total_deposit_q


from itertools import product

def calculate_N_corr(lambda_poisson, N_sel, p_diff):
    """
    Computes the corrected expected number of pixels N_corr(p) for each pattern p
    based on a Poisson distribution and the diffusion probabilities p_diff(q → p).

    Parameters:
    - lambda_poisson: Poisson mean (λ) for the background model.
    - N_sel: total number of selected pixels.
    - p_diff: dictionary with diffusion probabilities {(q, p): p_diff(q → p)}.

    Returns:
    - N_corr: dictionary mapping each pattern p to the corrected expected number of pixels.
    """
    N_corr = {}

    # Compute expected Poisson counts for q = 1, 2, 3, 4, 5
    N_poisson = {q: poisson.pmf(q, lambda_poisson) * N_sel for q in range(1, 6)} 

    # Direct Poisson contribution for simple patterns p = q
    for q in range(1, 6):
        N_corr[str(q)] = N_poisson[q] # e.g., N_corr('1') = N_poisson(1)

    # Correction for composite patterns (e.g., '11', '21', etc.)
    for p in set(p for (_, p) in p_diff.keys()):
        if p not in N_corr:
            N_corr[p] = 0  # Initialize if not already in dictionary

        # Add contributions from diffusion of q ≠ p
        for q in range(1, 6):
            if (q, p) in p_diff:
                N_corr[p] += p_diff[(q, p)] * N_poisson[q]

        # Add direct Poisson contribution if p is a composite pattern
        if len(p) > 1:
            p_elements = list(map(int, p))
            poisson_prob = np.prod([poisson.pmf(e, lambda_poisson) for e in p_elements])
            N_corr[p] += poisson_prob * N_sel

    return N_corr

    

def compute_p_diff(N_events=50000, sigma_noise=0.16):
    """
    Runs full simulation over a range of energies to estimate the diffusion probabilities p_diff(q → p).

    Returns:
    - p_diff: dictionary with diffusion probabilities for each (q, p) pair
    """
    
    combos1_nb = pythonlist_to_numba(make_pattern_order(1))
    combos2_nb = pythonlist_to_numba(make_pattern_order(2))
    combos3_nb = pythonlist_to_numba(make_pattern_order(3))

    energies = np.linspace(3.75, 20, 50)
    pre, post, changed, transitions, total_stats, total_deposit_q = simulate_for_energy_range(
        energies, combos1_nb, combos2_nb, combos3_nb, N_events=N_events, sigma_noise=sigma_noise)

    print(f"Simulated events: {sum(pre)}")
    print(f"Events with changed category after diffusion: {changed}")
    
    for (pre_val, post_val), count in transitions.items():
        print(f"Pixels with {pre_val}e⁻ → {post_val}e⁻: {count} times")
        
    # Format y-axis in millions
    formatter = FuncFormatter(lambda x, _: f'{x*1e-6:.1f}')

    plt.figure(figsize=(12, 10))
    plt.hist(range(16), weights=pre, bins=np.arange(17) - 0.5, alpha=0.7, label="Before diff.", color='blue', histtype='step', linewidth=2.5)
    plt.hist(range(16), weights=post, bins=np.arange(17) - 0.5, alpha=0.7, label="After diff.", color='red', histtype='step', linewidth=2.5)
    plt.xlabel(r"charge (e$^{-}$)")
    plt.ylabel(r"count ($\times 10^6$)") 
    plt.legend()
    plt.grid(False)
    plt.xlim(0, 10)
    plt.xticks()
    plt.yticks()
    ax = plt.gca()
    set_axis_labels_aligned(ax)
    ax.set_ylabel(ax.get_ylabel(), verticalalignment='top', y=0.8, labelpad=25)
    plt.tight_layout()
    plt.show()
    
    print("Detected pattern statistics:\n(q, pattern_str) => count")
    for key, val in total_stats.items():
        print(f"  {key} => {val}")         

    # Compute p_diff(q → p)
    p_diff = {}
    for (q, pat_str), count in total_stats.items():
        if q in total_deposit_q and total_deposit_q[q] > 0:
            p_diff[(q, pat_str)] = count / total_deposit_q[q]
            print(f"Total deposits of {q}: {total_deposit_q[q]}")

    return p_diff


def main():
    json_path = Path("p_diff.json")

    # 1) Load or compute p_diff
    if json_path.exists():
        with json_path.open() as f:
            tmp = json.load(f)
        # Reconstruct dictionary keys as tuple (q, p)
        p_diff = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in tmp.items()}
        print("p_diff.json found and loaded.")
    else:
        print("p_diff.json not found → running simulation...")
        p_diff = compute_p_diff(N_events=50_000, sigma_noise=0.16)

        # Save to disk
        p_diff_serializable = {f"{q}|{pat}": prob for (q, pat), prob in p_diff.items()}
        with json_path.open("w") as f:
            json.dump(p_diff_serializable, f, indent=2)
        print("File p_diff.json created.")

    # 2) Display p_diff
    print("\n--- Diffusion probabilities p_diff(q → p) ---")
    for (q, p), prob in sorted(p_diff.items()):
        print(f"  p_diff({q} → {p}) = {prob:.6f}")

    # 3) Plot for each q
    q_values = sorted({q for (q, _) in p_diff.keys()})
    for q in q_values:
        patterns = [p for (qq, p) in p_diff if qq == q]
        probs =   [p_diff[(q, p)] for p in patterns]

        plt.figure(figsize=(12, 10))
        plt.bar(patterns, probs, alpha=0.7, color='blue')
        plt.xlabel("Pattern detected")
        plt.ylabel(f"p_diff({q} → p)")
        set_axis_labels_aligned()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
