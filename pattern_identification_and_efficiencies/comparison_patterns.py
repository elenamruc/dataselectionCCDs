#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz
"""

from itertools import product
from math import sqrt
from scipy.stats import poisson

def make_pattern_order(n):
    """
    Generate pattern combinations where each element is in [1..5],
    filtered by sum(c) <= 5, sorted by total sum.
    """
    combos = []
    for c in product(range(1, 6), repeat=n):
        if sum(c) <= 5:
            combos.append((sum(c), c))
    combos.sort(key=lambda x: x[0])
    return [c for (_, c) in combos]

######################################
# 1) Read efficiency probabilities from files 
######################################

def read_probabilities(ruta_txt):
    """
    Reads a file with columns:
      RealPattern   DetectedPattern   p(D->E)   recall(D)   misID(D)
    Returns:
      p_detect: dict[D][E] = p(D->E)
      recall_map: dict[D] = recall(D)
      misid_map: dict[D] = misID(D)
    where D and E are tuples (e.g. (1,), (1,1), etc.).
    """
    p_detect = {}
    p_err_detect = {}
    recall_map = {}
    misid_map = {}

    with open(ruta_txt, "r", encoding="utf-8") as f:
        next(f, None) # skip header
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            # Example line:
            # (1, 1, 1)   (1,)    0.009882    0.910954    0.000028
            real_pat_str, det_pat_str, pDE_str, err_pDE_str, recall_str, err_rec_str, misid_str, err_misid_str = linea.split("\t")

            pDE = float(pDE_str)
            err_pDE = float(err_pDE_str)
            recall_ = float(recall_str)
            mis_ = float(misid_str)

            real_pat = eval(real_pat_str)
            det_pat  = eval(det_pat_str)

            if real_pat not in p_detect:
                p_detect[real_pat] = {}
                p_err_detect[real_pat] = {}
                
            p_detect[real_pat][det_pat] = pDE
            p_err_detect[real_pat][det_pat] = err_pDE

            recall_map[real_pat] = recall_
            misid_map[real_pat]  = mis_

    return p_detect, p_err_detect, recall_map, misid_map


######################################
# 2) Number of patterns by Poisson  
######################################

def prob_poisson_de_patron(D, lam, n_cols=3):
    """
    Computes the probability of observing a pattern D = (k1, k2, ...)
    in 1 pixel (with n_cols "subpixels"), assuming Poisson(lam) per subpixel.

    If D uses fewer than n_cols subpixels, remaining are assumed to have k=0.
    """
    used_cols = len(D)
    p = 1.0
    for ki in D:
        p *= poisson.pmf(ki, lam)
    missing = n_cols - used_cols
    if missing > 0:
        p *= (poisson.pmf(0, lam))**missing
    return p

def calculate_events_per_pattern_detected(ruta_txt, lam, N_pix, n_cols=3):
    """
    Calculates the expected number of detected patterns using:
        N_correct(E) = N_pix * sum_{D=E}[ p_real(D)*p(D->E) ]
        N_wrong(E)   = N_pix * sum_{D!=E}[ p_real(D)*p(D->E) ]
        N_total(E)   = N_correct + N_wrong
    And statistical variance due to error in p(D->E):
        var_stat(E) = sum_D [ (N_pix * p_real(D) * err_p(D->E))^2 ]

    Returns:
      central_results: dict[E] = [N_correct, N_wrong, N_total]
      var_stat: dict[E] = statistical variance of N_total(E)
    """
    p_detect, p_err_detect, recall_map, misid_map = leer_probabilidades(ruta_txt)

    # 1) Compute real probability p_real(D) using Poisson
    p_real = {}
    for D in p_detect.keys():
        p_real[D] = prob_poisson_de_patron(D, lam, n_cols=n_cols)

    # 2) Collect all possible detected patterns
    uniq_patter_detected = set()
    for D in p_detect:
        for E in p_detect[D]:
            uniq_patter_detected.add(E)

    # 3) Initialize accumulators
    central_results = {}
    var_stat = {}
    for E in uniq_patter_detected:
        central_results[E] = [0.0, 0.0, 0.0]
        var_stat[E] = 0.0

    # 4) Accumulate contributions
    for D, dict_DE in p_detect.items():
        pD = p_real.get(D, 0.0)
        if pD <= 0:
            continue
        for E, pDE in dict_DE.items():
            contrib = pD * pDE * N_pix
            if E == D:
                central_results[E][0] += contrib  # Correct
            else:
                central_results[E][1] += contrib  # Wrong
            central_results[E][2] += contrib      # Total

            err_pDE = p_err_detect.get(D, {}).get(E, 0.0)
            if err_pDE is None:
                err_pDE = 0.0
            var_stat[E] += (N_pix * pD * err_pDE)**2

    return central_results, var_stat


def events_with_error(ruta_txt, lam, err_lam, N_pix):
    """
    For each detected pattern E, returns:
      (N_total, errN_total) with error combined from:
        - Lambda error: central diff ~ 0.5 * |N(lam+err) - N(lam-err)|
        - Statistical error from uncertainty in p(D->E)

    Returns:
      dict[E] = (N_total, total_error)
    """
    res_central, var_stat_central = calculate_events_per_pattern_detected(ruta_txt, lam, N_pix)

    lam_plus  = lam + err_lam
    lam_minus = max(lam - err_lam, 0.0)
    res_up, _   = calculate_events_per_pattern_detected(ruta_txt, lam_plus, N_pix)
    res_down, _ = calculate_events_per_pattern_detected(ruta_txt, lam_minus, N_pix)

    final_dict = {}
    patrones_todos = set(res_central.keys()) | set(res_up.keys()) | set(res_down.keys())
    for E in patrones_todos:
        N_central = res_central.get(E, [0.0, 0.0, 0.0])[2]
        N_up      = res_up.get(E, [0.0, 0.0, 0.0])[2]
        N_down    = res_down.get(E, [0.0, 0.0, 0.0])[2]

        # Lambda error
        err_lambda = 0.5 * abs(N_up - N_down)

        # Statistical error p(D->E)
        var_stat = var_stat_central.get(E, 0.0)
        err_stat = sqrt(var_stat)

        # Total error
        err_total = sqrt(err_lambda**2 + err_stat**2)
        final_dict[E] = (N_central, err_total)

    return final_dict


######################################
# 3) Definition of datasets values
######################################

param_sets = [
    {
        "fecha": "14/10/2024",
        "lambdas_103":     [0.0002815, 0.0002345, 0.0002592, 0.0003058],
        "lambdas_err_103": [0.00005598, 0.00004604, 0.00005881, 0.00005128],
        "sigma_e_103":     [0.155, 0.159, 0.162, 0.167],
        "N_total": 47376000,  # 6300 * 7520
        "N_selc_pix": [
            47376000 - 3835241,
            47376000 - 2220083,
            47376000 - 8559317,
            47376000 - 4713324
        ],
        "rutas_txt": [
            "results_efficiency_blinded/results_efficiencies_full_0.0002815_0.155.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002345_0.159.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002592_0.162.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0003086_0.167.txt",
        ],
    },
    {
        "fecha": "28/10/2024",
        "lambdas_103":     [0.0002808, 0.0002215, 0.0002708, 0.0003119],
        "lambdas_err_103": [0.00003991, 0.00003707, 0.00004203, 0.00003733],
        "sigma_e_103":     [0.153, 0.158, 0.161, 0.164],
        "N_total": 6300 * 17952,  
        "N_selc_pix": [
            (6300*17952) - 11230913,
            (6300*17952) - 13026489,
            (6300*17952) - 20249576,
            (6300*17952) - 6374802
        ],
        "rutas_txt": [
            "results_efficiency_blinded/results_efficiencies_full_0.0002805_0.153.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002216_0.158.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002709_0.161.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0003114_0.164.txt",
        ],
    },
    {
        "fecha": "19/11/2024",
        "lambdas_103":     [0.0002913, 0.0002324, 0.0002892, 0.000323],
        "lambdas_err_103": [0.00003630, 0.00003650, 0.00003643, 0.00003591],
        "sigma_e_103":     [0.153, 0.157, 0.160, 0.164],
        "N_total": 6300 * 22176,  
        "N_selc_pix": [
            (6300*22176) - 13183218,
            (6300*22176) - 14721216,
            (6300*22176) - 25651122,
            (6300*22176) - 8346097
        ],
        "rutas_txt": [
            "results_efficiency_blinded/results_efficiencies_full_0.000291_0.153.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002317_0.157.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002894_0.160.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0003211_0.164.txt",
        ],
    },
    {
        "fecha": "17/12/2024",
        "lambdas_103":     [0.0002905, 0.0002321, 0.0002743, 0.0003224],
        "lambdas_err_103": [0.00003738, 0.00003786, 0.00003479, 0.00003298],
        "sigma_e_103":     [0.153, 0.155, 0.158, 0.164],
        "N_total": 6300 * 16096,  
        "N_selc_pix": [
            (6300*16096) - 9650087,
            (6300*16096) - 10482327,
            (6300*16096) - 18817613,
            (6300*16096) - 6457755
        ],
        "rutas_txt": [
            "results_efficiency_blinded/results_efficiencies_full_0.0002906_0.153.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002321_0.155.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0002744_0.158.txt",
            "results_efficiency_blinded/results_efficiencies_full_0.0003217_0.164.txt",
        ],
    },
]

param_sets_ub = [
    {
        "fecha": "06/10/2024",
        "lambdas_103":     [0.00036, 0.000301, 0.000319, 0.0004006],
        "lambdas_err_103": [0.0000475, 0.00005989, 0.00005366, 0.00005993],
        "sigma_e_103":     [0.152, 0.156, 0.159, 0.163],
        "N_total": 38304000, 
        "N_selc_pix": [
            38304000 - 1686474,
            38304000 - 3435622,
            38304000 - 6956058,
            38304000 - 4828293
        ],
        "rutas_txt": [
            "results_efficiency_unblinded/results_efficiencies_full_0.0003602_0.152.txt",
            "results_efficiency_unblinded/results_efficiencies_full_0.0003013_0.156.txt",
            "results_efficiency_unblinded/results_efficiencies_full_0.0003203_0.159.txt",
            "results_efficiency_unblinded/results_efficiencies_full_0.0004006_0.163.txt",
        ],
    },
]



def main():

    patterns_to_show = [
        (1,), (2,), (3,),
        (1,1), (1,2), (1,3),
        (1,1,1), (1,1,2)
    ]

    patterns_global = {}

    for scenario in param_sets:
        date_label          = scenario["fecha"]
        lambdas        = scenario["lambdas_103"]
        lambdas_err    = scenario["lambdas_err_103"]
        rutas_txt      = scenario["rutas_txt"]
        N_selc_pix     = scenario["N_selc_pix"]

        patterns_by_date = {}

        print("\n" + "="*80)
        print(f"Date: {date_label}")

        # ---------- SUB-CONFIGURATIONS ----------
        for i in range(len(lambdas)):
            final_dict = events_with_error(
                rutas_txt[i], lambdas[i], lambdas_err[i], N_selc_pix[i]
            )

            diffusion_contributions   = {}      # stores ΔN_diff in case it's used later

            # 1) Read the same .txt to get recall_map (= efficiency per pattern)
            p_detect, p_err_detect, recall_map, misid_map = read_probabilities(rutas_txt[i])
        
            print(f"    λ = {lambdas[i]:.6f} ± {lambdas_err[i]:.6f},  N_pix = {N_selc_pix[i]}")

            # Print only selected patterns from THIS sub-config
            for E in patterns_to_show:
                if E in final_dict:
                    N, err = final_dict[E]
                    print(f"    {E} : {N:.3e} ± {err:.3e}")

            # Accumulate for this DATE
            for E, (N, err) in final_dict.items():
                if E not in patterns_by_date:    
                    patterns_by_date[E] = [0.0, 0.0]
                patterns_by_date[E][0] += N
                patterns_by_date[E][1] += err**2

                # Accumulate in GLOBAL results
                if E not in patterns_global:
                    patterns_global[E] = [0.0, 0.0]
                patterns_global[E][0] += N
                patterns_global[E][1] += err**2


        # ---------- DATE SUMMARY ----------
        print("\nSummary for this date (sum of all 4 sub-configurations):")
        for E in patterns_to_show:
            N, err2 = patterns_by_date.get(E, (0.0, 0.0))
            if N:
                print(f"{E} : {N:.3e} ± {sqrt(err2):.3e}")    
            

    # ---------- GLOBAL SUMMARY----------
    print("\n" + "="*80)
    print("TOTAL for all dates")
    for E in patterns_to_show:
        N, err2 = patterns_global.get(E, (0.0, 0.0))    
        if N:
            print(f"{E} : {N:.3e} ± {sqrt(err2):.3e}")



if __name__ == "__main__":
    main()
