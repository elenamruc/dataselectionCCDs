#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last revision: 01/06/2025
@author: elenamunoz


Compute the expected number of detected patterns E produced by true
2-electron and 3-electron deposits, given:

  • p_diff(q → pattern)         (JSON file)
  • the observed counts of the single-pixel patterns [2] and [3]

No detection efficiencies are used.

Example:
--------
python3 expected_patterns_from_observed.py \
        --p_diff p_diff.json \
        --obs2 4 \
        --obs3 0
"""

import json
import argparse
import re
from collections import defaultdict


_pat_rx = re.compile(r"\((.*?)\)")

def pat_to_str(pat_txt: str) -> str:
    """
    Convert tuple-style pattern text to a compact string.
    '(2,)'   -> '2'
    '(1, 1)' -> '11'
    """
    txt = pat_txt.strip()
    m = _pat_rx.search(txt)
    if not m:                         # already without parentheses
        return txt.replace(",", "").replace(" ", "")
    return m.group(1).replace(",", "").replace(" ", "")

def compute_expected(Nq_real: dict, p_diff: dict):
    """
    Return a dictionary  res[pattern] = (ΔN_q=2, ΔN_q=3, ΔN_tot)

    Parameters
    ----------
    Nq_real : {2: N_real(2 e-), 3: N_real(3 e-)}
    p_diff  : {(q, pattern_str): probability}

    Detection efficiency is intentionally omitted.
    """
    res = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (q, pat), prob in p_diff.items():
        if q not in Nq_real:
            continue
        dN = Nq_real[q] * prob            # expected contribution
        if q == 2:
            res[pat][0] += dN
        if q == 3:
            res[pat][1] += dN
        res[pat][2] += dN                 # total from both q values
    return res

# ---------- main -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Expected multi-pixel patterns")
    ap.add_argument("--p_diff", default="p_diff.json",
                    help="JSON with probabilities p_diff(q|pattern)")
    ap.add_argument("--obs2", type=float, required=True,
                    help="observed count of single-pixel pattern [2]")
    ap.add_argument("--obs3", type=float, required=True,
                    help="observed count of single-pixel pattern [3]")
    args = ap.parse_args()

    # --- load p_diff --------------------------------------------------
    with open(args.p_diff) as f:
        raw = json.load(f)
    # keys are stored as "q|pattern"
    p_diff = {(int(k.split("|")[0]), k.split("|")[1]): v
              for k, v in raw.items()}

    # --- retrieve p22 and p33 ----------------------------------------
    p22 = p_diff.get((2, "2"))
    p33 = p_diff.get((3, "3"))
    if p22 is None or p33 is None:
        raise RuntimeError("Missing p_diff(2→2) or p_diff(3→3) in the JSON file.")

    # --- infer the *true* number of 2-e and 3-e deposits -------------
    N2_real = args.obs2 / p22
    N3_real = args.obs3 / p33

    print("# Estimated true deposits (no recall applied)")
    print(f"  q = 2 e- :  N_real = {N2_real:.3e}   (obs={args.obs2}, p22={p22:.3f})")
    print(f"  q = 3 e- :  N_real = {N3_real:.3e}   (obs={args.obs3}, p33={p33:.3f})\n")

    # --- expected contributions to every pattern --------------------
    res = compute_expected({2: N2_real, 3: N3_real}, p_diff)

    header = f"{'Pattern':>7s} | {'ΔN_q=2':>12s} | {'ΔN_q=3':>12s} | {'ΔN_tot':>12s}"
    print(header)
    print("-" * len(header))
    for pat, (d2, d3, tot) in sorted(res.items(), key=lambda x: -x[1][2]):
        if tot < 1e-3:              # skip negligible contributions
            continue
        print(f"{pat:>7s} | {d2:12.3e} | {d3:12.3e} | {tot:12.3e}")

if __name__ == "__main__":
    main()

