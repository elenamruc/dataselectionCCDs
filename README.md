# Data Selection and Pattern Analysis for CCDs (DAMIC-M)

This repository contains all scripts used for the **selection of low-energy events**, **masking**, and **pattern identification** in CCD images from dark matter experiments such as **DAMIC-M**. It includes tools for detection, efficiency analysis, confusion matrices, and visualization of simulated or real data.

---

## Repository Structure

### `results_analysis/`
Scripts for **aggregating and analyzing results** from multiple datasets or dates.

- `results_data_selection.py`: 
---
### `pattern_id/`
Contains all core scripts for **pattern detection** in CCD images after masking.

- `pattern_identification.py`: Main pipeline to detect isolated single and multi-pixel patterns using CDFs and a cascade method.
- `efficiencies_patterns.py`: Computes **recall**, **precision**, and **misidentification rate** for each pattern, using weighted counts and propagating uncertainties. Saves results to file.
- `comparison_patterns.py`: Calculates the expected number of patterns based on: A Poisson distribution for the number of unmasked pixels (N_selc_pix) and a diffusion model that gives the probability that a charge q generates a given pattern after diffusion.

---

## Scripts Overview

### `pattern_identification.py`

**Main script for detection**:
- Input: FITS image with masking extensions.
- Detects isolated charge clusters (1–3 pixels).
- Uses precomputed CDF tables and `-log(CDF)` scoring.
- Enforces constraints: score threshold, local isolation, shape uniqueness.
- Output: pattern histograms and charge distributions.

```bash
python3 pattern_identification.py -f image1.fits image2.fits
