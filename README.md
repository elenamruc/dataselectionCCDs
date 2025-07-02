# Data Selection and Pattern Analysis for CCDs (DAMIC-M)

This repository contains all scripts used for the **selection of low-energy events**, **masking**, and **pattern identification** in CCD images from dark matter experiments such as **DAMIC-M**. It includes tools for detection, efficiency analysis, confusion matrices, and visualization of simulated or real data.

---

## Repository Structure

### `data_selection_masks/`
Scripts for identifying and masking instrumental artifacts and noise sources in CCD images. Each file defines a specific masking algorithm. The final script summarizes masking performance.

- **`mask_HC_and_OD.py`**  
  Masks **Hot Columns (HC)** with persistently high charge and **Overdensity (OD)** regions with statistically significant excess pixel activations.
- **`mask_CTI.py`**  
  Masks **Charge Transfer Inefficiency (CTI)** artifacts, which create vertical or horizontal trails in the image due to incomplete charge transfer during readout.
- **`mask_charge_multiplicity.py`**  
  Masks pixels or areas based on **high local multiplicity of low-energy charges**.
- **`mask_isolated.py`**  
  Masks **isolated pixel columns** (between two masked columns).
- **`results_data_selection.py`**  
  Applies all previous masks and computes:
  - The **residual dark current** in clean regions.
  - The **efficiency of each mask** and the **cumulative masking efficiency**.
  - Outputs useful summaries for each run or date.

---
### `pattern_identification_and_efficiencies/`
Contains all core scripts for **pattern detection** in CCD images after masking.

- **`pattern_identification.py`**:
  Main pipeline to detect isolated single and multi-pixel patterns using CDFs and a cascade method.
- **`efficiencies_patterns.py`**:
  Computes **recall**, **precision**, and **misidentification rate** for each pattern, using weighted counts and propagating uncertainties. Saves results to file.
- **`comparison_patterns.py`**:
  Calculates the expected number of patterns based on: A Poisson distribution for the number of unmasked pixels (N_selc_pix)
- **`diffusion_probabilities/`**:
  Contains scripts and resources for simulating **charge diffusion** in a CCD and modeling the probability that a deposited charge \( q \) gives rise to a specific pattern \( p \).
    - **`p100K.npz`**: Precomputed pair-creation probabilities used for charge generation.
    - **`charge_ionization.py`**: Generates the number of e⁻–h⁺ pairs for a given energy.  
    - **`diffusion.py`**: Implements the depth- and energy-dependent diffusion model used in the simulation.  
    - **`simulator.py`**: Runs full Monte Carlo simulation to compute and save \( p_{\text{diff}}(q \rightarrow p) \) as `p_diff.json`.
    - **`p_diff.json`**: Output file with estimated diffusion probabilities.
    - **`diff_pattern.py`**: Estimates expected detected patterns from true 2e⁻ and 3e⁻ events using `p_diff.json`.

## Scripts Overview

Most scripts in this repository are executed using a command similar to:

```bash
python3 pattern_identification.py -f image1.fits image2.fits ...

You can provide as many FITS files as needed.
When additional parameters or options are required (e.g., output path, thresholds, minimum charge, JSON inputs), each script explicitly specifies them in the help message.
