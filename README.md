# Data Selection and Visualization

This repository contains Python scripts developed for CCD data cleaning, masking, validation and pattern recognition for low-energy charge detection in dark matter experiments (DAMIC-M).

## Scripts Overview

1. **histogram_noise_combined.py**
   - Combines noise histograms from different modules for comparison.
   - Features Gaussian fitting capabilities to analyze noise characteristics.

2. **histogram_SCurve_plots.py**
   - Generates SCurve plots from `.root` files.
   - Applies Gaussian fits to extract performance metrics from the SCurve distributions.

3. **hitsperpixel.py**
   - Draws and saves histograms of hits per pixel.
   - Capable of applying additional axis and adjusting visual elements like color scales.

4. **masked_noisy_stuck_pix.py**
   - Analyzes and visualizes distributions of masked, noisy, and stuck pixels.
   - Identifies problematic pixels and provides detailed statistics.

5. **plotsreverse.py**
   - Performs differential analysis between forward and reverse bias conditions.
   - Highlights shifts in threshold and noise values across conditions.

6. **save_histograms.py**
   - General-purpose script to save histograms from `.root` files.
   - Includes functionality to adjust visual settings and manage output files efficiently.

### Usage
Specified in each script, for example:
```bash
python histogram_noise_comined.py <input_file.root> <output_directory>
