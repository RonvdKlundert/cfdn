# CFDN
Tools for visualization and formatting of divisive normalization connective fields fits

How is use it:

1) linear detrend data and inspect using prepoc_baseline_hcp.ipynb
2) run fit_hcp_prf.py with sbatch -> submissions folder
3) 'format.py prf' to format the pRF fits
4) CF_processing_HCP999999.ipynb notebook to create CF design matrices and inspect the outcomes
5) run fit_hcp_cf.py with sbatch -> submissions folder
6) 'format.py cf' to format the cf fits and add x, y values based on vertex centre pRF fits
7) run viz.py to look at the results
