## CFDN repository
This repository contains tools used for formatting CF and pRF fits after fitting on HPC and contains necessary tools to create design matrices for CF fitting.

## installation
To use this repository you install using either run
pip install -e . OR python setup.py develop

You will need [pycortex](https://github.com/gallantlab/pycortex) to use this repository and you also need [cfpy](https://github.com/RonvdKlundert/cfpy), a fork of prfpy that contains the normalization conenctive field model


## Policy & To Do

- [x] install using `python setup.py develop` and `python -m pip install -e .`
- [x] add Subsurface functionality to this repo, so [cfpy](https://github.com/RonvdKlundert/cfpy) can lose the pycortex depency
- [x] create interactive visualization for CF and pRF results `viz.py`
  - [] add connective field visualization to `viz.py`
- [] recompile `prepreocess.py`, this code works but is far from good
- [] upload data to figshare so when the paths to data dont exist it will download from figshare
- [] make how to use each functionality more intuitive?
- [] ..




How I use it currently:

1) linear detrend data and inspect using prepoc_baseline_hcp.ipynb
2) run fit_hcp_prf.py with sbatch -> submissions folder
3) 'format.py prf' to format the pRF fits
4) CF_processing_HCP999999.ipynb notebook to create CF design matrices and inspect the outcomes
5) run fit_hcp_cf.py with sbatch -> submissions folder
6) 'format.py cf' to format the cf fits and add x, y values based on vertex centre pRF fits
7) run viz.py to look at the results
