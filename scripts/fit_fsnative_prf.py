import sys
sys.path.append('/home/klundert/cfdn/prfpy_cfdn/')
import os
import numpy as np
#import preprocess
#import cortex as cx
import numpy as np
import scipy as sp
import nilearn as nl
from nilearn.surface import load_surf_data
import os, shutil, urllib.request
#import cortex as cx
from matplotlib import rc
import nibabel as nb
from nibabel import cifti2
import h5py
import matplotlib.pyplot as plt
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.io import loadmat
from scipy.ndimage import median_filter, gaussian_filter, binary_propagation
from preprocess import get_cortex
from preprocess import split_given_size
import yaml
import numpy as np
import scipy as sp
import nilearn as nl
from nilearn.surface import load_surf_data
import os, shutil, urllib.request
import cortex as cx
from matplotlib import rc
import nibabel as nb
from nibabel import cifti2
import h5py
import matplotlib.pyplot as plt
import prfpy
from scipy.io import loadmat
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter, CSS_Iso2DGaussianFitter




########################################################################################
# set parameters
########################################################################################


fit_hrf = True
constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]


id = int(sys.argv[1])
n_jobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
fold = int(sys.argv[4])

sub = str(id+1)




new_dms = np.load('/home/klundert/cfdn/data/CF_fit_utils/prf_dm.npy')[5:,:,:]

prf_stim = PRFStimulus2D(screen_size_cm=69, 
                         screen_distance_cm=220, 
                         design_matrix=new_dms.T, 
                         TR=1.5)

grid_nr = 20
max_ecc_size = prf_stim.screen_size_degrees/2.0
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to set up parameter bounds in iterfit
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees


refit_mask = np.load('/home/klundert/cfdn/data/CF_fit_utils/refit_mask.npy')

if fold == 0:
    mydat_train_stim = np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold1_detrend_sub-0{sub}_psc_fsnative.npy'))[refit_mask]
    mydat_test_stim = np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold2_detrend_sub-0{sub}_psc_fsnative.npy'))[refit_mask]
else:
    mydat_train_stim = np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold2_detrend_sub-0{sub}_psc_fsnative.npy'))[refit_mask]
    mydat_test_stim = np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold1_detrend_sub-0{sub}_psc_fsnative.npy'))[refit_mask]




fitsize = np.ceil(len(mydat_train_stim)/350).astype(int)
print(f'fitsize is {fitsize}')

mydat_train = split_given_size(mydat_train_stim, fitsize)[slice_n]
mydat_test = split_given_size(mydat_test_stim, fitsize)[slice_n]

# mydat_test = split_given_size(mydat_test_stim, 3294)[slice_n]

#model=CFGaussianModel(train_stim)

# Define grid of parameters to search over

surround_amplitude_grid=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
surround_size_grid=np.array([3,5,8,12,18], dtype='float32')
neural_baseline_grid=np.array([0,1,10,50,100], dtype='float32')
surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')

# define the bounds of the grid

gauss_grid_bounds = [(0,1000)] #only prf amplitudes between 0 and 1000
norm_grid_bounds = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000, only neural baseline values between 0 and 1000


# define bounds for the iterative fit

gauss_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (eps, 1.5*ss),  # prf size
                (-1000, 1000),  # prf amplitude
                (0, 0)]  # bold baseline



norm_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (eps, 1.5*ss),  # prf size
                (-1000, 1000),  # prf amplitude
                (0, 0),  # bold baseline
                (0, 1000),  # surround amplitude
                (eps, 3*ss),  # surround size
                (0, 1000),  # neural baseline  7 B
                (1e-6, 1000)]  # surround baseline 8 D

if fit_hrf:
    norm_bounds += [(0,10),(0,0)]
    gauss_bounds += [(0,10),(0,0)]


# Define the model and fitter etc
gg = Iso2DGaussianModel(stimulus=prf_stim,
                        filter_predictions=False,
                        filter_type='dc')

gf_P = Iso2DGaussianFitter(data=mydat_train, model=gg, n_jobs=n_jobs, fit_hrf=fit_hrf)
gf_P.grid_fit(ecc_grid=eccs,
                 polar_grid=polars,
                 size_grid=sizes, 
                 n_batches=60,
                 fixed_grid_baseline=0,
                 grid_bounds=gauss_grid_bounds)


print('finished gridsearch gauss')
np.save(f'/home/klundert/fsnative_data2/gauss_grid_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_P.gridsearch_params)


print('starting iterative fit for gauss')
gf_P.iterative_fit(rsq_threshold=-1, verbose=True, bounds=gauss_bounds, constraints=[],  xtol=1e-5, ftol=1e-5)

gf_P.crossvalidate_fit(mydat_test, single_hrf=True)


print('finished iterative fit gauss')
np.save(f'/home/klundert/fsnative_data2/gauss_prf_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_P.iterative_search_params)


############################
# ftting norm model
############################


# Define the model and fitter etc
gg_norm = Norm_Iso2DGaussianModel(stimulus=prf_stim,
                                    filter_predictions=False,
                                    filter_type='dc',
                                    )

gf_norm = Norm_Iso2DGaussianFitter(data=mydat_train,
                                   model=gg_norm,
                                   n_jobs=n_jobs,
                                   fit_hrf=fit_hrf,
                                   previous_gaussian_fitter=gf_P)

gf_norm.grid_fit(surround_amplitude_grid,
                         surround_size_grid,
                         neural_baseline_grid,
                         surround_baseline_grid,
                         verbose=True,
                         n_batches=60,
                         fixed_grid_baseline=0,
                         rsq_threshold=0.01,
                         grid_bounds=norm_grid_bounds)

print('finished gridsearch for DN')
np.save(f'/home/klundert/fsnative_data2/norm_grid_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_norm.gridsearch_params)



if fit_hrf:
    A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
else:
    A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0]])

constraints_norm.append(LinearConstraint(A_ssc_norm,
                                            lb=0,
                                            ub=+inf))


constraints_norm.append(LinearConstraint(A_ssc_norm, lb=0, ub=+inf))


print('starting DN fit')
gf_norm.iterative_fit(rsq_threshold=0.1, verbose=True, bounds=norm_bounds, constraints=constraints_norm, xtol=1e-5, ftol=1e-5)

gf_norm.crossvalidate_fit(mydat_test, single_hrf=True)


print('finished iterative fit DN')

np.save(f'/home/klundert/fsnative_data2/DN_prf_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_norm.iterative_search_params)
