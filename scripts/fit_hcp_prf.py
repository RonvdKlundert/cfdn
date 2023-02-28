import sys
import yaml

import numpy as np
from scipy.optimize import LinearConstraint

from prfpy.rf import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter

from cftools.preprocess import split_given_size 

# get arguments from command line

id = int(sys.argv[1])
n_jobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
fold = int(sys.argv[4])

sub = str(id+1)


########################################################################################
# set parameters from yaml file
########################################################################################

yaml_dir = '/home/klundert/cfdn/analysis_config.yml'

with open(yaml_dir, 'r') as f:
    analysis_info = yaml.safe_load(f)

# import yaml settings
data_dir = analysis_info['data_dir']
save_dir_cf = analysis_info['save_dir_cf']
save_dir_pRF = analysis_info['save_dir_pRF']
threads = analysis_info['threads']
n_slices = analysis_info['n_slices']
n_batches = analysis_info['n_batches']
discard_volumes = analysis_info['discard_volumes']
xtol = float(analysis_info['xtol'])
ftol = float(analysis_info['ftol'])
constraints = analysis_info['constraints']
rsq_threshold = analysis_info['rsq_threshold']
CF_models_to_fit = analysis_info['CF_models_to_fit']
pRF_models_to_fit = analysis_info['pRF_models_to_fit']
fit_hrf = analysis_info['fit_hrf']
filter_predictions = analysis_info['filter_predictions']
filter_type = analysis_info['filter_type']
data_scaling = analysis_info['data_scaling']


# set up constraints
constraints_gauss, constraints_norm = [],[]

# get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# get the directory where the utils are located
utils_dir = script_dir.replace('/scripts', '/data/CF_fit_utils')

# load design matrix
new_dms = np.load(f'{utils_dir}/prf_dm.npy')[discard_volumes:,:,:]

# set up stimulus object
prf_stim = PRFStimulus2D(screen_size_cm=69, 
                         screen_distance_cm=220, 
                         design_matrix=new_dms.T, 
                         TR=1.5)

# set up grid of parameters to search over
grid_nr = 20
max_ecc_size = prf_stim.screen_size_degrees/2.0
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)


# some constants we can use for bounds
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees


# load data for this subject, split into train and test

if fold == 0:
    mydat_train_stim = np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_{data_scaling}_hcp.npy'))
    mydat_test_stim = np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_{data_scaling}_hcp.npy'))
else:
    mydat_train_stim = np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_{data_scaling}_hcp.npy'))
    mydat_test_stim = np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_{data_scaling}_hcp.npy'))

brainmask = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/roimask_wang_hcp.npy')

print(mydat_train_stim.shape)
# split data into n_slices and get the slice we want to fit
fitsize = np.ceil(len(mydat_train_stim)/n_slices).astype(int)
print(f'fitsize is {fitsize}')

mydat_train = split_given_size(mydat_train_stim, fitsize)[slice_n]
mydat_test = split_given_size(mydat_test_stim, fitsize)[slice_n]


# Define grid of parameters to search over

surround_amplitude_grid=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
surround_size_grid=np.array([3,5,8,12,18], dtype='float32')
neural_baseline_grid=np.array([0,1,10,100], dtype='float32')
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


# Define the model and fitter objects
gg = Iso2DGaussianModel(stimulus=prf_stim,
                        filter_predictions=filter_predictions,
                        filter_type=filter_type)

gf_P = Iso2DGaussianFitter(data=mydat_train, model=gg, n_jobs=n_jobs, fit_hrf=fit_hrf)
gf_P.grid_fit(ecc_grid=eccs,
                 polar_grid=polars,
                 size_grid=sizes, 
                 n_batches=n_batches,
                 fixed_grid_baseline=0,
                 grid_bounds=gauss_grid_bounds)


print('finished gridsearch gauss')

np.save(f'{save_dir_pRF}/gauss_grid_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_P.gridsearch_params)


print('starting iterative fit for gauss')

if constraints:
    gf_P.iterative_fit(rsq_threshold=-1, verbose=True, bounds=gauss_bounds, constraints=constraints_gauss,  xtol=1e-5, ftol=1e-5)
else:
    gf_P.iterative_fit(rsq_threshold=-1, verbose=True, bounds=gauss_bounds, xtol=1e-5, ftol=1e-5)


gf_P.crossvalidate_fit(mydat_test, single_hrf=True)


print('finished iterative fit gauss')
np.save(f'{save_dir_pRF}/gauss_prf_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_P.iterative_search_params)


if 'DN_prf' in pRF_models_to_fit:

    ############################
    # ftting norm model
    ############################


    # Define the model and fitter etc
    gg_norm = Norm_Iso2DGaussianModel(stimulus=prf_stim,
                                        filter_predictions=filter_predictions,
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
                            n_batches=n_batches,
                            fixed_grid_baseline=0,
                            rsq_threshold=rsq_threshold,
                            grid_bounds=norm_grid_bounds)

    print('finished gridsearch for DN')
    np.save(f'{save_dir_pRF}/norm_grid_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_norm.gridsearch_params)



    if fit_hrf:
        A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
    else:
        A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0]])

    constraints_norm.append(LinearConstraint(A_ssc_norm,
                                                lb=0,
                                                ub=+inf))


    constraints_norm.append(LinearConstraint(A_ssc_norm, lb=0, ub=+inf))


    print('starting DN fit')

    if constraints:
        gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, bounds=norm_bounds, constraints=constraints_norm, xtol=xtol, ftol=ftol)
    else:
        gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, bounds=norm_bounds, xtol=xtol, ftol=ftol)

    gf_norm.crossvalidate_fit(mydat_test, single_hrf=True)


    print('finished iterative fit DN')

    np.save(f'{save_dir_pRF}/DN_prf_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_norm.iterative_search_params)
