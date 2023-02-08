import sys
sys.path.append('/home/klundert/cfdn/prfpy_cfdn/')
import os
import numpy as np
import preprocess
import cortex as cx
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
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter
from prfpy.utils import Subsurface
from prfpy.stimulus import CFStimulus
from prfpy.model import CFGaussianModel
from prfpy.fit import CFFitter
from prfpy.model import Norm_CFGaussianModel
from prfpy.fit import Norm_CFGaussianFitter
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.io import loadmat
from scipy.ndimage import median_filter, gaussian_filter, binary_propagation
from preprocess import get_cortex
from preprocess import split_given_size
import cfhcpy
from cfhcpy.base import AnalysisBase
from cfhcpy.base import AnalysisBase
import yaml

########################################################################################
# set parameters
########################################################################################

with open('/home/klundert/cfdn/scripts/analysis_settings.yml') as f:
    analysis_info = yaml.safe_load(f)
    
all_subs = analysis_info['analysis']['full_data_subjects']



id = int(sys.argv[1])
n_jobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
fold = int(sys.argv[4])

sub = str(all_subs[id])


subsurface_verts = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/subsurface_verts_full.npy')
distance_matrix = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/distance_matrix_full.npy')
logvisual_distance_matrix = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/logvisual_distance_full_matrix.npy')
visual_distance_matrix = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/visual_distance_matrix_full.npy')
ROImask = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/visual_mask_hcp.npy')

ac = AnalysisBase()

ac.startup(subject=sub, experiment_id="ret", yaml_file="/home/klundert/hcp_movie/config.yml")

if fold == 0:
    mydat_train_stim = np.nan_to_num(get_cortex(ac._read_tc_data(run=0).T))
    mydat_test_stim = np.nan_to_num(get_cortex(ac._read_tc_data(run=1).T))
else:
    mydat_train_stim = np.nan_to_num(get_cortex(ac._read_tc_data(run=1).T))
    mydat_test_stim = np.nan_to_num(get_cortex(ac._read_tc_data(run=0).T))


# roi_index_dict = {
#     # somatosensory:
#     'CS1_4': 8, 'CS2_3a': 53, 'CS3_3b': 9, 'CS4_1': 51, 'CS5_2': 52,
#     # auditory:
#     'A1': 24, 'PBelt': 124, 'MBelt': 173, 'LBelt': 174, '52': 103, 'RI': 104,
#     # low-level visual:
#     'V1': 1, 'V2': 4, 'V3': 5,
#     # mid-level and high-level visual:
#     'V3A': 13, 'V3B': 19, 'IPS1': 17, 'LIPv': 48, 'LIPd': 95, 
#     'VIP': 49, 'FEF': 10, 'MST': 2, 'MT': 23, 'LO1': 20, 'LO2': 21, 'LO3': 159
#     }

# atlas_data = np.concatenate([load_surf_data(
#         os.path.join('/tank/klundert/content/data/atlas', f'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.59k_fs_LR.dlabel.{hemi}.gii'))
#          for hemi in ['L', 'R']])
# atlas_data_both_hemis = np.mod(atlas_data, 180)


# ROImask = atlas_data_both_hemis == roi_index_dict[mask]

# chunkedmasks = [ROImask[i:i+9882] for i in range(0,len(ROImask),9882)]
# chunkedtrain = [mydat_train_stim[i:i+9882] for i in range(0,len(mydat_train_stim),9882)]
# chunkedtest = [mydat_test_stim[i:i+9882] for i in range(0,len(mydat_test_stim),9882)]


############################
# ftting of logvisual space
############################

train_stim=CFStimulus(mydat_train_stim, subsurface_verts, logvisual_distance_matrix)
test_stim=CFStimulus(mydat_test_stim, subsurface_verts, logvisual_distance_matrix)

# mydat_train = chunkedtrain[slice_n][chunkedmasks[slice_n]]
# mydat_test = chunkedtest[slice_n][chunkedmasks[slice_n]]

mydat_train = split_given_size(mydat_train_stim[ROImask], 1693)[slice_n]
mydat_test = split_given_size(mydat_test_stim[ROImask], 1693)[slice_n]

model=CFGaussianModel(train_stim)

# Define sigmas
sigmas=np.array([0.5,1,2,3,4,5,7,10,20,30,40,60,80,110])

# Define the fitter
gf_vis = CFFitter(data=mydat_train,model=model)
gf_vis.n_jobs = n_jobs
# Perform the fitting.
print('fitting logvisual gauss now')
gf_vis.grid_fit(sigmas, verbose=False, n_batches=60)


CF_bounds = [(0.1, 150),  # sigmas
                (0, 1000),  # beta
                (0, 0.0001), # baseline
                (0, 0)] # vert

CF_bounds = np.array(CF_bounds)
CF_bounds = np.repeat(CF_bounds[np.newaxis,...], gf_vis.gridsearch_params.shape[0], axis=0)
CF_bounds[:,3,0] = gf_vis.vertex_centres
CF_bounds[:,3,1] = gf_vis.vertex_centres

gf_vis.iterative_fit(rsq_threshold=-1, verbose=True, constraints=[], starting_params=gf_vis.gridsearch_params, bounds=CF_bounds, ftol=1e-7, xtol=1e-7)

# get model predictions
fit_stimulus = np.copy(gf_vis.model.stimulus)
gf_vis.model.stimulus = test_stim

sigmasi, betai, baselinei, vert_centrei, R2i = np.copy(gf_vis.iterative_search_params.T)
model_tc_vis = np.zeros(mydat_train.shape)
i = 0
for i in range(np.size(vert_centrei)):
    model_tc_vis[i,:] = gf_vis.model.return_prediction(sigmasi[i], betai[i], baselinei[i], gf_vis.vertex_centres[i])

gf_vis.model.stimulus = train_stim

# crossvalidate rsq
CV_rsq = np.nan_to_num(1-np.sum((mydat_test-model_tc_vis)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))
CV_rsq[CV_rsq <= -1] = np.nan
CV_rsq[CV_rsq >= 1] = np.nan

gf_vis.iterative_search_params[:,-1] = CV_rsq
gf_vis.iterative_search_params[:,3] = gf_vis.vertex_centres

np.save(f'/home/klundert/dataf/CF_logvis_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_vis.iterative_search_params)



DNCF_bounds = [(0.1, 150),  # sigmas
            (0, 1000),  # prf amplitude
            (0, 0.0001), # baseline (A)
            (0, 0), # vert
            (0, 1000), # srf amplitude (C)
            (0.3, 180), # surround sigma 
            (0, 1000), # neural baseline (B)
            (1e-6, 1000)] # surround baseline (D)


DNCF_bounds = np.array(DNCF_bounds)
DNCF_bounds = np.repeat(DNCF_bounds[np.newaxis,...], gf_vis.gridsearch_params.shape[0], axis=0)
DNCF_bounds[:,3,0] = gf_vis.vertex_centres
DNCF_bounds[:,3,1] = gf_vis.vertex_centres

# set constraint for surround>centre sigma
constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]
A_ssc_norm = np.array([[-1,0,0,0,0,1,0,0]])  
constraints_norm.append(LinearConstraint(A_ssc_norm,
                                            lb=0,
                                            ub=+np.inf))

gfdn = Norm_CFGaussianModel(train_stim)

fitdn = Norm_CFGaussianFitter(data=mydat_train,
                                   model=gfdn,
                                   n_jobs=n_jobs,
                                   previous_gaussian_fitter=gf_vis)
print('fitting logvisual DN now')
fitdn.iterative_fit(rsq_threshold=-1, verbose=True, constraints=constraints_norm, starting_params=gf_vis.iterative_search_params, bounds=DNCF_bounds, ftol=1e-7, xtol=1e-7)

sig, pamp, boldb, verti, srfamp, srfsig, neurb, surrb, Rsq = fitdn.iterative_search_params.T

fit_stimulus = np.copy(fitdn.model.stimulus)
fitdn.model.stimulus = test_stim

dncf_tc = np.zeros(mydat_test.shape)
i = 0
for i in range(np.size(vert_centrei)):
    dncf_tc[i,:] = fitdn.model.return_prediction(sig[i], pamp[i], boldb[i], gf_vis.vertex_centres[i], srfamp[i], srfsig[i], neurb[i], surrb[i])

fitdn.model.stimulus = train_stim

CVdncf_rsq = np.nan_to_num(1-np.sum((mydat_test-dncf_tc)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))

CVdncf_rsq[CVdncf_rsq <= -1] = np.nan
CVdncf_rsq[CVdncf_rsq >= 1] = np.nan

fitdn.iterative_search_params[:,-1] = CVdncf_rsq
fitdn.iterative_search_params[:,3] = gf_vis.vertex_centres

np.save(f'/home/klundert/dataf/DNCF_logvis_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', fitdn.iterative_search_params)




############################
# ftting of regular visual space
############################

train_stim2=CFStimulus(mydat_train_stim, subsurface_verts, visual_distance_matrix)
test_stim2=CFStimulus(mydat_test_stim, subsurface_verts, visual_distance_matrix)

# mydat_train = chunkedtrain[slice_n][chunkedmasks[slice_n]]
# mydat_test = chunkedtest[slice_n][chunkedmasks[slice_n]]

mydat_train = split_given_size(mydat_train_stim[ROImask], 1693)[slice_n]
mydat_test = split_given_size(mydat_test_stim[ROImask], 1693)[slice_n]

model=CFGaussianModel(train_stim2)

# Define sigmas
sigmas=np.array([0.5,1,2,3,4,5,7,10,20,30,40,60,80,110])

# Define the fitter
gf_vis2 = CFFitter(data=mydat_train,model=model)
gf_vis2.n_jobs = n_jobs
# Perform the fitting.
print('fitting visual gauss now')
gf_vis2.grid_fit(sigmas, verbose=False, n_batches=60)


CF_bounds = [(0.1, 150),  # sigmas
                (0, 1000),  # beta
                (0, 0.0001), # baseline
                (0, 0)] # vert

CF_bounds = np.array(CF_bounds)
CF_bounds = np.repeat(CF_bounds[np.newaxis,...], gf_vis2.gridsearch_params.shape[0], axis=0)
CF_bounds[:,3,0] = gf_vis2.vertex_centres
CF_bounds[:,3,1] = gf_vis2.vertex_centres

gf_vis2.iterative_fit(rsq_threshold=-1, verbose=True, constraints=[], starting_params=gf_vis2.gridsearch_params, bounds=CF_bounds, ftol=1e-7, xtol=1e-7)

# get model predictions
fit_stimulus = np.copy(gf_vis2.model.stimulus)
gf_vis2.model.stimulus = test_stim2

sigmasi, betai, baselinei, vert_centrei, R2i = np.copy(gf_vis2.iterative_search_params.T)
model_tc_vis = np.zeros(mydat_train.shape)
i = 0
for i in range(np.size(vert_centrei)):
    model_tc_vis[i,:] = gf_vis2.model.return_prediction(sigmasi[i], betai[i], baselinei[i], gf_vis2.vertex_centres[i])

gf_vis2.model.stimulus = train_stim2

# crossvalidate rsq
CV_rsq = np.nan_to_num(1-np.sum((mydat_test-model_tc_vis)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))
CV_rsq[CV_rsq <= -1] = np.nan
CV_rsq[CV_rsq >= 1] = np.nan

gf_vis2.iterative_search_params[:,-1] = CV_rsq
gf_vis2.iterative_search_params[:,3] = gf_vis2.vertex_centres

np.save(f'/home/klundert/dataf/CF_vis_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_vis2.iterative_search_params)



DNCF_bounds = [(0.1, 150),  # sigmas
            (0, 1000),  # prf amplitude
            (0, 0.0001), # baseline (A)
            (0, 0), # vert
            (0, 1000), # srf amplitude (C)
            (0.3, 180), # surround sigma 
            (0, 1000), # neural baseline (B)
            (1e-6, 1000)] # surround baseline (D)


DNCF_bounds = np.array(DNCF_bounds)
DNCF_bounds = np.repeat(DNCF_bounds[np.newaxis,...], gf_vis2.gridsearch_params.shape[0], axis=0)
DNCF_bounds[:,3,0] = gf_vis2.vertex_centres
DNCF_bounds[:,3,1] = gf_vis2.vertex_centres

# set constraint for surround>centre sigma
constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]
A_ssc_norm = np.array([[-1,0,0,0,0,1,0,0]])  
constraints_norm.append(LinearConstraint(A_ssc_norm,
                                            lb=0,
                                            ub=+np.inf))

gfdn = Norm_CFGaussianModel(train_stim2)

fitdn = Norm_CFGaussianFitter(data=mydat_train,
                                   model=gfdn,
                                   n_jobs=n_jobs,
                                   previous_gaussian_fitter=gf_vis2)
print('fitting visual DN now')
fitdn.iterative_fit(rsq_threshold=-1, verbose=True, constraints=constraints_norm, starting_params=gf_vis2.iterative_search_params, bounds=DNCF_bounds, ftol=1e-7, xtol=1e-7)

sig, pamp, boldb, verti, srfamp, srfsig, neurb, surrb, Rsq = fitdn.iterative_search_params.T

fit_stimulus = np.copy(fitdn.model.stimulus)
fitdn.model.stimulus = test_stim2

dncf_tc = np.zeros(mydat_test.shape)
i = 0
for i in range(np.size(vert_centrei)):
    dncf_tc[i,:] = fitdn.model.return_prediction(sig[i], pamp[i], boldb[i], gf_vis2.vertex_centres[i], srfamp[i], srfsig[i], neurb[i], surrb[i])

fitdn.model.stimulus = train_stim2

CVdncf_rsq = np.nan_to_num(1-np.sum((mydat_test-dncf_tc)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))

CVdncf_rsq[CVdncf_rsq <= -1] = np.nan
CVdncf_rsq[CVdncf_rsq >= 1] = np.nan

fitdn.iterative_search_params[:,-1] = CVdncf_rsq
fitdn.iterative_search_params[:,3] = gf_vis2.vertex_centres

np.save(f'/home/klundert/dataf/DNCF_vis_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', fitdn.iterative_search_params)



############################
# ftting of cortical space
############################

train_stim3=CFStimulus(mydat_train_stim, subsurface_verts, distance_matrix)
test_stim3=CFStimulus(mydat_test_stim, subsurface_verts, distance_matrix)

# mydat_train = chunkedtrain[slice_n][chunkedmasks[slice_n]]
# mydat_test = chunkedtest[slice_n][chunkedmasks[slice_n]]

mydat_train = split_given_size(mydat_train_stim[ROImask], 1693)[slice_n]
mydat_test = split_given_size(mydat_test_stim[ROImask], 1693)[slice_n]

model=CFGaussianModel(train_stim3)

# Define sigmas
sigmas=np.array([0.5,1,2,3,4,5,7,10,20,30,40])

# Define the fitter
gf_vis3 = CFFitter(data=mydat_train,model=model)
gf_vis3.n_jobs = n_jobs
# Perform the fitting.
print('fitting cortical gauss now')
gf_vis3.grid_fit(sigmas, verbose=False, n_batches=60)


CF_bounds = [(0.1, 45),  # sigmas
                (0, 1000),  # beta
                (0, 0.0001), # baseline
                (0, 0)] # vert

CF_bounds = np.array(CF_bounds)
CF_bounds = np.repeat(CF_bounds[np.newaxis,...], gf_vis3.gridsearch_params.shape[0], axis=0)
CF_bounds[:,3,0] = gf_vis3.vertex_centres
CF_bounds[:,3,1] = gf_vis3.vertex_centres

gf_vis3.iterative_fit(rsq_threshold=-1, verbose=True, constraints=[], starting_params=gf_vis3.gridsearch_params, bounds=CF_bounds, ftol=1e-7, xtol=1e-7)

# get model predictions
fit_stimulus = np.copy(gf_vis3.model.stimulus)
gf_vis3.model.stimulus = test_stim3

sigmasi, betai, baselinei, vert_centrei, R2i = np.copy(gf_vis3.iterative_search_params.T)
model_tc_vis = np.zeros(mydat_train.shape)
i = 0
for i in range(np.size(vert_centrei)):
    model_tc_vis[i,:] = gf_vis3.model.return_prediction(sigmasi[i], betai[i], baselinei[i], gf_vis3.vertex_centres[i])

gf_vis3.model.stimulus = train_stim3

# crossvalidate rsq
CV_rsq = np.nan_to_num(1-np.sum((mydat_test-model_tc_vis)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))
CV_rsq[CV_rsq <= -1] = np.nan
CV_rsq[CV_rsq >= 1] = np.nan

gf_vis3.iterative_search_params[:,-1] = CV_rsq
gf_vis3.iterative_search_params[:,3] = gf_vis3.vertex_centres

np.save(f'/home/klundert/dataf/CF_cortical_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_vis3.iterative_search_params)



DNCF_bounds = [(0.1, 50),  # sigmas
            (0, 1000),  # prf amplitude
            (0, 0.0001), # baseline (A)
            (0, 0), # vert
            (0, 1000), # srf amplitude (C)
            (0.3, 100), # surround sigma 
            (0, 1000), # neural baseline (B)
            (1e-6, 1000)] # surround baseline (D)


DNCF_bounds = np.array(DNCF_bounds)
DNCF_bounds = np.repeat(DNCF_bounds[np.newaxis,...], gf_vis3.gridsearch_params.shape[0], axis=0)
DNCF_bounds[:,3,0] = gf_vis3.vertex_centres
DNCF_bounds[:,3,1] = gf_vis3.vertex_centres

# set constraint for surround>centre sigma
constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]
A_ssc_norm = np.array([[-1,0,0,0,0,1,0,0]])  
constraints_norm.append(LinearConstraint(A_ssc_norm,
                                            lb=0,
                                            ub=+np.inf))

gfdn = Norm_CFGaussianModel(train_stim3)

fitdn = Norm_CFGaussianFitter(data=mydat_train,
                                   model=gfdn,
                                   n_jobs=n_jobs,
                                   previous_gaussian_fitter=gf_vis3)
print('fitting cortical DN now')
fitdn.iterative_fit(rsq_threshold=-1, verbose=True, constraints=constraints_norm, starting_params=gf_vis3.iterative_search_params, bounds=DNCF_bounds, ftol=1e-7, xtol=1e-7)

sig, pamp, boldb, verti, srfamp, srfsig, neurb, surrb, Rsq = fitdn.iterative_search_params.T

fit_stimulus = np.copy(fitdn.model.stimulus)
fitdn.model.stimulus = test_stim3

dncf_tc = np.zeros(mydat_test.shape)
i = 0
for i in range(np.size(vert_centrei)):
    dncf_tc[i,:] = fitdn.model.return_prediction(sig[i], pamp[i], boldb[i], gf_vis3.vertex_centres[i], srfamp[i], srfsig[i], neurb[i], surrb[i])

fitdn.model.stimulus = train_stim3

CVdncf_rsq = np.nan_to_num(1-np.sum((mydat_test-dncf_tc)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))

CVdncf_rsq[CVdncf_rsq <= -1] = np.nan
CVdncf_rsq[CVdncf_rsq >= 1] = np.nan

fitdn.iterative_search_params[:,-1] = CVdncf_rsq
fitdn.iterative_search_params[:,3] = gf_vis3.vertex_centres

np.save(f'/home/klundert/dataf/DNCF_cortical_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', fitdn.iterative_search_params)
