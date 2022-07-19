import sys
sys.path.append('/tank/klundert/projects/cfdn/prfpy_cfdn/')
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
import cfhcpy
from cfhcpy.base import AnalysisBase
from cfhcpy.base import AnalysisBase


########################################################################################
# set parameters
########################################################################################


sub = sys.argv[1]
n_jobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
chunk_n = int(sys.argv[4])


subsurface_verts = np.load(f'/scratch/2021/nprf_ss/derivatives/cf-fits/Surface_dm/subsurface_verts.npy')
distance_matrix = np.load(f'/scratch/2021/nprf_ss/derivatives/cf-fits/Surface_dm/distance_matrix.npy')
logvisual_distance_matrix = np.load(f'/scratch/2021/nprf_ss/derivatives/cf-fits/Surface_dm/logvisual_distance_matrix.npy')
visual_distance_matrix = np.load(f'/scratch/2021/nprf_ss/derivatives/cf-fits/Surface_dm/visual_distance_matrix.npy')
ROImask = np.load(f'/tank/klundert/projects/cfdn/data/CF_fit_utils/visual_mask_hcp.npy')

ac = AnalysisBase()
ac.startup(subject=sub, experiment_id="ret", yaml_file="/tank/klundert/projects/hcp_movie/config.yml")
mydat_train_stim = get_cortex(ac._read_tc_data(run=0).T)
mydat_test_stim = get_cortex(ac._read_tc_data(run=1).T)


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

chunkedmasks = [ROImask[i:i+9882] for i in range(0,len(ROImask),9882)]
chunkedtrain = [mydat_train_stim[i:i+9882] for i in range(0,len(mydat_train_stim),9882)]
chunkedtest = [mydat_test_stim[i:i+9882] for i in range(0,len(mydat_test_stim),9882)]


############################
# ftting of logvisual space
############################

train_stim=CFStimulus(mydat_train_stim, subsurface_verts, logvisual_distance_matrix)
test_stim=CFStimulus(mydat_test_stim, subsurface_verts, logvisual_distance_matrix)

mydat_train = chunkedtrain[slice_n][chunkedmasks[slice_n]]
mydat_test = chunkedtest[slice_n][chunkedmasks[slice_n]]

model=CFGaussianModel(train_stim)

# Define sigmas
sigmas=np.array([0.5,1,2,3,4,5,7,10,20,30,40,60,80,110])

# Define the fitter
gf_vis = CFFitter(data=mydat_train,model=model)
gf_vis.n_jobs = n_jobs
# Perform the fitting.
print('fitting now')
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

np.save(f'/tank/klundert/projects/cfdn/scripts/data_check_sub-{sub}_slice-{chunk_n}.npy', gf_vis.iterative_search_params)
# np.save(f'/scratch/2021/nprf_ss/derivatives/cf-fits/sub-0{sub}/limit_sample_gauss/visual_space/sub-0{sub}_task-prf_space-fsLR_den-170k_desc-preproc_gauss_CF_params_LOGvisual_space_fold{fold}_zsc.npy', gf_vis.iterative_search_params)



# DNCF_bounds = [(0.1, 150),  # sigmas
#             (0, 1000),  # prf amplitude
#             (0, 0.0001), # baseline (A)
#             (0, 0), # vert
#             (0, 1000), # srf amplitude (C)
#             (0.3, 180), # surround sigma 
#             (0, 1000), # neural baseline (B)
#             (1e-6, 1000)] # surround baseline (D)


# DNCF_bounds = np.array(DNCF_bounds)
# DNCF_bounds = np.repeat(DNCF_bounds[np.newaxis,...], gf_vis.gridsearch_params.shape[0], axis=0)
# DNCF_bounds[:,3,0] = gf_vis.vertex_centres
# DNCF_bounds[:,3,1] = gf_vis.vertex_centres

# # set constraint for surround>centre sigma
# constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]
# A_ssc_norm = np.array([[-1,0,0,0,0,1,0,0]])  
# constraints_norm.append(LinearConstraint(A_ssc_norm,
#                                             lb=0,
#                                             ub=+np.inf))

# gfdn = Norm_CFGaussianModel(train_stim)

# fitdn = Norm_CFGaussianFitter(data=mydat_train,
#                                    model=gfdn,
#                                    n_jobs=n_jobs,
#                                    previous_gaussian_fitter=gf_vis)

# fitdn.iterative_fit(rsq_threshold=-1, verbose=True, constraints=constraints_norm, starting_params=gf_vis.iterative_search_params, bounds=DNCF_bounds, ftol=1e-7, xtol=1e-7)

# sig, pamp, boldb, verti, srfamp, srfsig, neurb, surrb, Rsq = fitdn.iterative_search_params.T

# fit_stimulus = np.copy(fitdn.model.stimulus)
# fitdn.model.stimulus = test_stim

# dncf_tc = np.zeros(mydat_test.shape)
# i = 0
# for i in range(np.size(vert_centrei)):
#     dncf_tc[i,:] = fitdn.model.return_prediction(sig[i], pamp[i], boldb[i], gf_vis.vertex_centres[i], srfamp[i], srfsig[i], neurb[i], surrb[i])

# fitdn.model.stimulus = train_stim

# CVdncf_rsq = np.nan_to_num(1-np.sum((mydat_test-dncf_tc)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))

# CVdncf_rsq[CVdncf_rsq <= -1] = np.nan
# CVdncf_rsq[CVdncf_rsq >= 1] = np.nan

# fitdn.iterative_search_params[:,-1] = CVdncf_rsq
# fitdn.iterative_search_params[:,3] = gf_vis.vertex_centres

# np.save(f'/tank/klundert/projects/cfdn/scripts/data_check.npy', fitdn.iterative_search_params)
