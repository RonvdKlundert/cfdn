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
#import cfhcpy
#from cfhcpy.base import AnalysisBase
#from cfhcpy.base import AnalysisBase
import yaml


########################################################################################
# set parameters
########################################################################################

# with open('/home/klundert/cfdn/scripts/analysis_settings.yml') as f:
#     analysis_info = yaml.safe_load(f)
    
# all_subs = analysis_info['analysis']['full_data_subjects']

print('started')

folder = 'data_zsc_psc'


id = int(sys.argv[1])
n_jobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
fold = int(sys.argv[4])

sub = str(id+1)

subsurface_verts = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/subsurface_verts_sub-0{sub}_hcp.npy')
distance_matrix = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/distance_matrix_sub-0{sub}_hcp.npy')


ROImask = np.load(f'/home/klundert/cfdn/data/CF_fit_utils/roimask_wang_hcp.npy')


if fold == 0:
    mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold1_detrend_sub-0{sub}_zsc_hcp.npy')))
    mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold2_detrend_sub-0{sub}_zsc_hcp.npy')))
else:
    mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold2_detrend_sub-0{sub}_zsc_hcp.npy')))
    mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'/home/klundert/cfdn/data/CF_fit_utils/data_fold1_detrend_sub-0{sub}_zsc_hcp.npy')))




############################
# ftting of cortical space
############################

train_stim3=CFStimulus(mydat_train_stim, subsurface_verts, distance_matrix)
test_stim3=CFStimulus(mydat_test_stim, subsurface_verts, distance_matrix)

# mydat_train = chunkedtrain[slice_n][chunkedmasks[slice_n]]
# mydat_test = chunkedtest[slice_n][chunkedmasks[slice_n]]

fitsize = np.ceil(len(mydat_train_stim[ROImask])/50).astype(int)
print(f'fitsize is {fitsize}')

mydat_train = split_given_size(mydat_train_stim[ROImask], fitsize)[slice_n]
mydat_test = split_given_size(mydat_test_stim[ROImask], fitsize)[slice_n]

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
                (0, 0), # baseline
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

np.save(f'/home/klundert/{folder}/CF_cortical_zsc_fit_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_vis3.iterative_search_params)



