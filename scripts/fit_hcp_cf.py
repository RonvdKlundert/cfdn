import sys
import yaml
import numpy as np
from scipy.optimize import LinearConstraint
from prfpy.stimulus import CFStimulus
from prfpy.model import CFGaussianModel, Norm_CFGaussianModel
from prfpy.fit import CFFitter, Norm_CFGaussianFitter


sys.path.append('/home/klundert/cfdn/cf-tools/')
from preprocess import get_cortex, split_given_size


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
masktype = analysis_info['masktype']
data_scaling = analysis_info['data_scaling']


subsurface_verts = np.load(f'{data_dir}/subsurface_verts_sub-0{sub}_hcp_NoR2.npy')
distance_matrix = np.load(f'{data_dir}/distance_matrix_sub-0{sub}_hcp_NoR2.npy')
logvisual_distance_matrix = np.load(f'{data_dir}/logvisual_distance_matrix_sub-0{sub}_hcp_NoR2.npy')
visual_distance_matrix = np.load(f'{data_dir}/visual_distance_matrix_sub-0{sub}_hcp_NoR2.npy')


if masktype == 'wang':
    ROImask = np.load(f'{data_dir}/roimask_wang_hcp.npy')
    print('using the wang brainmask and drawn V1')
    print(np.sum(ROImask))

elif masktype == 'NoR2':
    ROImask = np.load(f'{data_dir}/brainmask_sub-0{sub}_NoR2.npy')
    print('using the whole brainmask and drawn V1 without applied R2 threshold')
    print(np.sum(ROImask))


# print which models will be fitted
print(f'CF models to fit: {CF_models_to_fit}')

if fold == 0:
    mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_{data_scaling}_hcp.npy')))
    mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_{data_scaling}_hcp.npy')))
else:
    mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_{data_scaling}_hcp.npy')))
    mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_{data_scaling}_hcp.npy')))


# this will discard the first 5 volumes extra!

# if fold == 0:
#     mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_psc_hcp.npy'))[:,5:])
#     mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_psc_hcp.npy'))[:,5:])
# else:
#     mydat_train_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold2_detrend_sub-0{sub}_psc_hcp.npy'))[:,5:])
#     mydat_test_stim = get_cortex(np.nan_to_num(np.load(f'{data_dir}/data_fold1_detrend_sub-0{sub}_psc_hcp.npy'))[:,5:])


############################
# ftting of cortical space
############################


# set up stimulus
train_stim3=CFStimulus(mydat_train_stim, subsurface_verts, distance_matrix)
test_stim3=CFStimulus(mydat_test_stim, subsurface_verts, distance_matrix)

# print the fit size
fitsize = np.ceil(len(mydat_train_stim[ROImask])/n_slices).astype(int)
print(f'fitsize is {fitsize}')

# split the data into slices
mydat_train = split_given_size(mydat_train_stim[ROImask], fitsize)[slice_n]
mydat_test = split_given_size(mydat_test_stim[ROImask], fitsize)[slice_n]


# Define the model
model=CFGaussianModel(train_stim3)

# Define sigmas
sigmas=np.array([0.5,1,2,3,4,5,7,10,20,30,40])

# Define the fitter
gf_vis3 = CFFitter(data=mydat_train,model=model)
gf_vis3.n_jobs = n_jobs


# Perform the fitting.
print('fitting cortical gauss now')
gf_vis3.grid_fit(sigmas, verbose=True, n_batches=n_batches)

# setup up the bounds for the iterative fit

CF_bounds = [(0.1, 45),  # sigmas
                (0, 1000),  # beta
                (0, 0), # baseline
                (0, 0)] # vert

CF_bounds = np.array(CF_bounds)
CF_bounds = np.repeat(CF_bounds[np.newaxis,...], gf_vis3.gridsearch_params.shape[0], axis=0)
CF_bounds[:,3,0] = gf_vis3.vertex_centres
CF_bounds[:,3,1] = gf_vis3.vertex_centres


if constraints:
    gf_vis3.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, constraints=[], starting_params=gf_vis3.gridsearch_params, bounds=CF_bounds, ftol=ftol, xtol=xtol)
else:
    gf_vis3.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, starting_params=gf_vis3.gridsearch_params, bounds=CF_bounds, ftol=ftol, xtol=xtol)


# get model predictions for rsq calculation
fit_stimulus = np.copy(gf_vis3.model.stimulus)
gf_vis3.model.stimulus = test_stim3

sigmasi, betai, baselinei, vert_centrei, R2i = np.copy(gf_vis3.iterative_search_params.T)
model_tc_vis = np.zeros(mydat_train.shape)


for i in range(np.size(vert_centrei)):
    model_tc_vis[i,:] = gf_vis3.model.return_prediction(sigmasi[i], betai[i], baselinei[i], gf_vis3.vertex_centres[i])

gf_vis3.model.stimulus = train_stim3

# crossvalidate rsq
CV_rsq = np.nan_to_num(1-np.sum((mydat_test-model_tc_vis)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))
CV_rsq[CV_rsq <= -1] = np.nan
CV_rsq[CV_rsq >= 1] = np.nan

gf_vis3.iterative_search_params[:,-1] = CV_rsq
gf_vis3.iterative_search_params[:,3] = gf_vis3.vertex_centres

np.save(f'{save_dir_cf}/CF_cortical_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', gf_vis3.iterative_search_params)


if 'DNCF' in CF_models_to_fit:

    # setup up the bounds for the iterative fit

    DNCF_bounds = [(0.1, 50),  # sigmas
                (0, 1000),  # prf amplitude
                (0, 0), # baseline (A)
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

    constraints_gauss, constraints_norm = [],[]

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

    if constraints:
        fitdn.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, constraints=constraints_norm, starting_params=gf_vis3.iterative_search_params, bounds=DNCF_bounds, ftol=ftol, xtol=xtol)
    else:
        fitdn.iterative_fit(rsq_threshold=rsq_threshold, verbose=True, starting_params=gf_vis3.iterative_search_params, bounds=DNCF_bounds, ftol=ftol, xtol=xtol)


    sig, pamp, boldb, verti, srfamp, srfsig, neurb, surrb, Rsq = fitdn.iterative_search_params.T

    fit_stimulus = np.copy(fitdn.model.stimulus)
    fitdn.model.stimulus = test_stim3

    dncf_tc = np.zeros(mydat_test.shape)

    for i in range(np.size(vert_centrei)):
        dncf_tc[i,:] = fitdn.model.return_prediction(sig[i], pamp[i], boldb[i], gf_vis3.vertex_centres[i], srfamp[i], srfsig[i], neurb[i], surrb[i])

    fitdn.model.stimulus = train_stim3

    CVdncf_rsq = np.nan_to_num(1-np.sum((mydat_test-dncf_tc)**2, axis=-1)/(mydat_test.shape[-1]*mydat_test.var(-1)))

    CVdncf_rsq[CVdncf_rsq <= -1] = np.nan
    CVdncf_rsq[CVdncf_rsq >= 1] = np.nan

    fitdn.iterative_search_params[:,-1] = CVdncf_rsq
    fitdn.iterative_search_params[:,3] = gf_vis3.vertex_centres

    np.save(f'{save_dir_cf}/DNCF_cortical_sub-{sub}_fold-{fold}_slice-{slice_n}.npy', fitdn.iterative_search_params)
