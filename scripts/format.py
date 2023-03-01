import numpy as np
import pickle
import sys
import os
from cfdn.utils import FormatData


# get whether to format prf or cf data from input
toformat = sys.argv[1]
# print(toformat)

# check wheter the input is valid
if toformat not in ['prf', 'cf']:
    raise ValueError('Input must be either prf or cf')


prfdir = '/tank/shared/2021/visual/DN-CF/spliced_fits/prf_fits'
cfdir = '/tank/shared/2021/visual/DN-CF/spliced_fits/cf_fits'


if toformat == 'prf':

    # first get the weighted average of the prf fits for both folds, these will be used for assginment 
    # of x and y coordinates to the vertex centres of the CF fits

    prf_dat = FormatData(prfdir, ['gauss_prf', 'DN_prf'], [1,2])
    prf_dat.arrange_fits()
    prf_dat.average_folds()


    np.save('/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted/pRF-params_sub-01_gauss_weighted.npy', prf_dat.arranged_data['gauss_prf_sub-01_average'])
    np.save('/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted/pRF-params_sub-01_norm_weighted.npy', prf_dat.arranged_data['DN_prf_sub-01_average'])

    np.save('/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted/pRF-params_sub-02_gauss_weighted.npy', prf_dat.arranged_data['gauss_prf_sub-02_average'])
    np.save('/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted/pRF-params_sub-02_norm_weighted.npy', prf_dat.arranged_data['DN_prf_sub-02_average'])

    # get prf fits per fold for each subject and save them as dataframes

    prf_dat = FormatData(prfdir, ['gauss_prf', 'DN_prf'], [1,2])
    prf_dat.arrange_fits()
    prf_dat.create_dataframe()

    # specify the directory path to save the file
    file_path = '/tank/shared/2021/visual/DN-CF/viz_data/fit_data/wholebrain_pRF-fits.pickle'

    # open the file with the specified directory path
    with open(file_path, 'wb') as f:
        pickle.dump(prf_dat.arranged_data, f)


elif toformat == 'cf':
    # check whether the weighted prf fits files of at least gauss are present for both subjects
    # if not, raise an error

    # implement a better check later:
    subject_ids = ['sub-01', 'sub-02']
    for sub_id in subject_ids:
        filepath = f'/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted/pRF-params_{sub_id}_gauss_weighted.npy'
        if not os.path.exists(filepath):
            raise ValueError(f'Weighted prf fits file not found for {sub_id}, run prf format first')



    cf_dat = FormatData(cfdir, ['CF_cortical_fit', 'DNCF_cortical_fit'], [1,2])
    cf_dat.arrange_fits()
    cf_dat.add_xy_values('/tank/shared/2021/visual/DN-CF/derivatives/fits_weighted')
    cf_dat.create_dataframe('/tank/shared/2021/visual/DN-CF/CF_fit_utils/brainmask')

    # specify the directory path to save the file
    file_path = '/tank/shared/2021/visual/DN-CF/viz_data/fit_data/wholebrain_CF-fits.pickle'

    # open the file with the specified directory path
    with open(file_path, 'wb') as f:
        pickle.dump(cf_dat.arranged_data, f)

