import numpy as np
from IPython import embed as shell
import re
import os
import scipy.io as io
import nibabel as nb
import time
import sys
import h5py
from nilearn.surface import load_surf_data
sys.path.append('/tank/klundert/projects/cfdn/scripts/')





#################################
subs = [100610]
n_jobs = 23
ROImask = np.load(f'/tank/klundert/projects/cfdn/data/CF_fit_utils/visual_mask_hcp.npy')


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

for sub in subs:
    chunk_n = 0
    for slice_n, chunk in enumerate(chunkedmasks):
        if chunk.sum() > 0:
            print(f'starting fit {chunk_n} for {sub} with {chunk.sum()} voxels')
            jobscript = open('jobscript_base.sh')
            working_string = jobscript.read()
            jobscript.close()

            RE_dict =  {
            '---sub---': str(sub),
            '---n_jobs---': str(n_jobs),
            '---slice_n---': str(slice_n),
            '---chunk_n---': str(chunk_n)}


            for e in RE_dict:
                rS = re.compile(e)
                working_string = re.sub(rS, RE_dict[e], working_string)

            of = open('jobscript', 'w')
            of.write(working_string)
            of.close()
            print(working_string)
            os.system(working_string)
            chunk_n = chunk_n + 1