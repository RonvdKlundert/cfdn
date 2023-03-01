import numpy as np
import nibabel as nb
from nibabel import cifti2
import h5py
import natsort
import glob
import os
import pandas as pd

class FormatData:

    """FormatData
    Class for formatting data after fitting it as slices on LISA

    data must be saved in the following way:

        dir/{model}_sub-{sub_id}_fold-{fold_id}_slice-{slice_id}.npy
        where {model} is the name of the model, {sub_id} is the subject id,
    """

    def __init__(self, dir, models, subjects):
        """__init__

        parameters
        ----------
        prfdir : str (optional)
            path to the prf fits for the given subjects
            optional, as this is only needed for assinging x and y coordinates
            to cf fits
        dir : str
            directory where the data is stored
        models : list
            list of models that were fitted
        subjects : list    
            subject id(s)
        weighted : bool
            whether the data should be weighted by R2 or not
        """

        
        self.dir = dir
        self.models = models
        self.subjects = subjects


    def arrange_fits(self, weighted=True):
        """arrange_fits

        will load fits from split/sliced up data from a directory 
        and arrange it to its orignal shape.
        data must be saved in the following way:

        dir/{model}_sub-{sub_id}_fold-{fold_id}_slice-{slice_id}.npy
        where {model} is the name of the model, {sub_id} is the subject id,

        parameters
        ----------
        dir : str
            directory where the data is stored
        models : list
            list of models that were fitted
        subjects : list    
            subject id(s)
        weighted : bool
            whether the data should be weighted by R2 or not

        returns
        -------
        data_dic : dict
            dictionary with the data arranged to its original shape for each subject and fold
        """

        # first check whether a file in the directory exist with the right naming  
        # if not, raise an error
        for model in self.models:
            if not os.path.isfile(f'{self.dir}/{model}_sub-{self.subjects[0]}_fold-0_slice-0.npy'):
                raise ValueError('The directory does not contain these files. Please check the naming of the files.')   



        data_dic = {}


        for i, sub in enumerate(self.subjects):        
            
            for model in self.models:

                both_folds = []
                for fold in [0,1]:

                    for j, infile in enumerate(natsort.natsorted((glob.glob(f'{self.dir}/{model}_sub-{sub}_fold-{fold}_slice-*.npy')))):

                        if j == 0:
                            dattest = np.load(infile)
                        else:
                            dattest = np.vstack([dattest, np.load(infile)])

                        
                    both_folds = dattest[np.newaxis,:]

                    # weight parameters by R2
                    a = np.nan_to_num(np.array(both_folds)[:,:,-1])
                    a[a<0] = 1e-6
                    a[a==0] = 1e-6
                    b = np.repeat(a[:, :, np.newaxis], np.array(both_folds).shape[-1], axis=2)
                    data_dic.update({f'{model}_sub-0{sub}_fold-{fold}': np.average(both_folds, axis=0, weights=b)})

        self.arranged_data = data_dic




    def add_xy_values(self, prf_dir):
        """add_xy_values

        will add x and y values to CF fits based on vertex centres

        parameters
        ----------
        prf_dir : string
            path to one of the prf fits, assumes prf fits are 170k
        """

        selections = {}


        for key in self.arranged_data.keys():
            # get subject from key
            sub = key.split('_')[3]

            # Create a list of files containing the words "gauss" and "sub-01"
            file_list = glob.glob(f'{prf_dir}/*pRF*{sub}*')

            # load prf fits from the subject
            # Load the file directly if there's only one matching file
            # Check if a file has already been selected for this subject
            if sub in selections:
                selection = selections[sub]
                print(f"Loadind data for {sub} from {file_list[selection]}")
            # Prompt the user to select a file if there are multiple matching files for a new subject
            else:
                if len(file_list) == 1:
                    selection = 0
                    selections[sub] = selection
                    print(f"Loading data for {sub} from {file_list[selection]}")
                else:
                    print(f"Multiple files found for {sub}:")
                    for i, file in enumerate(file_list):
                        print(f"{i+1}: {file}")
                    selection = int(input("Enter the number of the file to load: ")) - 1
                    selections[sub] = selection
                    print(f"Loading data for {sub} from {file_list[selection]}")

            x, y = get_cortex(np.load(file_list[selection])[:,0]), get_cortex(np.load(file_list[selection])[:,1])

            d = {'x': x, 'y': y}
            ret_frame = pd.DataFrame(data=d)

            spliced_lookup=ret_frame.iloc[self.arranged_data[key][:,3].astype(int)]

            self.arranged_data[key] = np.hstack([self.arranged_data[key], np.asarray(spliced_lookup)])

    def average_folds(self, weighted=True):
        """average_folds
        
        will average the fits from the two folds of the same subject
        weighted by R2

        """
            
        for key in self.arranged_data.keys():
            if key.split('_')[3] == 'fold-0':
                # get R2 values of first fold
                a = np.nan_to_num(np.array(self.arranged_data[key])[:,-1])
                a[a<0] = 1e-6
                a[a==0] = 1e-6
                w_first_fold = np.repeat(a[:, np.newaxis], np.array(self.arranged_data[key]).shape[-1], axis=1)

                # get R2 values of second fold
                b = np.nan_to_num(np.array(self.arranged_data[key.replace('fold-0', 'fold-1')])[:,-1])
                b[b<0] = 1e-6
                b[b==0] = 1e-6
                w_second_fold = np.repeat(b[:, np.newaxis], np.array(self.arranged_data[key.replace('fold-0', 'fold-1')]).shape[-1], axis=1)


                # average the two folds
                if weighted:
                    self.arranged_data[key] = np.average([self.arranged_data[key], self.arranged_data[key.replace('fold-0', 'fold-1')]], axis=0, weights=[w_first_fold, w_second_fold])
                else:
                    self.arranged_data[key] = np.average([self.arranged_data[key], self.arranged_data[key.replace('fold-0', 'fold-1')]], axis=0)

            

        
        # rename the keys to replace "fold-0" with "average"
        self.arranged_data = {key.replace('fold-0', 'average'): value for key, value in self.arranged_data.items()}

        # remove the second fold
        self.arranged_data = {key: value for key, value in self.arranged_data.items() if 'fold-1' not in key}


    def create_dataframe(self, maskdir=None):
        """create_dataframe

        will create a dataframe from the arranged data

        parameters
        ----------
        mask : str (optional)
            directory of mask to get data back to original 118k shape
            form: /path/to/mask_sub-0*.npy, where you can leave out the {_sub_0*} part
            same as the one used for fitting the CF
            necessary if data is not in 118k for plotting
        """

        for key in self.arranged_data.keys():

            # make CF dataframe
            if self.arranged_data[key].shape[1] == 7:
                sub = key.split('_')[3]
                brainmask = np.load(f'{maskdir}_{sub}.npy')
                cf_gauss = np.zeros([118584, 7])
                cf_gauss[:] = np.nan
                # fill in the data for original 118k
                cf_gauss[brainmask] = self.arranged_data[key]
                # Create a list of parameter names
                parameter_names = ['cf size', 'cf amp', 'bold baseline', 'vert', 'rsq', 'x', 'y']
                cf_gauss = pd.DataFrame(cf_gauss, columns=parameter_names)
                cf_gauss.index = range(cf_gauss.shape[0])
                self.arranged_data[key] = cf_gauss
                
            elif self.arranged_data[key].shape[1] == 11:
                sub = key.split('_')[3]
                brainmask = np.load(f'{maskdir}_{sub}.npy')
                cf_dn = np.zeros([118584, 11])
                cf_dn[:] = np.nan
                cf_dn[brainmask] = self.arranged_data[key]
                parameter_names = ['cf size', 'cf amp', 'bold baseline', 'vert', 'surr amp', 'surr size', 'B', 'D', 'rsq', 'x', 'y']
                cf_dn = pd.DataFrame(cf_dn, columns=parameter_names)
                cf_dn.index = range(cf_dn.shape[0])
                self.arranged_data[key] = cf_dn

            # make pRF dataframe
            
            elif self.arranged_data[key].shape[1] == 8:
                parameter_names = ['x', 'y', 'pRF size', 'amplitude', 'bold-baseline', 'hrf1', 'hrf2', 'rsq']
                prf_gauss = pd.DataFrame(get_cortex(self.arranged_data[key]), columns=parameter_names)
                prf_gauss.index = range(prf_gauss.shape[0])
                self.arranged_data[key] = prf_gauss

            
            elif self.arranged_data[key].shape[1] == 12:
                parameter_names = ['x', 'y', 'pRF size', 'amplitude', 'bold baseline', 'surr amp', 'surr size', 'B', 'D', 'hrf1', 'hrf2', 'rsq']
                prf_dn = pd.DataFrame(get_cortex(self.arranged_data[key]), columns=parameter_names)
                prf_dn.index = range(prf_dn.shape[0])
                self.arranged_data[key] = prf_dn
            
            else:
                raise ValueError(f'number of parameters in {key} non-standard, cant assign parameter names')   









# get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# get the directory where the utils are located
utils_dir = script_dir.replace('/cfdn/cfdn', '/cfdn/data/CF_fit_utils')


# add possibility to give own idxs and where to find the indices later
# this works only on the node230 server
idxs = h5py.File(f'{utils_dir}/cifti_indices.hdf5', "r")
lidxs = np.array(idxs['Left_indices'])
ridxs = np.array(idxs['Right_indices'])
allidxs = np.concatenate([lidxs, ridxs])

def get_cortex(dat):
    """get just the cortex from 170k HCP data."""

    l, r, = dat[lidxs], dat[ridxs]

    # Replace the minus 1
    l[lidxs == -1] = np.zeros_like(l[lidxs == -1])
    r[ridxs == -1] = np.zeros_like(r[ridxs == -1])

    # Last dimension time.
    # l, r = l.T, r.T

    data = np.concatenate([l, r])
    return data

def split_cortex(dat):
    """split the data from 170k HCP data into two hemispheres."""
    l, r, = dat[lidxs], dat[ridxs]

    # Replace the minus 1
    l[lidxs == -1] = np.zeros_like(l[lidxs == -1])
    r[ridxs == -1] = np.zeros_like(r[ridxs == -1])


    return l, r

def split_given_size(a, size):
    """split the data into chunks of size."""
    return np.split(a, np.arange(size,len(a),size))



def write_newcifti(filename, old_cifti, data_arr):
    """
    Saves a CIFTI file that has a new size of timepoints
    
    Parameters
    ----------
    filename : str
        name of output CIFTI file
    old_cifti : CIFTI file
        previous nibabel.cifti2.cifti2.Cifti2Image
    data_arr : array
        data to be stored as vector or matrix (shape: n_timepoints x n_voxels)
        or a scalar value for each voxel (shape: n_voxels)
    """

    # in case of data_arr where you have one value for each voxel (e.g. std for each voxel)
    if len(data_arr.shape) == 1: 
        matrix = cifti2.Cifti2Matrix()
        brain_model = old_cifti.header.get_axis(1)
        matrix.append(brain_model.to_mapping(0))
        newheader = cifti2.Cifti2Header(matrix)
        img = cifti2.Cifti2Image(data_arr, newheader)
        img.to_filename(filename)
        
    # in case of same or different 2 dimensional shape (e.g. removing first 3 TR of timeseries)
    else:
        start = old_cifti.header.get_axis(0).start
        step = old_cifti.header.get_axis(0).step
        brain_model = old_cifti.header.get_axis(1)  
        size = data_arr.shape[0]
        series = cifti2.SeriesAxis(start, step, size)
        brain_model = old_cifti.header.get_axis(1)
        newheader = cifti2.Cifti2Header.from_axes((series, brain_model))

        img = cifti2.Cifti2Image(data_arr, newheader)
        img.to_filename(filename)