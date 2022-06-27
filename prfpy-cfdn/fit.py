import os
import numpy as np
import nibabel as nb
from nibabel import cifti2
from nilearn.surface import load_surf_data
import glob
import cortex.polyutils
import h5py

idxs = h5py.File('/tank/shared/timeless/atlases/cifti_indices.hdf5', "r")
lidxs = np.array(idxs['Left_indices'])
ridxs = np.array(idxs['Right_indices'])
allidxs = np.concatenate([lidxs, ridxs])

def get_cortex(dat):
    l, r, = dat[lidxs], dat[ridxs]

    # Replace the minus 1
    l[lidxs == -1] = np.zeros_like(l[lidxs == -1])
    r[ridxs == -1] = np.zeros_like(r[ridxs == -1])

    # Last dimension time.
    # l, r = l.T, r.T

    data = np.concatenate([l, r])
    return data

def split_cortex(dat):
    l, r, = dat[lidxs], dat[ridxs]

    # Replace the minus 1
    l[lidxs == -1] = np.zeros_like(l[lidxs == -1])
    r[ridxs == -1] = np.zeros_like(r[ridxs == -1])

    # Last dimension time.
    # l, r = l.T, r.T

#     data = np.concatenate([l, r])
    return l, r


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



class participant:
    """Participant is a single participant, containing the timecourses of all voxels,
    and multiple runs
    """
    
    def __init__(self, subject, derivatives_dir, scalar_dir):
        """__init__

        constructor for participant

        Parameters
        ----------
        subject : subject id in form of 'sub-01'
        derivatives_dir : directory where pybest preprocessed data can be found
        scalar_dir : directory where mean and std of data should be stored or is stored for
        further processing.
        """ 
    
        self.subject = subject
        self.derivatives_dir = derivatives_dir
        self.scalar_dir = scalar_dir

        self.prep_dir = os.path.join(self.derivatives_dir, 'fmriprep', self.subject)
        self.preproc_dir = os.path.join(self.derivatives_dir, 'pybest', self.subject)

        ses = []
        for root,dirs,files in os.walk(f'{self.prep_dir}/'):
            for a_dir in dirs:
                if ("ses" in a_dir):
                    ses.append(a_dir)

        self.sessions = ses
    
    def get_scalars(self):
        os.makedirs(self.scalar_dir, exist_ok=True)
        
        for sessions in self.sessions:
            for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                runs = runs+1
                datvol = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_bold.dtseries.nii')
                dat = np.asanyarray(datvol.dataobj)
                write_newcifti(os.path.join(scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'), datvol, dat.mean(axis=0))
                write_newcifti(os.path.join(scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_std.scalar.nii'), datvol, dat.std(axis=0))
        
        
        
    def convert_to_psc(self):
        """
        Converts zsc data from pybest preprocessing to psc using scalars 
        """
        
        
        
    def get_folds(self, remove_edge=False, smoothing=5):
        """
        Removes vertices that are at the edge of the scanbox
        """
        data = []
        
        if remove_edge is False:
            for sessions in self.sessions:            
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.preproc_dir + f'/{sessions}/preproc/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-preproc_bold.npy')                
                    data.append(hp)
                    
        else:
            subject = 'hcp_999999'
            # First we need to import the surfaces for this subject
            surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "inflated")]
            
            for sessions in self.sessions:            
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.preproc_dir + f'/{sessions}/preproc/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-preproc_bold.npy')      
                    

                    datvols = nb.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_std.scalar.nii'))
                    std = np.asanyarray(datvols.dataobj)


                    datvolm = nb.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'))
                    mean = np.asanyarray(datvolm.dataobj)

                    binmapL, binmapR = split_cortex(np.multiply(mean > 0, 1))
                    smooth_map = np.concatenate([surfs[0].smooth(binmapL, smoothing), surfs[0].smooth(binmapR, smoothing)])
                    mean2 = np.copy(get_cortex(mean))
                    mean2[smooth_map<=0.9] = np.nan
                    mean[allidxs] = mean2

                #     hp_psc = psc((hp * std.reshape(-1,1))+mean.reshape(-1,1))
                #     dn_psc = psc((dn * std.reshape(-1,1))+mean.reshape(-1,1))

                    hp[np.isnan(mean)] = np.nan                              
                    data.append(hp)
        
        self.data_train = np.nanmean(np.array(data[::2]), axis=0)
        self.data_test = np.nanmean(np.array(data[1::2]), axis=0)
                
                
        
        