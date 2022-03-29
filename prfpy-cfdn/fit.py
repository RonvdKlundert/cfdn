import os
import numpy as np
import nibabel as nb
from nibabel import cifti2
from nilearn.surface import load_surf_data
import glob


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
    
    def __init__(self, subject, derivatives_dir):
    
    
    
        self.subject = subject
        self.derivatives_dir = derivatives_dir

        self.prep_dir = os.path.join(self.derivatives_dir, 'fmriprep', self.subject)
        self.preproc_dir = os.path.join(self.derivatives_dir, 'pybest', self.subject)

        ses = []
        for root,dirs,files in os.walk(self.prep_dir+'/'):
            for a_dir in dirs:
                if ("ses" in a_dir):
                    ses.append(a_dir)

        self.sessions = ses
    
    def get_scalars(self, scalar_dir):
        for sessions in self.sessions:
            for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                runs = runs+1
                datvol = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_bold.dtseries.nii')
                dat = np.asanyarray(datvol.dataobj)
                write_newcifti(os.path.join(scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'), datvol, dat.mean(axis=0))
        
        
    
#     def preprocess(self):
        





class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"