import os
import numpy as np

from .cf_utils import get_cortex, split_cortex, split_given_size
from nilearn.surface import load_surf_data
import glob
import cortex.polyutils
from nibabel import cifti2
from scipy import stats
import numpy as np
import scipy as sp
import nibabel as nb
import pickle
from numpy.linalg import inv
import h5py


idxs = h5py.File('/tank/shared/timeless/atlases/cifti_indices.hdf5', "r")
lidxs = np.array(idxs['Left_indices'])
ridxs = np.array(idxs['Right_indices'])
allidxs = np.concatenate([lidxs, ridxs])

class participant:
    """Participant is a single participant, containing the timecourses of all voxels,
    and multiple runs
    """
    
    def __init__(self, subject, derivatives_dir=None, scalar_dir=None, ldtr_dir=None):
        """__init__

        constructor for participant

        Parameters
        ----------
        subject : subject id in form of 'sub-01'
        derivatives_dir : directory where pybest/fmriprep/freesurfer folders can be found
        scalar_dir : directory where mean and std of data should be stored or is stored for
        further processing.
        """ 
    
        self.subject = subject
        self.derivatives_dir = derivatives_dir
        self.scalar_dir = scalar_dir
        self.ldtr_dir = ldtr_dir
        
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
                write_newcifti(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'), datvol, dat.mean(axis=0))
                write_newcifti(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_std.scalar.nii'), datvol, dat.std(axis=0))
                
                
                
                
    def get_scalars_native(self):
        os.makedirs(self.scalar_dir, exist_ok=True)
        
        for sessions in self.sessions:
            for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                runs = runs+1
                datvol_l = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnative_hemi-L_bold.func.gii')
                datvol_r = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnative_hemi-R_bold.func.gii')
                
                
                
                dat = np.concatenate([datvol_l.agg_data(), datvol_r.agg_data()]).T
                
                np.save(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_fsnativeLR_mean.npy'), dat.mean(axis=0))
                np.save(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_fsnativeLR_std.npy'), dat.std(axis=0))                    
               
     
    
    
    def linear_detrend_psc(self, X_conv, idx_baseline, omit_TR):
        os.makedirs(self.ldtr_dir, exist_ok=True)
        
        for sessions in self.sessions:
            os.makedirs(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}', exist_ok=True)
            for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                runs = runs+1
                datvol = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_bold.dtseries.nii')
                voxel_signal = np.asanyarray(datvol.dataobj)[omit_TR:,:]
                # print(voxel_signal.shape)
                betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
                ldt = X_conv[:,1][:,np.newaxis] @ betas_conv[1,:][np.newaxis,:]
                ldt_data = voxel_signal - ldt
                baseline = np.mean(ldt_data[idx_baseline,:], axis=0)

                psc_data = ((ldt_data - baseline) / baseline) * 100
                
                np.save(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-linear-detrend_bold_psc.npy', psc_data)
                
    
    
    def linear_detrend_psc_native(self, X_conv, idx_baseline, omit_TR):
        os.makedirs(self.ldtr_dir, exist_ok=True)
        
        for sessions in self.sessions:
            os.makedirs(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}', exist_ok=True)
            for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                runs = runs+1
                datvol_l = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnative_hemi-L_bold.func.gii')
                datvol_r = nb.load(self.prep_dir + f'/{sessions}/func/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnative_hemi-R_bold.func.gii')
                
                self.lidx = datvol_l.agg_data().shape[0]
                self.ridx = datvol_r.agg_data().shape[0]

                dat = np.concatenate([datvol_l.agg_data(), datvol_r.agg_data()]).T
                
                voxel_signal = dat[omit_TR:,:]
                # print(voxel_signal.shape)
                betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
                ldt = X_conv[:,1][:,np.newaxis] @ betas_conv[1,:][np.newaxis,:]
                ldt_data = voxel_signal - ldt
                baseline = np.mean(ldt_data[idx_baseline,:], axis=0)

                psc_data = ((ldt_data - baseline) / baseline) * 100
                
                np.save(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnativeLR_den-300k_desc-linear-detrend_bold_psc.npy', psc_data)
        
        
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
                    hp = np.load(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-linear-detrend_bold_psc.npy')

                    datvols = nb.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_std.scalar.nii'))
                    std = np.asanyarray(datvols.dataobj)


                    datvolm = nb.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'))
                    mean = np.asanyarray(datvolm.dataobj)

                    binmapL, binmapR = split_cortex(np.multiply(mean > 0, 1))
                    smooth_map = np.concatenate([surfs[0].smooth(binmapL, smoothing), surfs[1].smooth(binmapR, smoothing)])
                    mean2 = np.copy(get_cortex(mean))
                    mean2[smooth_map<=0.9] = np.nan
                    mean[allidxs] = mean2

                #     hp_psc = psc((hp * std.reshape(-1,1))+mean.reshape(-1,1))
                #     dn_psc = psc((dn * std.reshape(-1,1))+mean.reshape(-1,1))

                    hp[np.isnan(mean)] = np.nan                              
                    data.append(hp)
        self.data = np.array(data)
        self.data_train = np.nanmean(np.array(data[::2]), axis=0)
        self.data_test = np.nanmean(np.array(data[1::2]), axis=0)

        
        
        
        
        
        
    def get_folds_detrend(self, remove_edge=False, smoothing=5):
        """
        Removes vertices that are at the edge of the scanbox
        """
        data = []
        
        if remove_edge is False:
            for sessions in self.sessions:            
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-linear-detrend_bold_psc.npy').T              
                    data.append(hp)
                    
        else:
            subject = 'hcp_999999'
            # First we need to import the surfaces for this subject
            surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "inflated")]
            
            for sessions in self.sessions:
                print(f'processing {sessions}')
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsLR_den-170k_desc-linear-detrend_bold_psc.npy').T 
                    



                    datvolm = nb.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_mean.scalar.nii'))
                    mean = np.asanyarray(datvolm.dataobj)

                    binmapL, binmapR = split_cortex(np.multiply(mean > 0, 1))
                    smooth_map = np.concatenate([surfs[0].smooth(binmapL, smoothing), surfs[1].smooth(binmapR, smoothing)])
                    mean2 = np.copy(get_cortex(mean))
                    mean2[smooth_map<=0.9] = np.nan
                    mean[allidxs] = mean2



                    hp[np.isnan(mean)] = np.nan                              
                    data.append(hp)
        self.data = np.array(data)
        self.data_train = np.nanmean(np.array(data[::2]), axis=0)
        self.data_test = np.nanmean(np.array(data[1::2]), axis=0)
        
        
        
    def get_folds_detrend_native(self, remove_edge=False, smoothing=5):
        """
        Removes vertices that are at the edge of the scanbox
        """
        data = []

        
        
        if remove_edge is False:
            for sessions in self.sessions:            
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnativeLR_den-300k_desc-linear-detrend_bold_psc.npy').T              
                    data.append(hp)
                    
        else:
            subject = self.subject
            # First we need to import the surfaces for this subject
            surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "inflated")]
            
            for sessions in self.sessions:
                print(f'processing {sessions}')
                for runs in range(len(glob.glob(self.prep_dir + f'/{sessions}/func/*.dtseries.nii'))):
                    runs = runs+1
                    hp = np.load(self.ldtr_dir + f'/{self.subject}' + f'/{sessions}' + f'/{self.subject}_{sessions}_task-prf_run-{runs}_space-fsnativeLR_den-300k_desc-linear-detrend_bold_psc.npy').T 
                    
                   


                    mean = np.load(os.path.join(self.scalar_dir, self.subject, f'{self.subject}_{sessions}_task-prf_run-{runs}_fsnativeLR_mean.npy'))
                   

                    binmapL, binmapR = np.multiply(mean > 0, 1)[:self.lidx], np.multiply(mean > 0, 1)[self.lidx:]
                    smooth_map = np.concatenate([surfs[0].smooth(binmapL, smoothing), surfs[1].smooth(binmapR, smoothing)])
                    mean2 = np.copy(mean)
                    mean2[smooth_map<=0.9] = np.nan
                    
                    
                  

                    hp[np.isnan(mean2)] = np.nan                              
                    data.append(hp)
        self.data = np.array(data)
        self.data_train = np.nanmean(np.array(data[::2]), axis=0)
        self.data_test = np.nanmean(np.array(data[1::2]), axis=0)



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


        
class Ciftihandler(object):

    """Ciftihandler
        This is a utility for loading, splitting and saving cifi data.
    """

    def __init__(self,dfile="/scratch/2019/visual/hcp_movie/subjects/999999/tfMRI_MOVIE1_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii"):        
        self.dfile=dfile
        self.get_brain_model()
        self.load_data()
        
    def load_data(self):
        """ load_data
        """ 
        
        self.img=nb.load(self.dfile)
        self.header=self.img.nifti_header
        self.brain_models = self.img.header.get_axis(1)  # Assume we know this
        
    def get_brain_model(self):
        self.brainmodel= cifti.read(self.dfile)[1][1]
        
    def get_data(self):
        """ get_data
        Loads the cifti data into memory.
        """ 
        
        self.load_data()
        self.data = self.img.get_fdata(dtype=np.float32)
        
        
    def surf_data_from_cifti(self,data, axis, surf_name):
        """ surf_data_from_cifti
        Gets surface data from cifti file.
        """ 
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        if not hasattr(self,'vtx_indices'):
            self.vtx_indices=[]
        for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
            
            if name == surf_name:                                 # Just looking for a surface
                data = data.T[data_indices]    
                vtx_indices = model.vertex
                self.vtx_indices.append(data_indices)
                # Generally 1-N, except medial wall vertices
                surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                #print(surf_data.shape)
                surf_data[vtx_indices] = data
                return surf_data
            
        raise ValueError(f"No structure named {surf_name}")

    def volume_from_cifti(self,data, axis, header):
        """ volume_from_cifti
        Gets volume data from cifti file
        """ 
        
        self.affine=axis.affine
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        data = data.T[axis.volume_mask]                          # Assume brainmodels axis is last, move it to front
        self.volmask = axis.volume_mask                               # Which indices on this axis are for voxels?
        vox_indices = tuple(axis.voxel[axis.volume_mask].T)      # ([x0, x1, ...], [y0, ...], [z0, ...])
        self.test=axis.voxel
        vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                            dtype=data.dtype)
        vol_data[vox_indices] = data  
        self.vox_flat=np.where(self.volmask)[0]
        self.vox_indices=vox_indices
        
        return nb.Nifti1Image(vol_data, axis.affine) 
    
    

    def decompose_cifti(self,data):
        """ decompose_cifti
        Decomposes the data in the cifti_file
        """ 
        
        self.subcortex=self.volume_from_cifti(data,self.brain_models, header=self.img.nifti_header)
        self.surf_left=self.surf_data_from_cifti(data,self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        self.surf_right=self.surf_data_from_cifti(data,self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")
        self.surface=np.vstack([self.surf_left,self.surf_right])
        
        if data.ndim==1:
            self.surface=np.concatenate([surf_left,surf_right])
        else: 
            self.surface=np.vstack([surf_left,surf_right])
            
        
        
    def decompose_data(self,data):
        subcortex=self.volume_from_cifti(data,self.brain_models, header=self.header)
        surf_left=self.surf_data_from_cifti(data,self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        surf_right=self.surf_data_from_cifti(data,self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")
        
        if data.ndim==1:
            surface=np.concatenate([surf_left,surf_right])
        else: 
            surface=np.vstack([surf_left,surf_right])
            
        return surface,subcortex
    
    def recompose_data(self,ldat,rdat,sdat):
        
        empt=np.zeros(self.data.shape[-1])
        test_dat=np.array(range(self.data.shape[-1]))
        split_testdat=self.decompose_data(test_dat)
        
        linds=split_testdat[0][:int((split_testdat[0].shape[-1])/2)]
        rinds=split_testdat[0][int((split_testdat[0].shape[-1])/2):]

        subc=sdat[self.vox_indices]
        empt[self.vox_flat]=subc
        empt[linds]=ldat
        empt[rinds]=rdat
    
        return empt.astype(int)
    
    def save_cii(self,data, names,filename):
        """ save_cii
        Saves data out to a cifti file, using the brainmodel.
        """ 
        cifti.write(filename, data, (cifti.Scalar.from_names(names), self.brainmodel))
        
    def save_subvol(self,data,filename):
        """ save_subvol
        Saves subcortical data out to a nifti file. 
        """ 
        
        nb.save(data,filename) 
