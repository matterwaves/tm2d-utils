import numpy as np
import pandas as pd

from . import optics_functions

class ParticleStack:
    def __init__(
        self,
        im_orig: list,
        pixel_size_orig: float,
        Cs_nom_mm: float = 0.01,
        HT: float = 300e3, # [V]
        amp_contrast: float = 0.07,
        mic_fpath: list | None = None, # full path to source micrograph
        inds_in_mic: list | None = None, # [pix] if you want to slice image out of the micrograph
        phi_in: list | None = None,
        theta_in: list | None = None,
        psi_in: list | None = None,
        defocus_u_in: list | None = None,
        defocus_v_in: list | None = None,
        defocus_ang_in: list | None = None,
        phase_shift_in: list | None = None,
        rln_star_fpath: list | None = None, # full path to source star
        rln_optics_df: pd.DataFrame | None = None, # if you want to initialize optics parameters from a Relion optics table
        rln_optics_group_id: list | None = None, # if not provided, will use first group
        mag_matrix: list | None = None, # gets overriden by rln_optics_df
        rln_odd_zernike: list | None = None,
        rln_even_zernike: list | None = None,
    ):
        self.im_orig = list(im_orig) # archived
        self.pixel_size_orig = pixel_size_orig # archived
        self.Cs_nom_mm = Cs_nom_mm
        self.HT = HT
        self.amp_contrast = amp_contrast
        
        self.mic_fpath = [None] * len(im_orig) if mic_fpath is None else list(mic_fpath)
        self.inds_in_mic = [None] * len(im_orig) if inds_in_mic is None else list(inds_in_mic)
        self.rln_star_fpath = [None] * len(im_orig) if rln_star_fpath is None else list(rln_star_fpath)
        
        self.phi_in = [None] * len(im_orig) if phi_in is None else list(phi_in)
        self.theta_in = [None] * len(im_orig) if theta_in is None else list(theta_in)
        self.psi_in = [None] * len(im_orig) if psi_in is None else list(psi_in)
        self.defocus_u_in = [None] * len(im_orig) if defocus_u_in is None else list(defocus_u_in)
        self.defocus_v_in = [None] * len(im_orig) if defocus_v_in is None else list(defocus_v_in)
        self.defocus_ang_in = [None] * len(im_orig) if defocus_ang_in is None else list(defocus_ang_in)
        self.phase_shift_in = [None] * len(im_orig) if phase_shift_in is None else list(phase_shift_in)

        # allocate modifiable image and pixel size to be updated if preprocessing
        self.im = list(im_orig)
        self.pixel_size = pixel_size_orig
        
        self.wlen = optics_functions.get_eWlenFromHT(HT) # [A] electron wavelength
        
        # handle higher-order aberrations
        self.rln_optics_df = rln_optics_df
        self.mag_matrix = [np.eye(2)] * len(im_orig) if mag_matrix is None else list(mag_matrix) # may be overwritten momentarily
        self.rln_odd_zernike = [np.zeros(6)] * len(im_orig) if rln_odd_zernike is None else list(rln_odd_zernike) # may be overwritten momentarily
        self.rln_even_zernike = [np.zeros(9)] * len(im_orig) if rln_even_zernike is None else list(rln_even_zernike) # may be overwritten momentarily
        self.Cs_ref_mm = [Cs_nom_mm] * len(im_orig) # may be overwritten momentarily
              
        if rln_optics_df is not None:
            self.rln_optics_group_id = [1] * len(im_orig) if rln_optics_group_id is None else list(rln_optics_group_id)
            for p_ind in range(len(im_orig)):
                self.update_from_optics_df(p_ind, group_id=self.rln_optics_group_id[p_ind])
        
        # prepare for parameter estimation
        self.phi_out = [None] * len(im_orig)
        self.theta_out = [None] * len(im_orig)
        self.psi_out = [None] * len(im_orig)
        self.defocus_u_out = [None] * len(im_orig)
        self.defocus_v_out = [None] * len(im_orig)
        self.defocus_ang_out = [None] * len(im_orig)
        self.phase_shift_out = [None] * len(im_orig)
        self.z_score_out = [None] * len(im_orig)
        self.mip_out = [None] * len(im_orig)
        self.snr_out = [None] * len(im_orig)
        self.p_value_out = [None] * len(im_orig)

    def add_particle(
        self,
        im_orig,
        mic_fpath=None,
        inds_in_mic=None,
        phi_in=None,
        theta_in=None,
        psi_in=None,
        defocus_u_in=None,
        defocus_v_in=None,
        defocus_ang_in=None,
        phase_shift_in=None,
        rln_star_fpath=None,
        rln_optics_df=None,
        rln_optics_group_id=None,
        mag_matrix=None,
        rln_odd_zernike=None,
        rln_even_zernike=None,
    ):
        self.im_orig.append(im_orig)
        self.pixel_size_orig = self.pixel_size_orig # assuming same pixel size for new particle, otherwise need to handle differently
        self.mic_fpath.append(mic_fpath)
        self.inds_in_mic.append(inds_in_mic)
        self.phi_in.append(phi_in)
        self.theta_in.append(theta_in)
        self.psi_in.append(psi_in)
        self.defocus_u_in.append(defocus_u_in)
        self.defocus_v_in.append(defocus_v_in)
        self.defocus_ang_in.append(defocus_ang_in)
        self.phase_shift_in.append(phase_shift_in)
        self.rln_star_fpath.append(rln_star_fpath)
        self.rln_optics_group_id.append(rln_optics_group_id if rln_optics_group_id is not None else 1)
        self.mag_matrix.append(mag_matrix if mag_matrix is not None else np.eye(2))
        self.rln_odd_zernike.append(rln_odd_zernike if rln_odd_zernike is not None else np.zeros(6))
        self.rln_even_zernike.append(rln_even_zernike if rln_even_zernike is not None else np.zeros(9))
        self.Cs_ref_mm.append(self.Cs_nom_mm) # will get updated if appropriate
        
        # initialize modifiable image
        self.im.append(im_orig)
        self.pixel_size = self.pixel_size_orig
        
        # handle higher-order aberrations if optics dataframe provided
        if rln_optics_df is not None:
            group_id = rln_optics_group_id if rln_optics_group_id is not None else 1
            self.update_from_optics_df(len(self.im)-1, group_id=group_id) # update from optics dataframe for the new particle
        else:
            # otherwise, just append provided values or defaults
            self.mag_matrix.append(mag_matrix if mag_matrix is not None else np.eye(2))
            self.rln_odd_zernike.append(rln_odd_zernike if rln_odd_zernike is not None else np.zeros(6))
            self.rln_even_zernike.append(rln_even_zernike if rln_even_zernike is not None else np.zeros(9))
            self.Cs_ref_mm.append(self.Cs_nom_mm)

    def get_substack(self, particle_inds):
        substack = ParticleStack(
            [self.im_orig[i] for i in particle_inds],
            self.pixel_size_orig,
            self.Cs_nom_mm,
            self.HT,
            self.amp_contrast,
            [self.mic_fpath[i] for i in particle_inds],
            [self.inds_in_mic[i] for i in particle_inds],
            [self.phi_in[i] for i in particle_inds],
            [self.theta_in[i] for i in particle_inds],
            [self.psi_in[i] for i in particle_inds],
            [self.defocus_u_in[i] for i in particle_inds],
            [self.defocus_v_in[i] for i in particle_inds],
            [self.defocus_ang_in[i] for i in particle_inds],
            [self.phase_shift_in[i] for i in particle_inds],
            [self.rln_star_fpath[i] for i in particle_inds],
            self.rln_optics_df,
            [self.rln_optics_group_id[i] for i in particle_inds],
            [self.mag_matrix[i] for i in particle_inds],
            [self.rln_odd_zernike[i] for i in particle_inds],
            [self.rln_even_zernike[i] for i in particle_inds],
        )
        return substack
    
    def __len__(self):
        return len(self.im_orig)
    
    def get_substacks_inds_by_field(self, field='mic_fpath'):
        idxs_by_value = {}
        for i, s in enumerate(getattr(self, field)):
            idxs_by_value.setdefault(s, []).append(i)
            unique_and_inds = list(idxs_by_value.items())  # [(value, [i1, i2, ...]), ...]
        return unique_and_inds

def get_micograph_groups(df_particles):
    micrograph_groups = {k: g for k, g in df_particles.groupby('rlnMicrographName')}
    return micrograph_groups