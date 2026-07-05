import os
import re

import numpy as np
import pandas as pd

from . import optics_functions



def parse_rln_vector(v):
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return np.array([], dtype=float)
        toks = [t for t in re.split(r"[\s,;]+", s.replace(",", " ")) if t]
        return np.array([float(t) for t in toks], dtype=float)
    return np.asarray(v, dtype=float).ravel().copy()


def extract_particle(mic, rln_coord, box_len):
    half_box = int(box_len) // 2
    row_start = int(rln_coord[1] - half_box)
    row_stop = int(rln_coord[1] + half_box)
    col_start = int(rln_coord[0] - half_box)
    col_stop = int(rln_coord[0] + half_box)
    if row_start < 0 or col_start < 0 or row_stop > mic.shape[0] or col_stop > mic.shape[1]:
        return None
    return mic[row_start:row_stop, col_start:col_stop]


def convert_rlnBeamTilt_to_rlnOddZernike(beam_tilt_x_mrad, beam_tilt_y_mrad, Cs_mm, HT=300e3, rln_odd_zernike_len=6):
    wlen = optics_functions.get_eWlenFromHT(HT)
    scale = float(Cs_mm) * 20000.0 * wlen * wlen * np.pi
    d = np.zeros(max(int(rln_odd_zernike_len), 5), dtype=float)
    z3x = -scale * float(beam_tilt_x_mrad) / 3.0
    z3y = -scale * float(beam_tilt_y_mrad) / 3.0
    d[1] += 2.0 * z3x
    d[0] += 2.0 * z3y
    d[4] += z3x
    d[3] += z3y
    return d


def get_delta_Cs_from_rlnEvenZernike(rln_even_zernike, HT=300e3):
    z40 = rln_even_zernike[6] if rln_even_zernike.size > 6 else 0.0
    wlen = optics_functions.get_eWlenFromHT(HT)
    return (12.0 * z40) / (np.pi * wlen**3) * 1e-7


def get_star_of_type(job_type):
    return {
        "Refine3D": "run_data.star",
        "CtfRefine": "particles_ctf_refine.star",
        "Polish": "shiny.star",
    }.get(job_type, "run_data.star")

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
              
        self.rln_optics_group_id = [1] * len(im_orig) if rln_optics_group_id is None else list(rln_optics_group_id)
        if rln_optics_df is not None:
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

    def update_from_optics_df(self, particle_ind, group_id=1):
        optics_data = self.rln_optics_df.iloc[int(group_id) - 1]
        beam_tilt_x_mrad = optics_data["rlnBeamTiltX"] if "rlnBeamTiltX" in optics_data else 0.0
        beam_tilt_y_mrad = optics_data["rlnBeamTiltY"] if "rlnBeamTiltY" in optics_data else 0.0
        odd_zernike = parse_rln_vector(optics_data["rlnOddZernike"]) if "rlnOddZernike" in optics_data else np.zeros(6)
        differential_odd_zernike = convert_rlnBeamTilt_to_rlnOddZernike(
            beam_tilt_x_mrad,
            beam_tilt_y_mrad,
            self.Cs_nom_mm,
            self.HT,
            rln_odd_zernike_len=len(odd_zernike),
        )
        odd_zernike += differential_odd_zernike

        m00 = optics_data["rlnMagMatrix_00"] if "rlnMagMatrix_00" in optics_data else 1.0
        m01 = optics_data["rlnMagMatrix_01"] if "rlnMagMatrix_01" in optics_data else 0.0
        m10 = optics_data["rlnMagMatrix_10"] if "rlnMagMatrix_10" in optics_data else 0.0
        m11 = optics_data["rlnMagMatrix_11"] if "rlnMagMatrix_11" in optics_data else 1.0
        mag_matrix = np.array([[m00, m01], [m10, m11]])

        even_zernike = parse_rln_vector(optics_data["rlnEvenZernike"]) if "rlnEvenZernike" in optics_data else np.zeros(9)
        delta_Cs_mm = get_delta_Cs_from_rlnEvenZernike(even_zernike, self.HT)
        Cs_refined_mm = self.Cs_nom_mm + delta_Cs_mm

        self.mag_matrix[particle_ind] = mag_matrix
        self.rln_odd_zernike[particle_ind] = odd_zernike
        self.rln_even_zernike[particle_ind] = even_zernike
        self.Cs_ref_mm[particle_ind] = Cs_refined_mm

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

def get_micrograph_groups(df_particles):
    micrograph_groups = {k: g for k, g in df_particles.groupby('rlnMicrographName')}
    return micrograph_groups


def get_dfs_from_session(rln_session, job_type="Refine3D"):
    ws = rln_session.workspace
    star_fpath = os.path.join(ws.get_job_dir(job_type, getattr(rln_session, job_type)), get_star_of_type(job_type))
    star_data = ws.read_starfile(star_fpath)
    return star_data["optics"], star_data["particles"], star_fpath


get_micograph_groups = get_micrograph_groups

def read_stack_from_dfs(df_optics, df_particles, star_fpath, workspace, job_type="Refine3D", n_particles=None):
    pixel_size = df_optics["rlnMicrographOriginalPixelSize"].iloc[0]
    box_len_pix = df_optics["rlnImageSize"].iloc[0]
    Cs_nom_mm = df_optics["rlnSphericalAberration"].iloc[0]
    HT = df_optics["rlnVoltage"].iloc[0] * 1e3
    amp_contrast = df_optics["rlnAmplitudeContrast"].iloc[0]
    micrograph_groups = get_micrograph_groups(df_particles)
    n_particles_so_far = 0
    if n_particles is None:
        n_particles = np.inf

    stack = ParticleStack([], pixel_size, Cs_nom_mm, HT, amp_contrast, rln_optics_df=df_optics)
    try:
        workspace.open_remote()
        for mic_fpath_rel, df_mic in micrograph_groups.items():
            mic_fpath = os.path.join(workspace.root_dir, mic_fpath_rel)
            mic = workspace.read_mrc_remote(mic_fpath)
            rln_coords_orig = df_mic[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy()
            rln_angles = df_mic[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy()
            rln_defocus_u = df_mic["rlnDefocusU"].to_numpy()
            rln_defocus_v = df_mic["rlnDefocusV"].to_numpy()
            rln_defocus_angle = df_mic["rlnDefocusAngle"].to_numpy()
            rln_phase_shift = df_mic["rlnPhaseShift"].to_numpy() if "rlnPhaseShift" in df_mic.columns else np.zeros(len(df_mic))
            if "rlnOriginXAngst" in df_mic.columns and "rlnOriginYAngst" in df_mic.columns:
                rln_coords_diff_A = df_mic[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy()
            else:
                rln_coords_diff_A = np.zeros_like(rln_coords_orig)
            rln_optics_group = df_mic["rlnOpticsGroup"].to_numpy() if "rlnOpticsGroup" in df_mic.columns else np.ones(len(df_mic), dtype=int)
            rln_coords = rln_coords_orig - rln_coords_diff_A / pixel_size

            for particle_ind in range(len(rln_coords)):
                part_im = extract_particle(mic, rln_coords[particle_ind], box_len_pix)
                if part_im is not None:
                    stack.add_particle(
                        part_im.astype(np.float64),
                        mic_fpath=mic_fpath,
                        inds_in_mic=rln_coords[particle_ind],
                        phi_in=rln_angles[particle_ind, 0],
                        theta_in=rln_angles[particle_ind, 1],
                        psi_in=rln_angles[particle_ind, 2],
                        defocus_u_in=rln_defocus_u[particle_ind],
                        defocus_v_in=rln_defocus_v[particle_ind],
                        defocus_ang_in=rln_defocus_angle[particle_ind],
                        phase_shift_in=rln_phase_shift[particle_ind],
                        rln_star_fpath=star_fpath,
                        rln_optics_df=df_optics,
                        rln_optics_group_id=int(rln_optics_group[particle_ind]),
                    )
                    n_particles_so_far += 1
                if n_particles_so_far >= n_particles:
                    break
            if n_particles_so_far >= n_particles:
                break
    finally:
        workspace.close_remote()
    return stack


def read_stack_from_session(rln_session, job_type="Refine3D", n_particles=None):
    df_optics, df_particles, star_fpath = get_dfs_from_session(rln_session, job_type=job_type)
    return read_stack_from_dfs(df_optics, df_particles, star_fpath, rln_session.workspace, job_type=job_type, n_particles=n_particles)
