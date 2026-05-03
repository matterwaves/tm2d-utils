import numpy as np
import tm2d

ice_mpf_A = 3000 # [A] mean free path for inelastic scattering in ice

def projected_potential_from_coords_and_pose(
        coords, pose, pixel_size, pot_shape,
        specimen_thickness_A=0):

    # get template
    template_atomic = tm2d.TemplateAtomic(
        pot_shape, # [pix]
        coords, # [A]
        disable_sigma_e=True,
        fuse_ctf_convolution=False,
        disable_ctf=True,
    )
    
    # get projected potential
    ta = template_atomic.make_template(
        rotations = pose, # [deg]
        pixel_size = pixel_size, # [A]
    )
    proj_pot = ta.read_real(0)[0].copy() # [V*A] projected potential
    
    # add projected potential of ice
    if specimen_thickness_A > 0:
        proj_pot += 3.6 * specimen_thickness_A # [V*A]

    return proj_pot # [V*A]

def pdf_from_coords_and_pose(
        coords, pose, ctf_params, pixel_size, pot_shape,
        specimen_thickness_A=0, inelastic_mfp_A=ice_mpf_A,
        offset_background=True):

    # get template
    template_atomic = tm2d.TemplateAtomic(
        pot_shape, # [pix]
        coords, # [A]
    )
    
    # calculate pdf with dc=0
    pdf = template_atomic.make_template(
        rotations = pose, # [deg]
        pixel_size = pixel_size, # [A]
        ctf_params = ctf_params,
    ).read_real(0)[0].copy()

    # shift background to 1
    if offset_background:
        pdf += 1

    # scale for solvent
    if specimen_thickness_A > 0:
        pdf *= np.exp(-specimen_thickness_A / inelastic_mfp_A)

    return pdf

def atom_histogram_from_coords_and_pose(
        coords, pose, pixel_size, pot_shape):

    template_atomic = tm2d.TemplateAtomic(
        pot_shape,
        coords,
        count_atoms=True,
    )
    ta = template_atomic.make_template(
        rotations = pose,
        pixel_size = pixel_size,
    )
    atom_histogram = ta.read_real(0)[0]

    return atom_histogram

def dose_A2_to_pixel(dose_A2, pixel_size):
    return dose_A2 * pixel_size ** 2 # convert [e/A^2] to [e/pix]

def get_image_from_pdf(pdf, dose_per_A2, pixel_size, snr=1):
    dose_per_pix = dose_A2_to_pixel(dose_per_A2, pixel_size) # [e/pix]
    white_noise = np.random.normal(0, np.sqrt(1 / snr), size=pdf.shape)
    pdf_noisy = pdf + white_noise # add white noise
    pdf_noisy[pdf_noisy < 0] = 0 # enforce non-negativity
    return dose_per_pix * pdf_noisy # [e]