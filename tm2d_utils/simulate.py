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
        specimen_thickness_A=0, inelastic_mfp_A=ice_mpf_A):

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