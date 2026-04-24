import numpy as np
import vkdispatch as vd 
import tm2d
import tm2d.utilities as tu
import tm2d.utilities

from ..plan import Template

from typing import Tuple

from ..ctf import CTFParams

def get_pdf(template_atomic, rotation, pixel_size, ctf_params, thickness_A=0):
    template = template_atomic.make_template(
        rotations=rotation,
        pixel_size=pixel_size,
        ctf_params=ctf_params
    )
    im0 = template.read_real(0)[0]
    im1 = im0 + 1 # set baseline to 1
    return im1 * np.exp(-1 * thickness_A / 3000) # apply simple inelastic scattering model

def get_im_from_pdf(pdf, dose_per_A2, pix_size, snr=1):
    dose_per_pix = tu.optics_functions.dose_A2ToPix(dose_per_A2, pix_size)
    white_noise = np.random.normal(0, np.sqrt(1 / snr), size=pdf.shape)
    pdf_noisy = pdf + white_noise # add white noise
    pdf_noisy[pdf_noisy < 0] = 0 # enforce non-negativity
    return dose_per_pix * pdf_noisy # [e]