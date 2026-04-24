import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

def freq_axes(shape: tuple, pixel_size: float = None):
    if pixel_size is None:
        pixel_size = 1.0
    srow = np.fft.fftshift(np.fft.fftfreq(shape[0], d=pixel_size))
    scol = np.fft.fftshift(np.fft.fftfreq(shape[1], d=pixel_size))
    return srow, scol

def radius_grid(srow: np.ndarray, scol: np.ndarray):
    sr, sc = np.meshgrid(srow, scol, indexing='ij')
    return np.sqrt(sr * sr + sc * sc)

def radial_bins_linear(s_nyq: float, nfreq: int):
    return np.linspace(0.0, s_nyq, nfreq)

def annulus_bins(srad: np.ndarray, srad_ax: np.ndarray):
    """
    Map each pixel's radius to a bin index for annulus averaging.
    Bin edges are centered on srad_ax samples.
    """
    nfreq = srad_ax.size
    s_nyq = srad_ax[-1] if nfreq == 1 else srad_ax[-1]
    # interior edges at midpoints, plus 0 and last+step for right edge
    last_step = (srad_ax[-1] - srad_ax[-2]) if nfreq > 1 else (s_nyq / max(nfreq, 1))
    edges = np.concatenate([[0.0], 0.5 * (srad_ax[1:] + srad_ax[:-1]), [srad_ax[-1] + last_step]])
    idx = np.searchsorted(edges, srad, side="right") - 1
    return np.clip(idx, 0, nfreq - 1)

def azimuthal_average_from_radius(arr2d: np.ndarray, srad: np.ndarray, nfreq: int):
    """Azimuthal average of a 2D array using annulus binning on a provided radius map."""
    s_nyq = np.max(srad)
    srad_ax = radial_bins_linear(s_nyq, nfreq)

    idx = annulus_bins(srad, srad_ax)
    counts = np.bincount(idx.ravel(), minlength=nfreq)
    sums = np.bincount(idx.ravel(), weights=arr2d.ravel(), minlength=nfreq)
    avg1d = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

    # evaluate the 1d curve onto the 2d grid
    avg2d = np.interp(srad, srad_ax, avg1d, left=avg1d[0], right=avg1d[-1])

    return srad_ax, avg1d, avg2d

def azimuthal_average(arr2d: np.ndarray, nfreq: int = 512, pixel_size: float = None):
    """Wrapper for azimuthal_average_from_radius."""
    srow, scol = freq_axes(arr2d.shape, pixel_size)
    srad = radius_grid(srow, scol)
    srad_ax, avg1d, avg2d = azimuthal_average_from_radius(arr2d, srad, nfreq)
    avg1d = np.nan_to_num(avg1d, copy=False)
    avg2d = np.nan_to_num(avg2d, copy=False)
    return srad_ax, avg1d, avg2d, srow, scol

def raised_cosine_taper(srad: np.ndarray, s_nyq: float, r0_frac: float, r1_frac: float):
    """
    Low-frequency raised-cosine taper.
    r0_frac, r1_frac are fractions of nyquist.
    """
    r0 = r0_frac * s_nyq
    r1 = r1_frac * s_nyq
    if r1 <= r0:
        return np.ones_like(srad)
    t = np.clip((srad - r0) / (r1 - r0), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)

def get_psd2d(im: np.ndarray, pad_factor: int = 0):
    """Energy-preserving 2d psd with optional zero-padding."""
    pad = int(pad_factor * max(im.shape))
    im_pad = np.pad(im, ((pad, pad), (pad, pad)), mode='constant') if pad > 0 else im
    im_ft = np.fft.fftshift(np.fft.fft2(im_pad))
    return np.abs(im_ft)**2 / im_pad.size

def get_psd_all(im: np.ndarray, pixel_size: float | None = None, nfreq: int = 512, pad_factor: int = 0):
    """Energy-preserving 2d psd and azimuthal averages (annulus binning)."""
    psd2d = get_psd2d(im, pad_factor=pad_factor)
    srad_ax, psd1d, psd1d2d, srow, scol = azimuthal_average(psd2d, nfreq=nfreq, pixel_size=pixel_size)
    # hygiene
    psd2d   = np.nan_to_num(psd2d, copy=False)
    psd1d   = np.nan_to_num(psd1d, copy=False)
    psd1d2d = np.nan_to_num(psd1d2d, copy=False)
    return psd2d, srad_ax, psd1d, psd1d2d, srow, scol

def get_psd_stats(psd2d: np.ndarray):
    """Recover |mean| and population variance from an energy-preserving, fftshifted PSD."""
    energy = float(np.sum(psd2d))
    mean_sq = float(psd2d[psd2d.shape[0] // 2, psd2d.shape[1] // 2]) / psd2d.size
    var = float(np.sum(psd2d)) / psd2d.size - mean_sq # (mean of squares) - (square of mean)
    var = max(var, 0.0)  # avoid round-off negatives
    mean_abs = float(np.sqrt(max(mean_sq, 0.0)))
    return mean_abs, var

def get_hpf(shape: tuple, pixel_size: float, cuton_start: float, cuton_end: float):
    srow, scol = freq_axes(shape, pixel_size)
    srad = radius_grid(srow, scol)
    s_nyq = 0.5 / pixel_size
    return raised_cosine_taper(srad, s_nyq, r0_frac=cuton_start / s_nyq, r1_frac=cuton_end / s_nyq)

def apply_fourier_filt2d(im: np.ndarray, filt2d: np.ndarray):
    """Apply a provided 2d filter in fourier space (fftshifted)."""
    im_ft = np.fft.fftshift(np.fft.fft2(im))
    im_ft_filt = im_ft * filt2d
    im_filt = np.real(np.fft.ifft2(np.fft.ifftshift(im_ft_filt)))
    return im_filt

def build_whitening_filter(
    psd2d: np.ndarray,
    srow: np.ndarray,
    scol: np.ndarray,
    nfreq: int = 1024,
    eps: float = 1e-10,
    r0_frac: float = 0.0,
    r1_frac: float = 2e-2,
    mode: str = '1d',
    double_whiten: bool = False,
):
    """
    Construct a whitening filter from a psd.
      mode:
        - '1d' -> azimuthal average, then evaluate back on grid
        - '2d' -> use full 2d psd as-is
      double_whiten:
        - False: 1/sqrt(psd)
        - True: 1/psd
    """
    srad = radius_grid(srow, scol)
    s_nyq = min(np.max(np.abs(srow)), np.max(np.abs(scol)))
    if mode == '2d':
        psd_map = psd2d
    elif mode == '1d':
        _, _, psd_map = azimuthal_average_from_radius(psd2d, srad, nfreq)

    taper = raised_cosine_taper(srad, s_nyq, r0_frac, r1_frac)
    base = np.maximum(psd_map, eps)
    filt = taper / (base if double_whiten else np.sqrt(base))
    return filt

def high_pass_filter_image(
    im: np.ndarray,
    pixel_size: float,
    cuton_start: float,
    cuton_end: float,
    return_filter: bool = False,
):
    """
    High-pass filter an image in fourier space using a raised-cosine taper.
    - cuton_start, cuton_end are in [1/A]
    """
    hpf = get_hpf(im.shape, pixel_size, cuton_start, cuton_end)
    im_filt = apply_fourier_filt2d(im, hpf)
    return im_filt, hpf if return_filter else im_filt

def whiten_image(
    im: np.ndarray,
    pixel_size: float | None = None,
    nfreq: int = 1024,
    eps: float = 1e-10,
    r0_frac: float = 0.0,
    r1_frac: float = 0.02,
    mode: str = '1d',
    double_whiten: bool = False,
    return_filter: bool = False,
):
    """
    Whiten an image in fourier space using either a 1d (azimuthally-averaged) or 2d psd.
    - pixel_size=None gives index units (no physical scaling needed for averaging).
    """
    psd2d = get_psd2d(im)
    srow, scol = freq_axes(im.shape, pixel_size)

    filt2d = build_whitening_filter(
        psd2d, srow, scol, nfreq=nfreq, eps=eps,
        r0_frac=r0_frac, r1_frac=r1_frac,
        mode=mode, double_whiten=double_whiten
    )
    im_filt = apply_fourier_filt2d(im, filt2d)
    return (im_filt, filt2d) if return_filter else im_filt

def read_pixel(buff: Buff[c64],
               x_ind: vc.ShaderVariable,
               y_ind: vc.ShaderVariable,
               batch_index: vc.ShaderVariable):
    result = vc.new_vec2_register(0.0)

    vc.if_all(x_ind >= 0, y_ind >= 0, x_ind < buff.shape[2], y_ind < buff.shape[1])
    result[:] = buff[batch_index * buff.shape[2] * buff.shape[1] + y_ind * buff.shape[2] + x_ind].to_dtype(vc.v2)
    vc.end()

    return result

def interp_pixel(
        buff: Buff[c64],
        x_coord: vc.ShaderVariable,
        y_coord: vc.ShaderVariable,
        batch_index: vc.ShaderVariable):
    x_ind0 = vc.floor(x_coord).to_dtype(vc.i32)
    y_ind0 = vc.floor(y_coord).to_dtype(vc.i32)

    x_ind1 = vc.ceil(x_coord).to_dtype(vc.i32)
    y_ind1 = vc.ceil(y_coord).to_dtype(vc.i32)

    value00 = read_pixel(buff, x_ind0, y_ind0, batch_index)
    value01 = read_pixel(buff, x_ind0, y_ind1, batch_index)
    value10 = read_pixel(buff, x_ind1, y_ind0, batch_index)
    value11 = read_pixel(buff, x_ind1, y_ind1, batch_index)

    dx = x_coord - x_ind0
    dy = y_coord - y_ind0

    r1 = value00 * (1 - dx) + value10 * dx
    r2 = value01 * (1 - dx) + value11 * dx

    interpolated_value = vc.new_vec2_register(r1 * (1 - dy) + r2 * dy)

    return interpolated_value.x * interpolated_value.x + interpolated_value.y * interpolated_value.y

@vd.reduce.map_reduce(reduction=vd.reduce.SubgroupAdd, axes=[2, ])
def azimuthal_sum(buff: Buff[c64]) -> vc.f32:
    ind = vd.reduce.mapped_io_index()

    template_index = ind % (buff.shape[2] * buff.shape[1])
    batch_index = ind // (buff.shape[2] * buff.shape[1])

    radius_index = vc.new_float_register(template_index / buff.shape[2])

    result_value = vc.new_float_register(0.0)

    vc.if_statement(radius_index < buff.shape[2])
    angle_index = vc.new_float_register(np.pi/2 - (np.pi * (template_index % buff.shape[2])) / buff.shape[2])

    x_coord = vc.new_float_register(radius_index * vc.cos(angle_index))
    y_coord = vc.new_float_register(radius_index * vc.sin(angle_index))

    vc.if_statement(y_coord < 0)
    y_coord += buff.shape[1]  # Adjust y_coord to be positive
    vc.end()

    result_value[:] = interp_pixel(buff, x_coord, y_coord, batch_index)
    vc.end()

    return result_value

@vd.shader("buff.size")
def apply_whiten_filter(buff: Buff[c64], sums_buffer: Buff[f32], return_filter: Const[i32] = 0):
    ind = vc.global_invocation_id().x

    template_index = ind % (buff.shape[2] * buff.shape[1])
    batch_index = ind // (buff.shape[2] * buff.shape[1])

    x_index = template_index // buff.shape[2]
    y_index = template_index % buff.shape[2]

    radius0 = vc.new_float_register(vc.sqrt(x_index * x_index + y_index * y_index))

    x_index = buff.shape[1] - x_index

    radius1 = vc.new_float_register(vc.sqrt(x_index * x_index + y_index * y_index))

    true_radius = vc.min(radius0, radius1)

    rad_floor = vc.floor(true_radius).to_dtype(vc.i32)
    rad_ceil = vc.ceil(true_radius).to_dtype(vc.i32)

    value_floor = sums_buffer[rad_floor + batch_index * buff.shape[1]] / buff.shape[2]
    value_ceil = sums_buffer[rad_ceil + batch_index * buff.shape[1]] / buff.shape[2]

    drad = true_radius - rad_floor
    psd = value_floor * (1 - drad) + value_ceil * drad

    filter_value = vc.new_complex64_register(0.0)

    with vc.if_block(psd > 1e-10):
        filter_value.real = vc.sqrt(1 / psd)
        filter_value.imag = filter_value.real

    with vc.if_block(return_filter == 1):
        buff[ind] = filter_value
    
    with vc.else_block():
        buff[ind].real *= filter_value.real
        buff[ind].imag *= filter_value.imag

def whiten_buffer(buffer: vd.Buffer, return_filter: bool = False):
    vd.fft.rfft2(buffer)
    sums_buffer = azimuthal_sum(buffer)
    apply_whiten_filter(buffer, sums_buffer, 1 if return_filter else 0)
    if not return_filter:
        vd.fft.irfft2(buffer)