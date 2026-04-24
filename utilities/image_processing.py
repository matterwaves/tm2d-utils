import numpy as np

from .whitener import whiten_image

def normalize_image(im, n_std=5, remove_outliers=True):
    im_ref = im.copy()
    if remove_outliers:
        outliers_idxs = np.abs(im - im.mean()) > n_std * im.std()
        im_ref[outliers_idxs] = im_ref.mean()
    return (im_ref - im_ref.mean()) / im_ref.std()

def downsample_image(
    im: np.ndarray,
    N: int,
    conserve: str | None = 'match_sum',  # 'match_sum' | 'theoretical' | None
    gaussian_std: float | None = None, # frequency-pixel sigma for soft edge
    norm_ortho: bool = False
) -> np.ndarray:
    """
    Fourier downsampling for square 2D images with optional antialiasing and sum conservation.

    Parameters
    ----------
    im : np.ndarray
        2D real array (H, W). Must be square.
    N : int
        Target side length (N x N).
    rescale : bool
        Legacy alias: if True and conserve is None, uses 'match_sum'.
    conserve : str or None
        'match_sum'     -> scale so sum(out) == sum(in)
        'theoretical'   -> scale by (N/M)^2 (continuous integral view)
        None            -> no scaling
    gaussian_std : float or None
        If set, multiplies cropped spectrum by a centered gaussian (reduces ringing).
    norm_ortho : bool
        If True, use norm='ortho' in FFTs.

    Returns
    -------
    np.ndarray
        Real-valued (N, N) array.
    """
    def _center_slice(M, N):
        cM = M // 2
        cN = N // 2
        return slice(cM - cN, cM + cN) if N % 2 == 0 else slice(cM - cN, cM + cN + 1)
    
    im = np.asarray(im)

    _norm = 'ortho' if norm_ortho else None

    # forward fft (centered)
    im_ft = np.fft.fftshift(np.fft.fft2(im, norm=_norm))

    # center crop in frequency
    sl = _center_slice(im.shape[0], N)
    im_ft_crop = im_ft[sl, sl]

    # (optional) soft gaussian edge in frequency
    if gaussian_std is not None and gaussian_std > 0:
        yy, xx = np.ogrid[:N, :N]
        cy = N // 2
        cx = N // 2
        r2 = (yy - cy)**2 + (xx - cx)**2
        g = np.exp(-0.5 * r2 / (gaussian_std**2))
        im_ft_crop = im_ft_crop * g

    # inverse fft and real part
    im_ds = np.fft.ifft2(np.fft.ifftshift(im_ft_crop), norm=_norm).real

    # sum conservation
    if conserve:
        src_sum = float(im.sum())
        dst_sum = float(im_ds.sum())
        if dst_sum != 0.0:
            if conserve == 'match_sum':
                im_ds *= (src_sum / dst_sum)
            elif conserve == 'theoretical':
                s = N / im.shape[0]
                target_sum = src_sum * (s**2)
                im_ds *= (target_sum / dst_sum)
            else:
                raise ValueError("conserve must be 'match_sum', 'theoretical', or None.")

    return im_ds

def process_raw_micrograph(
        data: np.ndarray, 
        whiten: bool = True, 
        normalize_input: bool = True, 
        remove_outliers: bool = True, 
        n_std = 5):
    assert data.ndim == 2

    if normalize_input:
        reference_image = data.copy()
        test_image_normalized: np.ndarray = normalize_image(reference_image, n_std=n_std, remove_outliers=remove_outliers)
    else:
        test_image_normalized: np.ndarray = data.copy() # don't actually normalize

    if whiten:
        test_image_normalized = whiten_image(test_image_normalized)

    return test_image_normalized

def center_crop(im, edge_buffer_pix):
    if edge_buffer_pix <= 0:
        return im.copy()
    return im[edge_buffer_pix:-edge_buffer_pix, edge_buffer_pix:-edge_buffer_pix]

def center_pad(im, edge_buffer_pix, edge_value=np.nan):
    if edge_buffer_pix <= 0:
        return im.copy()
    return np.pad(im, pad_width=edge_buffer_pix, mode='constant', constant_values=edge_value)

def replace_edges(im, edge_buffer_pix, edge_value=np.nan):
    if edge_buffer_pix <= 0:
        return im.copy()
    im_out = im.copy()
    im_out[:edge_buffer_pix, :] = edge_value
    im_out[-edge_buffer_pix:, :] = edge_value
    im_out[:, :edge_buffer_pix] = edge_value
    im_out[:, -edge_buffer_pix:] = edge_value
    return im_out