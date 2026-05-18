import tm2d
import numpy as np

from .file_loading import DensityData

def run_tm2d_atomic_pixels(
                        micrographs: np.ndarray,
                        param_set: tm2d.ParamSet,
                        template_box_size: tuple[int, int],
                        atomic_coords: np.ndarray,
                        template_batch_size: int = 4,
                        enable_progress_bar: bool = False):

    assert micrographs.ndim == 3, "Micrographs must be a 3D array (num_micrographs, height, width)."
    assert len(template_box_size) == 2, "Template box size must be a tuple of two integers (height, width)."
    assert atomic_coords.ndim == 2 and atomic_coords.shape[1] == 3, "Atomic coordinates must be a 2D array with shape (N, 3)."

    results = tm2d.ResultsPixel(micrographs.shape)

    plan = tm2d.Plan(
        template=tm2d.TemplateAtomic(
            shape=template_box_size,
            atomic_coords=atomic_coords
        ),
        comparator=tm2d.ComparatorCrossCorrelation(
            shape=micrographs.shape,
            template_shape=template_box_size
        ),
        results=results,
        ctf_params=param_set.ctf_set.ctf_params,
        template_batch_size=template_batch_size,
        enable_rotation_weights=param_set.rotations_weights is not None
    )

    plan.set_data(micrographs)

    plan.run(param_set, enable_progress_bar=enable_progress_bar)

    return results

def run_tm2d_atomic_params(
                        micrographs: np.ndarray,
                        param_set: tm2d.ParamSet,
                        template_box_size: tuple[int, int],
                        atomic_coords: np.ndarray,
                        template_batch_size: int = 4,
                        enable_progress_bar: bool = False):

    assert micrographs.ndim == 3, "Micrographs must be a 3D array (num_micrographs, height, width)."
    assert len(template_box_size) == 2, "Template box size must be a tuple of two integers (height, width)."
    assert atomic_coords.ndim == 2 and atomic_coords.shape[1] == 3, "Atomic coordinates must be a 2D array with shape (N, 3)."

    results = tm2d.ResultsParam(micrographs.shape[0], param_set.get_total_count())

    plan = tm2d.Plan(
        template=tm2d.TemplateAtomic(
            shape=template_box_size,
            atomic_coords=atomic_coords
        ),
        comparator=tm2d.ComparatorCrossCorrelation(
            shape=micrographs.shape,
            template_shape=template_box_size
        ),
        results=results,
        ctf_params=param_set.ctf_set.ctf_params,
        template_batch_size=template_batch_size,
        enable_rotation_weights=param_set.rotations_weights is not None
    )

    plan.set_data(micrographs)

    plan.run(param_set, enable_progress_bar=enable_progress_bar)

    return results

def run_tm2d_density_pixels(
                        micrographs: np.ndarray,
                        param_set: tm2d.ParamSet,
                        density: DensityData,
                        template_batch_size: int = 4,
                        enable_progress_bar: bool = False):

    assert micrographs.ndim == 3, "Micrographs must be a 3D array (num_micrographs, height, width)."

    results = tm2d.ResultsPixel(micrographs.shape)

    template=tm2d.TemplateDensity(
        density_array=density.density_array,
        density_pixel_size=density.pixel_size
    )

    plan = tm2d.Plan(
        template,
        comparator=tm2d.ComparatorCrossCorrelation(
            shape=micrographs.shape,
            template_shape=template.get_shape()
        ),
        results=results,
        ctf_params=param_set.ctf_set.ctf_params,
        template_batch_size=template_batch_size,
        enable_rotation_weights=param_set.rotations_weights is not None
    )

    plan.set_data(micrographs)

    plan.run(param_set, enable_progress_bar=enable_progress_bar)

    return results

def run_tm2d_density_params(
                        micrographs: np.ndarray,
                        param_set: tm2d.ParamSet,
                        density: DensityData,
                        template_batch_size: int = 4,
                        enable_progress_bar: bool = False):

    assert micrographs.ndim == 3, "Micrographs must be a 3D array (num_micrographs, height, width)."

    results = tm2d.ResultsParam(micrographs.shape[0], param_set.get_total_count())

    template=tm2d.TemplateDensity(
        density_array=density.density_array,
        density_pixel_size=density.pixel_size
    )

    plan = tm2d.Plan(
        template,
        comparator=tm2d.ComparatorCrossCorrelation(
            shape=micrographs.shape,
            template_shape=template.get_shape()
        ),
        results=results,
        ctf_params=param_set.ctf_set.ctf_params,
        template_batch_size=template_batch_size,
        enable_rotation_weights=param_set.rotations_weights is not None
    )

    plan.set_data(micrographs)

    plan.run(param_set, enable_progress_bar=enable_progress_bar)

    return results