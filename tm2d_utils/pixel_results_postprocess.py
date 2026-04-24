import numpy as np
import tm2d

def get_pixel_z_scores(results: tm2d.ResultsPixel):
    sum_cross = results.get_sum_cross()
    sum2_cross = results.get_sum2_cross()
    count = results.get_templates_count()
    mip = results.get_mip()

    cross_mean = sum_cross / count # per-pixel mean of cross-correlation
    cross_variance = sum2_cross / count - cross_mean * cross_mean # per-pixel variance of cross-correlation

    return (mip - cross_mean) / np.sqrt(cross_variance)

def get_locations_and_indicies_of_best_match(results: tm2d.ResultsPixel, optimize_by: str = "mip"):
    location_of_best_match = []
    index_of_params_match = []

    mip = results.get_mip()
    best_index_array = results.get_best_index_array()

    z_scores = None

    if optimize_by == 'z_score':
        z_scores = get_pixel_z_scores(results)

    for i in range(results.micrograph_count):
        if optimize_by == 'mip':
            location_of_best_match.append(np.unravel_index(np.argmax(mip[i]), mip.shape[1:]))
        elif optimize_by == 'z_score':
            location_of_best_match.append(np.unravel_index(np.argmax(z_scores[i]), z_scores.shape[1:]))
        else:
            raise ValueError("Invalid optimize_by value. Must be 'mip' or 'z_score'.")

        index_of_params_match.append(best_index_array[i][location_of_best_match[i]])

    return location_of_best_match, index_of_params_match
