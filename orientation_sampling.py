import numpy as np

from typing import Union

class OrientationRegion:
    symmetry: str
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    psi_min: float
    psi_max: float

    def __init__(self,
                 symmetry: str = "C1",
                 phi_min: float = 0,
                 phi_max: float = 360,
                 theta_min: float = 0,
                 theta_max: float = 180,
                 psi_min: float = -180,
                 psi_max: float = 180) -> None:
        self.symmetry = symmetry
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.psi_min = psi_min
        self.psi_max = psi_max

def resolve_region_arg(region_arg: Union[OrientationRegion, str, None]) -> OrientationRegion:
    if region_arg is None:
        return OrientationRegion()

    if isinstance(region_arg, str):
        return OrientationRegion(symmetry=region_arg)

    return region_arg

def angles_to_directions(angle_list):
    angles_radian = np.deg2rad(angle_list)

    result_array = np.zeros((angle_list.shape[0], 3), dtype=np.float32)
    result_array[:, 0] = np.sin(angles_radian[:, 1]) * np.cos(angles_radian[:, 0])
    result_array[:, 1] = np.sin(angles_radian[:, 1]) * np.sin(angles_radian[:, 0])
    result_array[:, 2] = np.cos(angles_radian[:, 1])

    return result_array

def make_normed_vector(vector):
    my_vec = np.array(vector, dtype=np.float32)
    my_vec /= np.sqrt(np.sum(my_vec ** 2))
    return my_vec

def apply_OOPA_limits(out_of_plane_angles: np.ndarray, region: OrientationRegion):
    return out_of_plane_angles[
        (out_of_plane_angles[:, 0] >= region.phi_min) &
        (out_of_plane_angles[:, 0] <= region.phi_max) &
        (out_of_plane_angles[:, 1] >= region.theta_min) &
        (out_of_plane_angles[:, 1] <= region.theta_max)
    ]

def apply_OOPA_symmetry(out_of_plane_angles: np.ndarray, region: OrientationRegion):
    symmetry_type = region.symmetry.upper()[0]
    symmetry_number = int(region.symmetry[1:]) if len(region.symmetry) > 1 else 0

    if symmetry_type == "C":
        if symmetry_number == 0:
            raise ValueError("Symmetry number must be greater than 0 for symmetry type C.")
        
        return out_of_plane_angles[
            out_of_plane_angles[:, 0] <= (360 / symmetry_number)
        ]
    elif symmetry_type == "D":
        if symmetry_number == 0:
            raise ValueError("Symmetry number must be greater than 0 for symmetry type D.")

        if symmetry_number == 1:
            return out_of_plane_angles[
                (out_of_plane_angles[:, 1] <= 90)
            ]
        
        shifted_out_of_plane_angles = np.copy(out_of_plane_angles)
        shifted_out_of_plane_angles[:, 0] = np.mod(shifted_out_of_plane_angles[:, 0] + 180, 360)
        shifted_out_of_plane_angles[:, 0] = shifted_out_of_plane_angles[:, 0] - 180

        shifted_out_of_plane_angles = shifted_out_of_plane_angles[
            (shifted_out_of_plane_angles[:, 0] <= (90 + 180 / symmetry_number)) &
            (shifted_out_of_plane_angles[:, 0] >= (90 - 180 / symmetry_number)) &
            (shifted_out_of_plane_angles[:, 1] <= 90)
        ]

        shifted_out_of_plane_angles[:, 0] = np.mod(shifted_out_of_plane_angles[:, 0], 360)

        return shifted_out_of_plane_angles

    elif symmetry_type == "T":
        _3_fold_axis_1_by_3_fold_axis_2 = make_normed_vector([-0.942809, 0, 0])
        _3_fold_axis_2_by_3_fold_axis_3 = make_normed_vector([0.471405, 0.272165, 0.7698])
        _3_fold_axis_3_by_3_fold_axis_1 = make_normed_vector([0.471404, 0.816497, 0])

        directions_array = angles_to_directions(out_of_plane_angles)

        return out_of_plane_angles[
            (out_of_plane_angles[:, 0] >= 90) &
            (out_of_plane_angles[:, 0] <= 150) &
            (np.dot(directions_array, _3_fold_axis_1_by_3_fold_axis_2) >= 0) &
            (np.dot(directions_array, _3_fold_axis_2_by_3_fold_axis_3) >= 0) &
            (np.dot(directions_array, _3_fold_axis_3_by_3_fold_axis_1) >= 0)
        ]
    elif symmetry_type == "O":
        _3_fold_axis_1_by_3_fold_axis_2 = make_normed_vector([0, -1, 1])
        _3_fold_axis_2_by_4_fold_axis = make_normed_vector([1, 1, 0])
        _4_fold_axis_by_3_fold_axis_1 = make_normed_vector([-1, 1, 0])

        directions_array = angles_to_directions(out_of_plane_angles)
        
        return out_of_plane_angles[
            (out_of_plane_angles[:, 0] >= 45) &
            (out_of_plane_angles[:, 0] <= 135) &
            (out_of_plane_angles[:, 1] <= 90) &
            (np.dot(directions_array, _3_fold_axis_1_by_3_fold_axis_2) >= 0) &
            (np.dot(directions_array, _3_fold_axis_2_by_4_fold_axis) >= 0) &
            (np.dot(directions_array, _4_fold_axis_by_3_fold_axis_1) >= 0)
        ]
    elif symmetry_type == "I":
        _5_fold_axis_1_by_5_fold_axis_2 = make_normed_vector([0, 1, 0])
        _5_fold_axis_2_by_3_fold_axis = make_normed_vector([-0.4999999839058737,
                                                            -0.8090170074556163,
                                                            0.3090169861701543])
        _3_fold_axis_by_5_fold_axis_1 = make_normed_vector([0.4999999839058737,
                                                            -0.8090170074556163,
                                                            0.3090169861701543])
        
        directions_array = angles_to_directions(out_of_plane_angles)

        return out_of_plane_angles[
            (np.dot(directions_array, _5_fold_axis_1_by_5_fold_axis_2) >= 0) &
            (np.dot(directions_array, _5_fold_axis_2_by_3_fold_axis) >= 0) &
            (np.dot(directions_array, _3_fold_axis_by_5_fold_axis_1) >= 0)
        ]
    else:
        raise ValueError("Invalid symmetry type. Must be one of C, D, T, O, or I.")

def apply_IPA_limits(in_plane_angles: np.ndarray, region: OrientationRegion):
    return in_plane_angles[
        (in_plane_angles >= region.psi_min) &
        (in_plane_angles <= region.psi_max)
    ]

def get_OOPA_from_region_limits(region: OrientationRegion, angular_step_size: float):
    phi_values = np.arange(region.phi_min, region.phi_max, angular_step_size)
    theta_values = np.arange(region.theta_min, region.theta_max, angular_step_size)
    return np.array(np.meshgrid(phi_values, theta_values)).T.reshape(-1, 2)

def get_IPA_from_region_limits(region: OrientationRegion, psi_step_size: float):
    return np.arange(region.psi_min, region.psi_max, psi_step_size)

def make_orientations_array(out_of_plane_angles: np.ndarray, in_plane_angles: np.ndarray):
    orientations_grid = np.zeros((out_of_plane_angles.shape[0], in_plane_angles.shape[0], 3), dtype=np.float32)
    orientations_grid[:, :, :2] = out_of_plane_angles[:, None, :]
    orientations_grid[:, :, 2] = in_plane_angles[None, :]

    return orientations_grid.reshape(-1, 3)

def make_rotation_array_from_OOPA_and_region(out_of_plane_angles: np.ndarray,  psi_step_size: float, region: Union[OrientationRegion, str, None], merge_out_and_in_plane: bool = True):
    actual_region = resolve_region_arg(region)

    out_of_plane_angles = apply_OOPA_limits(out_of_plane_angles, actual_region)
    out_of_plane_angles = apply_OOPA_symmetry(out_of_plane_angles, actual_region)
    in_plane_angles = get_IPA_from_region_limits(actual_region, psi_step_size)

    if merge_out_and_in_plane:
        return make_orientations_array(out_of_plane_angles, in_plane_angles)
    
    return out_of_plane_angles, in_plane_angles

def get_orientations_mercator(angular_step_size: float, psi_step_size: float, region: Union[OrientationRegion, str, None] = None, merge_out_and_in_plane: bool = True):
    return make_rotation_array_from_OOPA_and_region(
        get_OOPA_from_region_limits(resolve_region_arg(region), angular_step_size), 
        psi_step_size, 
        region,
        merge_out_and_in_plane
    )

def get_orientations_cube(angular_step_size: float, psi_step_size: float, region: Union[OrientationRegion, str, None] = None, merge_out_and_in_plane: bool = True):
    values = np.linspace(-1, 1, int(np.ceil(90 / angular_step_size)) + 1) # We add 1 to make sure we don't undersample

    X, Y = np.meshgrid(values, values)

    faces = np.zeros(shape=(6, X.flatten().shape[0], 3), dtype=np.float32)

    faces[0, :, 0] = -1
    faces[0, :, 1] = X.flatten()
    faces[0, :, 2] = Y.flatten()

    faces[1, :, 0] = 1
    faces[1, :, 1] = X.flatten()
    faces[1, :, 2] = Y.flatten()

    faces[2, :, 0] = X.flatten()
    faces[2, :, 1] = -1
    faces[2, :, 2] = Y.flatten()

    faces[3, :, 0] = X.flatten()
    faces[3, :, 1] = 1
    faces[3, :, 2] = Y.flatten()

    faces[4, :, 0] = X.flatten()
    faces[4, :, 1] = Y.flatten()
    faces[4, :, 2] = -1

    faces[5, :, 0] = X.flatten()
    faces[5, :, 1] = Y.flatten()
    faces[5, :, 2] = 1
    
    # Combine all the faces into one array
    points = faces.reshape(-1, 3)

    # Normalize the points
    points /= np.linalg.norm(points, axis=1)[:, None]

    # Compute the phi and theta angles
    phi_values = np.mod(np.arctan2(points[:, 2], points[:, 0]) * 180 / np.pi, 360)
    theta_values = np.arccos(points[:, 1]) * 180 / np.pi

    out_of_plane_angles = np.array([phi_values, theta_values]).T

    return make_rotation_array_from_OOPA_and_region(out_of_plane_angles, psi_step_size, region, merge_out_and_in_plane)

def get_orientations_healpix(angular_step_size: float, psi_step_size: float, region: Union[OrientationRegion, str, None] = None, merge_out_and_in_plane: bool = True):
    try:
        import healpy as hp

        steradian_step = np.deg2rad(angular_step_size) ** 2
        npix = int(np.ceil(4 * np.pi / steradian_step))

        nside_guess = np.sqrt(npix / 12)
        nside = int(2 ** np.round(np.log2(nside_guess)))
        pixels = np.arange(npix)
        theta_values, phi_values = hp.pix2ang(nside, pixels)

        out_of_plane_angles = np.array([np.rad2deg(phi_values), np.rad2deg(theta_values)]).T

        return make_rotation_array_from_OOPA_and_region(out_of_plane_angles, psi_step_size, region, merge_out_and_in_plane)
    except ImportError:
        raise ImportError("Healpy is required to use this function. Please install it using 'pip install tm2d[healpy]'.")