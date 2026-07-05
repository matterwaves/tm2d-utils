import numpy as np

from .rotation_matrices import get_cisTEM_rotation_matrix


__all__ = [
    'euler_to_rotation_matrix',
    'rotation_geodesic_error',
    'euler_geodesic_error',
    'point_group_symmetry_matrices',
    'filter_library_by_geodesic_distance',
]


def euler_to_rotation_matrix(angles):
    """
    Convert cisTEM-style Euler angles to a 3x3 rotation matrix.

    Parameters
    ----------
    angles : array-like
        Euler angles in degrees, ordered as:

            [phi, theta, psi]

        May be shape (3,) or (N, 3).

    Returns
    -------
    rotation_matrix : np.ndarray
        3x3 rotation matrix, or (N, 3, 3) for batched input.
    """

    angles = np.asarray(angles, dtype=np.float32)

    if angles.shape == (3,):
        rotation_matrix = get_cisTEM_rotation_matrix(angles[None, :])[0, :3, :3]
        return rotation_matrix.astype(np.float64)

    if angles.ndim == 2 and angles.shape[1] == 3:
        rotation_matrices = get_cisTEM_rotation_matrix(angles)[:, :3, :3]
        return rotation_matrices.astype(np.float64)

    raise ValueError('angles must have shape (3,) or (N, 3), ordered as [phi, theta, psi].')


def rotation_geodesic_error(rotation_matrix_0, rotation_matrix_1, degrees=True):
    """
    Compute the geodesic distance between two 3D rotation matrices.

    The inputs may be shape (3, 3) or batched shape (N, 3, 3).
    """

    rotation_matrix_0 = np.asarray(rotation_matrix_0, dtype=np.float64)
    rotation_matrix_1 = np.asarray(rotation_matrix_1, dtype=np.float64)

    if rotation_matrix_0.shape != rotation_matrix_1.shape:
        raise ValueError('rotation_matrix_0 and rotation_matrix_1 must have the same shape.')

    if rotation_matrix_0.shape == (3, 3):
        relative_rotation = rotation_matrix_0.T @ rotation_matrix_1
        cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if degrees:
            angle = np.rad2deg(angle)
        return float(angle)

    if rotation_matrix_0.ndim == 3 and rotation_matrix_0.shape[1:] == (3, 3):
        relative_rotation = np.matmul(
            np.swapaxes(rotation_matrix_0, -1, -2),
            rotation_matrix_1,
        )
        cos_angle = (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if degrees:
            angle = np.rad2deg(angle)
        return angle

    raise ValueError('rotation matrices must have shape (3, 3) or (N, 3, 3).')


def euler_geodesic_error(
    angles_0,
    angles_1,
    symmetry='C1',
    degrees=True,
    symmetry_side='right',
    sym_ops=None,
):
    """
    Compute the symmetry-aware geodesic error between two Euler-angle orientations.

    angles_0 and angles_1 may each be shape (3,) or (N, 3), with pairwise
    comparison for batched input.
    """

    angles_0 = np.asarray(angles_0, dtype=np.float64)
    angles_1 = np.asarray(angles_1, dtype=np.float64)

    scalar_input = (angles_0.shape == (3,) and angles_1.shape == (3,))

    if scalar_input:
        rotation_matrix_0 = euler_to_rotation_matrix(angles_0)
        rotation_matrix_1 = euler_to_rotation_matrix(angles_1)
    else:
        if angles_0.shape != angles_1.shape:
            raise ValueError('angles_0 and angles_1 must have the same shape.')
        if angles_0.ndim != 2 or angles_0.shape[1] != 3:
            raise ValueError('For batched input, angles_0 and angles_1 must have shape (N, 3).')

        rotation_matrix_0 = euler_to_rotation_matrix(angles_0)
        rotation_matrix_1 = euler_to_rotation_matrix(angles_1)

    if sym_ops is None:
        sym_ops = point_group_symmetry_matrices(symmetry)
    else:
        sym_ops = np.asarray(sym_ops, dtype=np.float64)

    if sym_ops.ndim != 3 or sym_ops.shape[1:] != (3, 3):
        raise ValueError('sym_ops must have shape (N, 3, 3).')

    if symmetry_side not in ('right', 'left'):
        raise ValueError("symmetry_side must be either 'right' or 'left'.")

    if scalar_input:
        errors = []
        for sym_op in sym_ops:
            if symmetry_side == 'right':
                equivalent_rotation_matrix_1 = rotation_matrix_1 @ sym_op
            else:
                equivalent_rotation_matrix_1 = sym_op @ rotation_matrix_1

            errors.append(
                rotation_geodesic_error(
                    rotation_matrix_0,
                    equivalent_rotation_matrix_1,
                    degrees=degrees,
                )
            )

        return float(np.min(errors))

    # batched path: shape (N, M, 3, 3)
    if symmetry_side == 'right':
        equivalent_rotation_matrix_1 = np.matmul(
            rotation_matrix_1[:, None, :, :],
            sym_ops[None, :, :, :],
        )
    else:
        equivalent_rotation_matrix_1 = np.matmul(
            sym_ops[None, :, :, :],
            rotation_matrix_1[:, None, :, :],
        )

    relative_rotation = np.matmul(
        np.swapaxes(rotation_matrix_0[:, None, :, :], -1, -2),
        equivalent_rotation_matrix_1,
    )

    cos_angle = (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    errors = np.arccos(cos_angle)
    if degrees:
        errors = np.rad2deg(errors)

    return np.min(errors, axis=1)


def filter_library_by_geodesic_distance(
    library,
    pose,
    max_distance,
    symmetry='C1',
    symmetry_side='right',
    sym_ops=None,
    degrees=True,
):
    """
    Return the subset of a pose library within a geodesic distance of a given pose.

    Parameters
    ----------
    library : array-like, shape (N, 3)
        Pose library of Euler angles [phi, theta, psi] in degrees.
    pose : array-like, shape (3,)
        Reference pose as Euler angles [phi, theta, psi] in degrees.
    max_distance : float
        Maximum symmetry-aware geodesic distance from `pose`. In degrees
        unless `degrees=False`, in which case radians.
    symmetry : str, optional
        Point-group symmetry string (e.g. 'C1', 'C4', 'D2', 'O'). Default 'C1'.
    symmetry_side : str, optional
        Which side to apply symmetry operators — 'right' or 'left'. Default 'right'.
    sym_ops : array-like, shape (M, 3, 3), optional
        Pre-computed symmetry matrices. If provided, `symmetry` is ignored.
    degrees : bool, optional
        If True (default), `max_distance` and distances are in degrees.

    Returns
    -------
    subset : np.ndarray, shape (K, 3)
        Rows of `library` whose symmetry-aware geodesic distance to `pose`
        is less than or equal to `max_distance`.
    mask : np.ndarray of bool, shape (N,)
        Boolean mask into `library` selecting the returned rows.
    """

    library = np.asarray(library, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)

    if library.ndim != 2 or library.shape[1] != 3:
        raise ValueError('library must have shape (N, 3).')
    if pose.shape != (3,):
        raise ValueError('pose must have shape (3,).')

    if sym_ops is None:
        sym_ops = point_group_symmetry_matrices(symmetry)
    else:
        sym_ops = np.asarray(sym_ops, dtype=np.float64)

    # Broadcast: compare every library pose against the single reference pose.
    # euler_geodesic_error batched path requires equal-length arrays, so we
    # tile the reference pose to match the library length.
    pose_tiled = np.tile(pose, (library.shape[0], 1))

    distances = euler_geodesic_error(
        library,
        pose_tiled,
        symmetry=symmetry,
        degrees=degrees,
        symmetry_side=symmetry_side,
        sym_ops=sym_ops,
    )

    mask = distances <= max_distance
    return library[mask], mask


def point_group_symmetry_matrices(symmetry):
    """
    Return proper rotational point-group symmetry matrices.

    Parameters
    ----------
    symmetry : str
        Point-group symmetry string. Supported values are:

            Cn, Dn, T, O, I

    Returns
    -------
    sym_ops : np.ndarray
        Array of shape (N, 3, 3), where N is the number of proper rotations
        in the point group.

        Cn has N operators.
        Dn has 2N operators.
        T has 12 operators.
        O has 24 operators.
        I has 60 operators.
    """

    symmetry = symmetry.upper()

    if len(symmetry) < 1:
        raise ValueError('symmetry must be a non-empty string.')

    symmetry_type = symmetry[0]
    symmetry_number = int(symmetry[1:]) if len(symmetry) > 1 else None

    if symmetry_type == 'C':
        if symmetry_number is None or symmetry_number < 1:
            raise ValueError('C symmetry must be written like C1, C2, C7, ...')

        return _cyclic_symmetry_matrices(symmetry_number)

    if symmetry_type == 'D':
        if symmetry_number is None or symmetry_number < 1:
            raise ValueError('D symmetry must be written like D1, D2, D7, ...')

        return _dihedral_symmetry_matrices(symmetry_number)

    if symmetry_type == 'T':
        return _tetrahedral_symmetry_matrices()

    if symmetry_type == 'O':
        return _octahedral_symmetry_matrices()

    if symmetry_type == 'I':
        return _icosahedral_symmetry_matrices()

    raise ValueError('symmetry must be one of Cn, Dn, T, O, or I.')


def _cyclic_symmetry_matrices(n):
    return np.stack([
        _rotation_z(2.0 * np.pi * k / n)
        for k in range(n)
    ], axis=0)


def _dihedral_symmetry_matrices(n):
    cyclic_ops = [
        _rotation_z(2.0 * np.pi * k / n)
        for k in range(n)
    ]

    twofold_ops = [
        _rotation_z(2.0 * np.pi * k / n) @ _rotation_x(np.pi)
        for k in range(n)
    ]

    return np.stack(cyclic_ops + twofold_ops, axis=0)


def _tetrahedral_symmetry_matrices():
    """
    Proper rotational symmetry group of a tetrahedron.

    This returns 12 rotations.
    """

    generators = [
        _axis_angle_matrix([1.0, 1.0, 1.0], 2.0 * np.pi / 3.0),
        _rotation_x(np.pi),
    ]

    sym_ops = _close_rotation_group(generators)

    if sym_ops.shape[0] != 12:
        raise RuntimeError(f'Expected 12 tetrahedral symmetries, got {sym_ops.shape[0]}.')

    return sym_ops


def _octahedral_symmetry_matrices():
    """
    Proper rotational symmetry group of a cube/octahedron.

    This returns 24 rotations.
    """

    generators = [
        _rotation_z(np.pi / 2.0),
        _axis_angle_matrix([1.0, 1.0, 1.0], 2.0 * np.pi / 3.0),
    ]

    sym_ops = _close_rotation_group(generators)

    if sym_ops.shape[0] != 24:
        raise RuntimeError(f'Expected 24 octahedral symmetries, got {sym_ops.shape[0]}.')

    return sym_ops


def _icosahedral_symmetry_matrices():
    """
    Proper rotational symmetry group of an icosahedron.

    This returns 60 rotations.

    The construction uses the vertices of a conventional icosahedron and
    finds all proper rotations that map the vertex set onto itself.
    """

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = []

    for sign_1 in (-1.0, 1.0):
        for sign_2 in (-1.0, 1.0):
            vertices.append([0.0, sign_1, sign_2 * golden_ratio])
            vertices.append([sign_1, sign_2 * golden_ratio, 0.0])
            vertices.append([sign_1 * golden_ratio, 0.0, sign_2])

    vertices = np.asarray(vertices, dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]

    reference_vertex = vertices[0]

    dots = vertices @ reference_vertex
    neighbor_dot = np.max(dots[dots < 1.0 - 1e-8])

    reference_neighbor_index = np.where(np.isclose(dots, neighbor_dot, atol=1e-8))[0][0]
    reference_neighbor = vertices[reference_neighbor_index]

    reference_frame = _frame_from_two_vectors(reference_vertex, reference_neighbor)

    sym_ops = []

    for vertex_index in range(vertices.shape[0]):
        target_vertex = vertices[vertex_index]

        target_dots = vertices @ target_vertex
        target_neighbor_indices = np.where(
            np.isclose(target_dots, neighbor_dot, atol=1e-8)
        )[0]

        for target_neighbor_index in target_neighbor_indices:
            target_neighbor = vertices[target_neighbor_index]
            target_frame = _frame_from_two_vectors(target_vertex, target_neighbor)

            rotation_matrix = target_frame @ reference_frame.T

            if np.linalg.det(rotation_matrix) < 0:
                continue

            mapped_vertices = vertices @ rotation_matrix.T

            distance_matrix = np.linalg.norm(
                mapped_vertices[:, None, :] - vertices[None, :, :],
                axis=2,
            )

            max_nearest_distance = np.max(np.min(distance_matrix, axis=1))

            if max_nearest_distance > 1e-6:
                continue

            if _matrix_is_new(rotation_matrix, sym_ops):
                sym_ops.append(rotation_matrix)

    sym_ops = np.stack(sym_ops, axis=0)

    if sym_ops.shape[0] != 60:
        raise RuntimeError(f'Expected 60 icosahedral symmetries, got {sym_ops.shape[0]}.')

    return sym_ops


def _frame_from_two_vectors(vector_0, vector_1):
    vector_0 = np.asarray(vector_0, dtype=np.float64)
    vector_1 = np.asarray(vector_1, dtype=np.float64)

    e0 = vector_0 / np.linalg.norm(vector_0)

    e1 = vector_1 - np.dot(vector_1, e0) * e0
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(e0, e1)

    return np.stack([e0, e1, e2], axis=1)


def _close_rotation_group(generators, atol=1e-8):
    """
    Generate a finite rotation group from generator matrices.
    """

    matrices = [np.eye(3, dtype=np.float64)]

    changed = True

    while changed:
        changed = False

        current_matrices = list(matrices)

        for matrix in current_matrices:
            for generator in generators:
                candidates = [
                    matrix @ generator,
                    generator @ matrix,
                ]

                for candidate in candidates:
                    if _matrix_is_new(candidate, matrices, atol=atol):
                        matrices.append(candidate)
                        changed = True

    return np.stack(matrices, axis=0)


def _matrix_is_new(matrix, matrices, atol=1e-8):
    return not any(np.allclose(matrix, existing_matrix, atol=atol) for existing_matrix in matrices)


def _axis_angle_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    one_minus_cos = 1.0 - cos_angle

    return np.array([
        [
            cos_angle + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_angle,
            x * z * one_minus_cos + y * sin_angle,
        ],
        [
            y * x * one_minus_cos + z * sin_angle,
            cos_angle + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_angle,
        ],
        [
            z * x * one_minus_cos - y * sin_angle,
            z * y * one_minus_cos + x * sin_angle,
            cos_angle + z * z * one_minus_cos,
        ],
    ], dtype=np.float64)


def _rotation_x(angle_rad):
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_angle, -sin_angle],
        [0.0, sin_angle, cos_angle],
    ], dtype=np.float64)


def _rotation_z(angle_rad):
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    return np.array([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)