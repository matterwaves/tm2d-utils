import numpy as np

import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

def load_coords_from_npz(file_path, remove_h: bool = True):
    atom_data = np.load(file_path)
    atom_coords: np.ndarray = atom_data["coords"].astype(np.float32)

    if remove_h:
        atom_proton_counts: np.ndarray = atom_data["proton_counts"].astype(np.float32)
        atom_coords = atom_coords[atom_proton_counts != 1]

    atom_coords -= np.sum(atom_coords, axis=0) / atom_coords.shape[0]

    return atom_coords

def load_density_from_mrc(file_path):
    import mrcfile

    with mrcfile.open(file_path) as mrc:
        print("Pixel size:", mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z)

        return mrc.data.astype(np.complex64)
