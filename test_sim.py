import vkdispatch as vd
import tm2d
import numpy as np

import sys

import tm2d.utilities as tu

from matplotlib import pyplot as plt

#vd.initialize(debug_mode=True)
vd.make_context(max_streams=True)

small_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=160, # Default is 0
        phi_max=200, # Default is 360
        theta_min=70, # Default is 0 
        theta_max=90, # Default is 180
        psi_min=300, # Default is 0
        psi_max=340 # Default is 360
    )

medium_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=100, # Default is 0
        phi_max=200, # Default is 360
        theta_min=70, # Default is 0 
        theta_max=100, # Default is 180
        psi_min=300, # Default is 0
        psi_max=340 # Default is 360
    )

big_region = tm2d.OrientationRegion(
        # symmetry="C1", # This is the default, so we can omit it
        phi_min=0, # Default is 0
        phi_max=360, # Default is 360
        theta_min=0, # Default is 0 
        theta_max=180, # Default is 180
        psi_min=0, # Default is 0
        psi_max=360 # Default is 360
    )

# A copy of the data folder can be found at:
# /BigData/Workspaces/shahar/data

output_dir = sys.argv[1]

template_atomic = tm2d.TemplateAtomic(
    (512, 512),
    tu.load_coords_from_npz("data/parsed_5lks_LSU.npz")
)

template_image = template_atomic.make_template(
    [188.84183,  78.82107, 326],
    1.056,
    tm2d.CTFParams.like_krios(
        defocus=12870
    )
).read_real(0)[0]

#np.save("template.npy", template_image)

data_array = np.array(
    [
        tu.whiten_image(template_image),
    ]
)

comparator = tm2d.ComparatorCrossCorrelation(
    data_array.shape, # shape of the micrographs
    template_atomic.get_shape() # shape of the template
)

results = tm2d.ResultsPixel(data_array.shape)
# results = tm2d.ResultsParam(data_array.shape[0], 5000000)

plan = tm2d.Plan(
    template_atomic,
    comparator,
    results,
    #rotation=[188.84183,  78.82107, 326],
    pixel_size=1.056, # pixel size in Angstroms
    ctf_params=tm2d.CTFParams.like_krios(
        defocus = None, # 12870
    ),
    template_batch_size=4
)

plan.set_data(data_array)

# Get rotation list with cube sampling
rotations = tm2d.get_orientations_cube( # Get (phi, theta, psi) angles with cube sampling
    angular_step_size=2, # out of plane rotation step size
    psi_step_size=1, # in plane rotation step size
    region=small_region # region of interest for the orientations
)

params = plan.make_param_set(
    rotations=rotations,
    # pixel_sizes = np.arange(1.046, 1.066, 0.01),

    defocus = np.arange(12700, 13000, 25),
    # Cs = np.arange(1.9e7, 2.2e7, 1e4),
    # B = np.arange(0, 500, 25),
)

plan.run(params, enable_progress_bar=True)

for i in range(results.count):
    np.save(f"{output_dir}/mip{i}.npy", results.get_mip()[i])
    np.save(f"{output_dir}/Z_score{i}.npy", results.get_z_score()[i])

    match_index = results.get_index_of_params_match()[i]
    match_location = results.get_location_of_best_match()[i]

    best_indicies = results.get_best_index_array()[i]

    print(f"Micrograph {i}:")
    print(f"\tMax cross-correlation: {results.get_mip()[i][match_location]}")

    for param_name, param_value in params.get_values_at_index(match_index).items():
        print(f"\tBest {param_name}: {param_value}")

    for param_name, param_values in params.get_values_at_index(best_indicies).items():
        np.save(f"{output_dir}/{param_name}_{i}.npy", param_values)