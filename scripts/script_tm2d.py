"""Reusable TM2D workflow helpers for command-line scripts."""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import tm2d
import tm2d_utils as tu
import script_utils as su


TM_COLUMNS = {
    "angle_rot": "rlnTM2DAngleRot",
    "angle_tilt": "rlnTM2DAngleTilt",
    "angle_psi": "rlnTM2DAnglePsi",
    "defocus": "rlnTM2DDefocus",
    "defocus_u": "rlnTM2DDefocusU",
    "defocus_v": "rlnTM2DDefocusV",
    "defocus_angle": "rlnTM2DDefocusAngle",
    "astigmatism": "rlnTM2DAstigmatism",
    "astigmatism_angle": "rlnTM2DAstigmatismAngle",
    "phase_shift": "rlnTM2DPhaseShift",
    "z_score": "rlnTM2DZScore",
    "mip": "rlnTM2DMIP",
    "peak_x": "rlnTM2DPeakX",
    "peak_y": "rlnTM2DPeakY",
    "peak_offset_x": "rlnTM2DPeakOffsetX",
    "peak_offset_y": "rlnTM2DPeakOffsetY",
    "pixel_size": "rlnTM2DPixelSize",
    "B_factor": "rlnTM2DBFactor",
    "param_index": "rlnTM2DParamIndex",
    "mic_index": "rlnTM2DMicrographIndex",
    "particle_rel_index": "rlnTM2DParticleRelIndex",
    "stack_index": "rlnTM2DStackIndex",
}


ITER_PIXEL_COLUMNS = {
    "angle_rot": "rlnTM2DIterAngleRot",
    "angle_tilt": "rlnTM2DIterAngleTilt",
    "angle_psi": "rlnTM2DIterAnglePsi",
    "defocus": "rlnTM2DIterDefocus",
    "defocus_u": "rlnTM2DIterDefocusU",
    "defocus_v": "rlnTM2DIterDefocusV",
    "defocus_angle": "rlnTM2DIterDefocusAngle",
    "astigmatism": "rlnTM2DIterAstigmatism",
    "astigmatism_angle": "rlnTM2DIterAstigmatismAngle",
    "phase_shift": "rlnTM2DIterPhaseShift",
    "z_score": "rlnTM2DIterZScore",
    "mip": "rlnTM2DIterMIP",
    "peak_x": "rlnTM2DIterPeakX",
    "peak_y": "rlnTM2DIterPeakY",
    "peak_offset_x": "rlnTM2DIterPeakOffsetX",
    "peak_offset_y": "rlnTM2DIterPeakOffsetY",
    "pixel_size": "rlnTM2DIterPixelSize",
    "B_factor": "rlnTM2DIterBFactor",
    "param_index": "rlnTM2DIterParamIndex",
    "mic_index": "rlnTM2DIterMicrographIndex",
    "particle_rel_index": "rlnTM2DIterParticleRelIndex",
    "stack_index": "rlnTM2DIterStackIndex",
}


def get_session_config(args):
    return tu.make_session_config(
        workspace_root=args.workspace_root,
        workspace_root_is_remote=args.workspace_root_is_remote,
        session_name=args.session_name,
        session_laser_state=args.session_laser_state,
        session_job_type=args.session_job_type,
        session_job_num=args.session_job_num,
        **su.session_workspace_kwargs(args),
    )


def make_pixel_grid(center, half_width, step):
    n_steps = int(round((2.0 * half_width) / step))
    values = center - half_width + step * np.arange(n_steps + 1)
    return np.round(values, 6)


def make_pose_library(protein_coords, search_res, symmetry, diameter_a):
    region = tu.OrientationRegion(symmetry=symmetry)
    diameter = diameter_a if diameter_a is not None else tu.get_protein_radius(protein_coords) * 2.0
    ang_step = tu.crowther_ang_step_from_resolution(search_res, diameter)
    pose_lib = tu.get_orientations_healpix(ang_step, ang_step, region)
    return pose_lib, diameter, ang_step


def get_particle_batches(num_particles, max_batch_size=24):
    if num_particles <= 0:
        return []
    batch_size = max(1, int(min(max_batch_size, num_particles)))
    return [
        list(range(start, min(start + batch_size, num_particles)))
        for start in range(0, num_particles, batch_size)
    ]


def build_tm2d_input_batch(
    mic_stack,
    particle_rel_inds,
    expected_shape,
    crop_shape,
    mic_ind,
    batch_number,
    white_noise_var,
    pixel_size,
    do_hpf,
    hpf_cuton_bounds,
):
    ims_in = []
    good_rel_inds = []

    for p_rel_ind in particle_rel_inds:
        curr_im, error_msg = preprocess_image_for_tm2d(
            mic_stack.im_orig[p_rel_ind],
            expected_shape=expected_shape,
            crop_shape=crop_shape,
            white_noise_var=white_noise_var,
            pixel_size=pixel_size,
            do_hpf=do_hpf,
            hpf_cuton_bounds=hpf_cuton_bounds,
        )
        if error_msg is not None:
            print(f"Bad image: mic {mic_ind+1}, batch {batch_number}, particle {p_rel_ind}: {error_msg}")
            continue

        ims_in.append(curr_im)
        good_rel_inds.append(p_rel_ind)

    if not ims_in:
        return None, []
    return np.asarray(ims_in), good_rel_inds


def preprocess_image_for_tm2d(
    curr_im,
    expected_shape,
    crop_shape,
    white_noise_var,
    pixel_size,
    do_hpf,
    hpf_cuton_bounds,
):
    if curr_im is None:
        return None, "curr_im is None"

    curr_im = np.asarray(curr_im)
    if curr_im.shape != expected_shape:
        return None, f"raw shape {curr_im.shape}, expected {expected_shape}"

    curr_im = curr_im.copy()
    curr_im = curr_im[
        curr_im.shape[0] // 2 - crop_shape[0] // 2 : curr_im.shape[0] // 2 + crop_shape[0] // 2,
        curr_im.shape[1] // 2 - crop_shape[1] // 2 : curr_im.shape[1] // 2 + crop_shape[1] // 2,
    ]

    if white_noise_var > 0:
        curr_im += np.random.normal(0, np.sqrt(white_noise_var), curr_im.shape)

    curr_im -= np.mean(curr_im)
    if do_hpf:
        whitening_filter = tu.whiten_image(curr_im, pixel_size=pixel_size, double_whiten=False, return_filter_only=True)
        high_pass_filter = tu.get_hpf(curr_im.shape, pixel_size, cuton_start=hpf_cuton_bounds[0], cuton_end=hpf_cuton_bounds[1])
        combined_filter = whitening_filter * high_pass_filter
        curr_im = tu.apply_fourier_filt2d(curr_im, combined_filter)
        curr_im /= np.std(curr_im)
        curr_im = tu.apply_fourier_filt2d(curr_im, combined_filter)
    else:
        curr_im, whitening_filter = tu.whiten_image(curr_im, pixel_size=pixel_size, double_whiten=False, return_filter=True)
        curr_im /= np.std(curr_im)
        curr_im = tu.apply_fourier_filt2d(curr_im, whitening_filter)

    return np.asarray(curr_im), None


def run_tm2d_pixels(
    micrographs,
    param_set,
    template,
    ctf_params,
    template_batch_size,
    enable_progress_bar,
):
    results = tm2d.ResultsPixel(micrographs.shape)
    comparator = tm2d.ComparatorCrossCorrelation(
        shape=micrographs.shape,
        template_shape=template.get_shape(),
    )
    plan = tm2d.Plan(
        template=template,
        comparator=comparator,
        results=results,
        ctf_params=ctf_params,
        template_batch_size=template_batch_size,
        enable_rotation_weights=param_set.rotations_weights is not None,
    )
    plan.set_data(micrographs)
    plan.run(param_set, enable_progress_bar=enable_progress_bar)
    return results


def load_protein_coords(pdb_fpath=None):
    if pdb_fpath is None:
        raise ValueError(
            "No PDB path configured. Pass --pdb-fpath or set DEFAULT_PDB_FPATH "
            "in tm2d_utils/local_parameters.py."
        )
    pdb_fpath = str(Path(pdb_fpath).expanduser())
    protein_coords = tu.load_coords_from_pdb(pdb_fpath)
    print(len(protein_coords), "atoms loaded from", pdb_fpath)
    return pdb_fpath, protein_coords


def resolve_density_helper_fpath(helper_fpath):
    if helper_fpath is not None:
        helper_path = Path(helper_fpath).expanduser()
        if not helper_path.exists():
            raise FileNotFoundError(f"Density helper not found: {helper_path}")
        return helper_path

    for candidate in su.DEFAULT_DENSITY_HELPER_FPATHS:
        helper_path = Path(candidate).expanduser()
        if helper_path.exists():
            return helper_path

    searched = ", ".join(su.DEFAULT_DENSITY_HELPER_FPATHS) or "no default helper paths configured"
    raise FileNotFoundError(
        "Could not find density helper. Pass --density-helper-fpath or set "
        f"DEFAULT_DENSITY_HELPER_FPATHS in tm2d_utils/local_parameters.py. Searched: {searched}"
    )


def make_density_mrc(
    pdb_fpath,
    pixel_size,
    image_shape,
    output_dir,
    overwrite_density,
    density_helper_fpath,
    density_helper_python,
    density_device,
):
    volumes_dir = Path(output_dir) / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    pixel_tag = f"{float(pixel_size):.6f}".replace(".", "p")
    output_prefix = f"{Path(pdb_fpath).stem}_p{pixel_tag}_box{image_shape[0]}"
    mrc_fpath = volumes_dir / f"model_density_{output_prefix}.mrc"
    yaml_fpath = volumes_dir / f"model_config_{output_prefix}.yaml"

    if mrc_fpath.exists() and not overwrite_density:
        return mrc_fpath

    helper_fpath = resolve_density_helper_fpath(density_helper_fpath)
    helper_python = density_helper_python or sys.executable
    cmd = [
        helper_python,
        str(helper_fpath),
        "--pdb-filepath",
        str(pdb_fpath),
        "--output-dir",
        str(volumes_dir),
        "--output-prefix",
        output_prefix,
        "--mrc-filepath",
        str(mrc_fpath),
        "--yaml-filepath",
        str(yaml_fpath),
        "--volume-length",
        str(image_shape[0]),
        "--pixel-size",
        str(float(pixel_size)),
        "--device",
        str(density_device),
        "--overwrite" if overwrite_density else "--no-overwrite",
    ]
    print(f"Generating density template with {helper_fpath}")
    subprocess.run(cmd, check=True)
    return mrc_fpath


def make_template(
    model_type,
    image_shape,
    protein_coords,
    pdb_fpath,
    pixel_size,
    output_dir,
    overwrite_density,
    density_helper_fpath=None,
    density_helper_python=None,
    density_device="cpu",
):
    if model_type == "atomic":
        return tm2d.TemplateAtomic(
            shape=image_shape,
            atomic_coords=protein_coords,
        )

    if model_type == "density":
        output_fpath = make_density_mrc(
            pdb_fpath=pdb_fpath,
            pixel_size=pixel_size,
            image_shape=image_shape,
            output_dir=output_dir,
            overwrite_density=overwrite_density,
            density_helper_fpath=density_helper_fpath,
            density_helper_python=density_helper_python,
            density_device=density_device,
        )
        density_data = tu.load_density_from_mrc(output_fpath)
        print(f"Loaded density from {output_fpath} with shape {density_data.density.shape} and pixel size {float(density_data.pixel_size):.5f} A")
        return tm2d.TemplateDensity(
            density_array=density_data.density,
            density_pixel_size=density_data.pixel_size,
            padding_factor=su.DEFAULT_DENSITY_PADDING_FACTOR,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def select_micrographs(mics_and_inds, num_micrographs, particles_per_micrograph, micrograph_start_index=0):
    start = int(micrograph_start_index)
    if start < 0:
        raise ValueError("micrograph_start_index must be non-negative")

    stop = len(mics_and_inds) if num_micrographs is None else min(len(mics_and_inds), start + int(num_micrographs))
    selected = []
    for mic_fpath, mic_abs_inds in mics_and_inds[start:stop]:
        inds = mic_abs_inds if particles_per_micrograph is None else mic_abs_inds[: int(particles_per_micrograph)]
        selected.append((mic_fpath, list(inds)))
    return selected


def update_ctf_for_micrograph(
    ctf_params,
    mic_stack,
    defocus_offset,
    defocus_override=None,
    astigmatism_override=None,
    astigmatism_angle_override=None,
    phase_shift_override=None,
):
    defocus_u_mic = np.asarray(mic_stack.defocus_u_in, dtype=float)
    defocus_v_mic = np.asarray(mic_stack.defocus_v_in, dtype=float)
    defocus_ang_mic = np.asarray(mic_stack.defocus_ang_in, dtype=float)
    phase_shift_mic = np.asarray(mic_stack.phase_shift_in, dtype=float)

    defocus_mic, A_mag_mic, A_ang_mic = tu.convert_rln_C1A1_to_tm2d(defocus_u_mic, defocus_v_mic, defocus_ang_mic)

    ctf_params.defocus = float((np.nanmean(defocus_mic) if defocus_override is None else defocus_override) + defocus_offset)
    ctf_params.A_mag = float(np.nanmean(A_mag_mic) if astigmatism_override is None else astigmatism_override)
    ctf_params.A_ang = float(np.nanmean(A_ang_mic) if astigmatism_angle_override is None else astigmatism_angle_override)
    ctf_params.lpp = float(np.nanmean(phase_shift_mic) if phase_shift_override is None else phase_shift_override)


def get_best_rows_by_particle(raw_rows):
    best_by_particle = {}
    for row in raw_rows:
        key = (row["mic_ind"], row["particle_rel_ind"])
        if key not in best_by_particle or row["z_score"] > best_by_particle[key]["z_score"]:
            best_by_particle[key] = row
    return list(best_by_particle.values())


def summarize_scores(pixel_size, stage, raw_rows, top_n, score_stat):
    best_rows = get_best_rows_by_particle(raw_rows)
    z_scores = np.asarray([row["z_score"] for row in best_rows], dtype=float)
    z_scores = z_scores[np.isfinite(z_scores)]
    z_scores = np.sort(z_scores)[::-1]
    top_scores = z_scores[:top_n] if top_n > 0 else z_scores
    best_offsets = np.asarray([row["defocus_offset"] for row in best_rows], dtype=float)

    if top_scores.size == 0:
        objective = np.nan
    elif score_stat == "median":
        objective = float(np.nanmedian(top_scores))
    else:
        objective = float(np.nanmean(top_scores))

    return {
        "stage": stage,
        "pixel_size": float(pixel_size),
        "n_particles": int(z_scores.size),
        "n_raw_matches": int(len(raw_rows)),
        "top_n": int(top_n),
        "score_stat": score_stat,
        "objective_z_score": objective,
        "mean_z_score": float(np.nanmean(z_scores)) if z_scores.size else np.nan,
        "median_z_score": float(np.nanmedian(z_scores)) if z_scores.size else np.nan,
        "max_z_score": float(np.nanmax(z_scores)) if z_scores.size else np.nan,
        "std_z_score": float(np.nanstd(z_scores)) if z_scores.size else np.nan,
        "mean_top_z_score": float(np.nanmean(top_scores)) if top_scores.size else np.nan,
        "median_top_z_score": float(np.nanmedian(top_scores)) if top_scores.size else np.nan,
        "mean_best_defocus_offset": float(np.nanmean(best_offsets)) if best_offsets.size else np.nan,
        "median_best_defocus_offset": float(np.nanmedian(best_offsets)) if best_offsets.size else np.nan,
    }


def run_pixel_size_candidate(
    stage,
    pixel_size,
    selected_mics,
    stack,
    image_shape,
    pose_lib,
    template,
    ctf_params,
    defocus_offsets,
    defocus_override,
    astigmatism_override,
    astigmatism_angle_override,
    phase_shift_override,
    batch_size,
    enable_progress_bar,
    template_batch_size=su.DEFAULT_PIXEL_TEMPLATE_BATCH_SIZE,
):
    import time

    raw_rows = []
    for mic_ind, (mic_fpath, mic_abs_inds) in enumerate(selected_mics):
        t_mic_start = time.time()
        mic_stack = stack.get_substack(mic_abs_inds)
        print(f"... micrograph {mic_ind+1}/{len(selected_mics)}: {len(mic_abs_inds)} particles from {os.path.basename(mic_fpath)}")

        for defocus_offset in defocus_offsets:
            update_ctf_for_micrograph(
                ctf_params,
                mic_stack,
                defocus_offset,
                defocus_override=defocus_override,
                astigmatism_override=astigmatism_override,
                astigmatism_angle_override=astigmatism_angle_override,
                phase_shift_override=phase_shift_override,
            )
            print(f"...... defocus {ctf_params.defocus:.2f} A, offset {defocus_offset:+.2f} A, lpp {ctf_params.lpp:.2f} deg")

            for b_ind, p_rel_inds in enumerate(get_particle_batches(len(mic_abs_inds), max_batch_size=batch_size), start=1):
                mics_in, good_rel_inds = build_tm2d_input_batch(
                    mic_stack,
                    particle_rel_inds=p_rel_inds,
                    expected_shape=tuple(image_shape),
                    crop_shape=tuple(image_shape),
                    mic_ind=mic_ind,
                    batch_number=b_ind,
                    white_noise_var=0,
                    pixel_size=pixel_size,
                    do_hpf=False,
                    hpf_cuton_bounds=None,
                )
                if mics_in is None:
                    continue

                param_set = tm2d.make_param_set(
                    tm2d.make_ctf_set(ctf_params),
                    rotations=pose_lib,
                    pixel_sizes=np.array([pixel_size]),
                )
                results = run_tm2d_pixels(
                    micrographs=mics_in,
                    param_set=param_set,
                    template=template,
                    ctf_params=ctf_params,
                    template_batch_size=template_batch_size,
                    enable_progress_bar=enable_progress_bar,
                )

                mip = results.get_mip()
                best_index_array = results.get_best_index_array()
                z_score = tu.get_pixel_z_scores(results)

                for particle_batch_ind, p_rel_ind in enumerate(good_rel_inds):
                    best_pixel = np.unravel_index(np.argmax(z_score[particle_batch_ind]), z_score.shape[1:])
                    best_params_index = int(best_index_array[particle_batch_ind][best_pixel])
                    if best_params_index < 0 or best_params_index >= len(pose_lib):
                        print(f"warning: skipping particle with invalid TM2D parameter index {best_params_index} (valid range 0..{len(pose_lib) - 1})")
                        continue
                    best_pose = pose_lib[best_params_index]
                    raw_rows.append(
                        {
                            "stage": stage,
                            "pixel_size": float(pixel_size),
                            "defocus_offset": float(defocus_offset),
                            "mic_ind": int(mic_ind),
                            "mic_fpath": mic_fpath,
                            "particle_rel_ind": int(p_rel_ind),
                            "phi": float(best_pose[0]),
                            "theta": float(best_pose[1]),
                            "psi": float(best_pose[2]),
                            "mip": float(mip[particle_batch_ind][best_pixel]),
                            "z_score": float(z_score[particle_batch_ind][best_pixel]),
                        }
                    )

        print(f"... finished micrograph in {time.time() - t_mic_start:.1f} s")
    return raw_rows


def load_model_fine_defaults(model_fine_dir):
    if model_fine_dir is None:
        return {}, {}
    model_fine_dir = Path(model_fine_dir)
    metadata = su.load_json_if_exists(model_fine_dir / "model_fine_metadata.json")
    peaks = su.load_json_if_exists(model_fine_dir / "model_fine_peak_estimates.json")
    best = su.load_json_if_exists(model_fine_dir / "model_fine_best.json")
    chosen = peaks.get("spline") or peaks.get("sampled") or best
    return metadata, chosen


def corrected_particle_coordinates(df_particles, pixel_size):
    coords = df_particles[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(dtype=float)
    if "rlnOriginXAngst" in df_particles.columns and "rlnOriginYAngst" in df_particles.columns:
        coords -= df_particles[["rlnOriginXAngst", "rlnOriginYAngst"]].to_numpy(dtype=float) / pixel_size
    return coords


def map_stack_indices_to_star_rows(stack, df_particles, workspace_root, pixel_size, atol=1e-3):
    coords = corrected_particle_coordinates(df_particles, pixel_size)
    candidates = {}
    for row_pos, (idx, row) in enumerate(df_particles.iterrows()):
        mic_abs = os.path.join(workspace_root, row["rlnMicrographName"])
        candidates.setdefault(mic_abs, []).append((idx, coords[row_pos]))

    pointers = {mic: 0 for mic in candidates}
    mapping = {}
    for stack_ind, (mic_abs, coord) in enumerate(zip(stack.mic_fpath, stack.inds_in_mic)):
        mic_candidates = candidates.get(mic_abs)
        if not mic_candidates:
            continue
        start = pointers[mic_abs]
        coord = np.asarray(coord, dtype=float)
        for cand_pos in range(start, len(mic_candidates)):
            star_index, star_coord = mic_candidates[cand_pos]
            if np.allclose(coord, star_coord, atol=atol, rtol=0):
                mapping[stack_ind] = star_index
                pointers[mic_abs] = cand_pos + 1
                break
    return mapping



def make_micrograph_ctf_grid(mic_stack, base_ctf_params, args, *, allow_dynamic=True):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)

    defocus_u = np.asarray(mic_stack.defocus_u_in, dtype=float)
    defocus_v = np.asarray(mic_stack.defocus_v_in, dtype=float)
    defocus_ang = np.asarray(mic_stack.defocus_ang_in, dtype=float)
    phase_shift = np.asarray(mic_stack.phase_shift_in, dtype=float)
    defocus_tm2d, A_mag, A_ang = tu.convert_rln_C1A1_to_tm2d(defocus_u, defocus_v, defocus_ang)

    base_defocus = float(np.nanmean(defocus_tm2d) if getattr(args, "ctf_defocus", None) is None else args.ctf_defocus)
    base_A_mag = float(np.nanmean(A_mag) if getattr(args, "ctf_astigmatism", None) is None else args.ctf_astigmatism)
    base_A_ang = float(np.nanmean(A_ang) if getattr(args, "ctf_astigmatism_angle", None) is None else args.ctf_astigmatism_angle)
    base_lpp = float(np.nanmean(phase_shift) if getattr(args, "ctf_phase_shift", None) is None else args.ctf_phase_shift)

    defocus_values = su.parse_float_values(getattr(args, "defocus_values_A", None))
    if defocus_values is None:
        offsets = np.asarray(getattr(args, "defocus_offsets_A", [0.0]), dtype=float)
        defocus_values = base_defocus + offsets
    phase_values = su.parse_float_values(getattr(args, "phase_shift_values_deg", None))
    if phase_values is None:
        offsets = np.asarray(getattr(args, "phase_shift_offsets_deg", [0.0]), dtype=float)
        phase_values = base_lpp + offsets

    defocus_values = np.asarray(defocus_values, dtype=float)
    phase_values = np.asarray(phase_values, dtype=float)
    ctf_params.B = float(args.B_factor)
    ctf_params.A_mag = base_A_mag
    ctf_params.A_ang = base_A_ang

    dynamic_values = {}
    if allow_dynamic and len(defocus_values) > 1:
        ctf_params.defocus = None
        dynamic_values["defocus"] = defocus_values
    else:
        ctf_params.defocus = float(defocus_values[0])
    if allow_dynamic and len(phase_values) > 1:
        ctf_params.lpp = None
        dynamic_values["lpp"] = phase_values
    else:
        ctf_params.lpp = float(phase_values[0])

    summary = {
        "base_defocus": base_defocus,
        "base_A_mag": base_A_mag,
        "base_A_ang": base_A_ang,
        "base_lpp": base_lpp,
        "defocus_values": [float(v) for v in defocus_values],
        "phase_shift_values": [float(v) for v in phase_values],
    }
    return ctf_params, dynamic_values, summary


def tm_result_row_from_pixel_best(
    *,
    stack_index,
    mic_ind,
    mic_fpath,
    particle_rel_ind,
    param_set,
    ctf_params,
    param_index,
    best_pixel,
    image_shape,
    z_score_value,
    mip_value,
):
    best_values = param_set.get_values_at_index(int(param_index))
    rotation = best_values["rotation"]
    defocus = float(best_values.get("defocus", ctf_params.defocus))
    phase = float(best_values.get("lpp", ctf_params.lpp))
    defocus_u, defocus_v, defocus_angle = tu.convert_tm2d_C1A1_to_rln(defocus, ctf_params.A_mag, ctf_params.A_ang)
    peak_y, peak_x = int(best_pixel[0]), int(best_pixel[1])
    return {
        "stack_index": int(stack_index),
        "mic_ind": int(mic_ind),
        "mic_fpath": mic_fpath,
        "particle_rel_ind": int(particle_rel_ind),
        "tmAngleRot": float(rotation[0]),
        "tmAngleTilt": float(rotation[1]),
        "tmAnglePsi": float(rotation[2]),
        "tmDefocus": defocus,
        "tmDefocusU": float(defocus_u),
        "tmDefocusV": float(defocus_v),
        "tmDefocusAngle": float(defocus_angle),
        "tmAstigmatism": float(ctf_params.A_mag),
        "tmAstigmatismAngle": float(ctf_params.A_ang),
        "tmPhaseShift": phase,
        "tmZScore": float(z_score_value),
        "tmMip": float(mip_value),
        "tmPeakX": peak_x,
        "tmPeakY": peak_y,
        "tmPeakOffsetX": float(peak_x - image_shape[1] / 2.0),
        "tmPeakOffsetY": float(peak_y - image_shape[0] / 2.0),
        "tmPixelSize": float(best_values.get("pixel_size", np.nan)),
        "tmBFactor": float(ctf_params.B),
        "tmParamIndex": int(param_index),
    }


def run_pixel_pose_search(
    *,
    selected_mics,
    stack,
    image_shape,
    pose_lib,
    template,
    base_ctf_params,
    args,
    ctf_builder=make_micrograph_ctf_grid,
    row_callback=None,
    partial_writer=None,
):
    rows = []
    ctf_summaries = []
    for mic_ind, (mic_fpath, mic_abs_inds) in enumerate(selected_mics):
        t0 = time.time()
        mic_stack = stack.get_substack(mic_abs_inds)
        ctf_params, dynamic_values, ctf_summary = ctf_builder(mic_stack, base_ctf_params, args)
        ctf_summaries.append({"mic_fpath": mic_fpath, **ctf_summary})
        ctf_set = tm2d.make_ctf_set(ctf_params, **dynamic_values)
        param_set = tm2d.make_param_set(ctf_set, rotations=pose_lib, pixel_sizes=np.array([float(args.pixel_size)]))
        print(
            f"micrograph {mic_ind + 1}/{len(selected_mics)}: {len(mic_abs_inds)} particles, "
            f"{len(pose_lib)} poses x {ctf_set.get_length()} CTF combinations"
        )

        for batch_number, p_rel_inds in enumerate(get_particle_batches(len(mic_abs_inds), max_batch_size=args.batch_size), start=1):
            mics_in, good_rel_inds = build_tm2d_input_batch(
                mic_stack,
                particle_rel_inds=p_rel_inds,
                expected_shape=tuple(image_shape),
                crop_shape=tuple(image_shape),
                mic_ind=mic_ind,
                batch_number=batch_number,
                white_noise_var=0,
                pixel_size=float(args.pixel_size),
                do_hpf=False,
                hpf_cuton_bounds=None,
            )
            if mics_in is None:
                continue

            results = run_tm2d_pixels(
                micrographs=mics_in,
                param_set=param_set,
                template=template,
                ctf_params=ctf_params,
                template_batch_size=int(args.template_batch_size),
                enable_progress_bar=args.progress,
            )
            mip = results.get_mip()
            best_index_array = results.get_best_index_array()
            z_score = tu.get_pixel_z_scores(results)

            for particle_batch_ind, p_rel_ind in enumerate(good_rel_inds):
                best_pixel = np.unravel_index(int(np.nanargmax(z_score[particle_batch_ind])), z_score.shape[1:])
                param_index = int(best_index_array[particle_batch_ind][best_pixel])
                if param_index < 0 or param_index >= param_set.get_total_count():
                    print(f"warning: skipping particle with invalid TM2D parameter index {param_index} (valid range 0..{param_set.get_total_count() - 1})")
                    continue

                row = tm_result_row_from_pixel_best(
                    stack_index=int(mic_abs_inds[p_rel_ind]),
                    mic_ind=mic_ind,
                    mic_fpath=mic_fpath,
                    particle_rel_ind=p_rel_ind,
                    param_set=param_set,
                    ctf_params=ctf_params,
                    param_index=param_index,
                    best_pixel=best_pixel,
                    image_shape=image_shape,
                    z_score_value=z_score[particle_batch_ind][best_pixel],
                    mip_value=mip[particle_batch_ind][best_pixel],
                )
                row["tmPixelSize"] = float(args.pixel_size)
                row["tmBFactor"] = float(args.B_factor)
                if row_callback is not None:
                    row = row_callback(
                        row=row,
                        mic_ind=mic_ind,
                        mic_fpath=mic_fpath,
                        mic_abs_inds=mic_abs_inds,
                        mic_stack=mic_stack,
                        p_rel_ind=p_rel_ind,
                        particle_batch_ind=particle_batch_ind,
                        best_pixel=best_pixel,
                        param_index=param_index,
                        param_set=param_set,
                        mip=mip,
                        z_score=z_score,
                    )
                if row is not None:
                    rows.append(row)
        print(f"  finished in {time.time() - t0:.1f} s")
        if partial_writer is not None:
            partial_writer(rows, ctf_summaries, mic_ind + 1, len(selected_mics))
    return rows, ctf_summaries


def attach_tm_rows_to_star(df_particles, rows, column_map, *, row_to_star_index=None):
    df_out = df_particles.copy()
    for column in column_map.values():
        df_out[column] = np.nan

    row_key_to_column = {
        "tmAngleRot": "angle_rot",
        "tmAngleTilt": "angle_tilt",
        "tmAnglePsi": "angle_psi",
        "tmDefocus": "defocus",
        "tmDefocusU": "defocus_u",
        "tmDefocusV": "defocus_v",
        "tmDefocusAngle": "defocus_angle",
        "tmAstigmatism": "astigmatism",
        "tmAstigmatismAngle": "astigmatism_angle",
        "tmPhaseShift": "phase_shift",
        "tmZScore": "z_score",
        "tmMip": "mip",
        "tmPeakX": "peak_x",
        "tmPeakY": "peak_y",
        "tmPeakOffsetX": "peak_offset_x",
        "tmPeakOffsetY": "peak_offset_y",
        "tmPixelSize": "pixel_size",
        "tmBFactor": "B_factor",
        "tmParamIndex": "param_index",
        "mic_ind": "mic_index",
        "particle_rel_ind": "particle_rel_index",
        "stack_index": "stack_index",
    }

    for row in rows:
        star_index = row.get("star_index")
        if star_index is None and row_to_star_index is not None:
            star_index = row_to_star_index.get(row.get("stack_index"))
        if star_index is None:
            continue
        for row_key, column_key in row_key_to_column.items():
            column = column_map.get(column_key)
            if column is not None and row_key in row:
                df_out.at[star_index, column] = row[row_key]
    return df_out


def find_star_file(directory, preferred_name):
    directory = Path(directory)
    candidate = directory / preferred_name
    if candidate.exists():
        return candidate
    matches = sorted(directory.glob("*.star"))
    if not matches:
        raise FileNotFoundError(f"No STAR file found in {directory}")
    return matches[0]


def require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required STAR columns: {', '.join(missing)}")


def make_centered_range(center, half_width, step):
    center = float(center)
    half_width = float(half_width)
    step = float(step)
    if half_width <= 0 or step <= 0:
        return np.asarray([center], dtype=float)
    values = center - half_width + step * np.arange(int(np.floor(2 * half_width / step + 1e-9)) + 1)
    if not np.any(np.isclose(values, center, atol=1e-9, rtol=0)):
        values = np.concatenate([values, [center]])
    values = values[(values >= center - half_width - 1e-9) & (values <= center + half_width + 1e-9)]
    return np.asarray(sorted(np.unique(np.round(values, 6))), dtype=float)


def run_single_pose_ctf_param_search(image, image_shape, pose, template, ctf_params, dynamic_values, pixel_size, template_batch_size, output_radius=None, progress=False):
    ctf_set = tm2d.make_ctf_set(ctf_params, **dynamic_values)
    param_set = tm2d.make_param_set(ctf_set, rotations=np.asarray([pose], dtype=float), pixel_sizes=np.array([float(pixel_size)]))
    results = tm2d.ResultsParam(1, param_set.get_total_count(), output_radius=output_radius)
    comparator = tm2d.ComparatorCrossCorrelation(shape=(1, *image_shape), template_shape=template.get_shape())
    plan = tm2d.Plan(
        template=template,
        comparator=comparator,
        results=results,
        ctf_params=ctf_params,
        template_batch_size=int(template_batch_size),
        output_radius=output_radius,
        enable_rotation_weights=False,
    )
    plan.set_data(np.asarray([image], dtype=np.float32))
    plan.run(param_set, enable_progress_bar=progress)
    z_values = np.squeeze(results.get_zscore_list(param_set))
    mip_values = np.squeeze(results.get_mip_list(param_set))
    flat_best = int(np.nanargmax(z_values))
    param_index = int(np.ravel_multi_index(np.unravel_index(flat_best, np.asarray(z_values).shape), np.asarray(z_values).shape)) if np.asarray(z_values).ndim > 0 else 0
    best_values = param_set.get_values_at_index(param_index)
    defocus = float(best_values.get("defocus", ctf_params.defocus))
    phase = float(best_values.get("lpp", ctf_params.lpp))
    return z_values, mip_values, param_index, defocus, phase, ctf_set
