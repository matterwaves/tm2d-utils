import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

import tm2d
import tm2d_utils as tu
import vkdispatch as vd

import script_utils as su
import script_tm2d as st
from tm2d_utils import particle_stack as ps


INITIAL_COLUMNS = st.TM_COLUMNS
REFINE_COLUMNS = {
    "angle_rot": "rlnTM2DRefineAngleRot",
    "angle_tilt": "rlnTM2DRefineAngleTilt",
    "angle_psi": "rlnTM2DRefineAnglePsi",
    "defocus": "rlnTM2DRefineDefocus",
    "defocus_u": "rlnTM2DRefineDefocusU",
    "defocus_v": "rlnTM2DRefineDefocusV",
    "defocus_angle": "rlnTM2DRefineDefocusAngle",
    "astigmatism": "rlnTM2DRefineAstigmatism",
    "astigmatism_angle": "rlnTM2DRefineAstigmatismAngle",
    "phase_shift": "rlnTM2DRefinePhaseShift",
    "z_score": "rlnTM2DRefineZScore",
    "mip": "rlnTM2DRefineMIP",
    "peak_x": "rlnTM2DRefinePeakX",
    "peak_y": "rlnTM2DRefinePeakY",
    "peak_offset_x": "rlnTM2DRefinePeakOffsetX",
    "peak_offset_y": "rlnTM2DRefinePeakOffsetY",
    "pixel_size": "rlnTM2DRefinePixelSize",
    "B_factor": "rlnTM2DRefineBFactor",
    "param_index": "rlnTM2DRefineParamIndex",
    "pose_count": "rlnTM2DRefinePoseCount",
    "ctf_count": "rlnTM2DRefineCTFCount",
    "stack_index": "rlnTM2DRefineStackIndex",
}


def load_fixed_inputs(args):
    fixed_metadata = {}
    if args.fixed_model_dir is not None:
        fixed_metadata = su.load_json_if_exists(Path(args.fixed_model_dir) / "fixed_model_tm2d_metadata.json")

    fixed_star = Path(args.fixed_star) if args.fixed_star is not None else st.find_star_file(args.fixed_model_dir, "fixed_model_tm2d_results.star")
    star_data = starfile.read(fixed_star)
    if not isinstance(star_data, dict):
        raise ValueError("Expected a RELION 5 STAR with optics and particles blocks.")
    return fixed_star, star_data, fixed_metadata


def resolve_arg(value, *metadata_default_pairs):
    if value is not None:
        return value
    for metadata, key, default in metadata_default_pairs:
        if metadata is not None and key in metadata and metadata[key] is not None:
            return metadata[key]
        if default is not None:
            return default
    return None


def select_refinement_rows(df_particles, args):
    st.require_columns(
        df_particles,
        [
            INITIAL_COLUMNS["angle_rot"],
            INITIAL_COLUMNS["angle_tilt"],
            INITIAL_COLUMNS["angle_psi"],
            INITIAL_COLUMNS["defocus"],
            INITIAL_COLUMNS["phase_shift"],
            INITIAL_COLUMNS["z_score"],
            INITIAL_COLUMNS["stack_index"],
        ],
    )
    df = df_particles[df_particles[INITIAL_COLUMNS["z_score"]].notna()].copy()
    if args.max_particles is not None:
        df = df.iloc[: int(args.max_particles)].copy()
    return df


def build_pose_library_for_particle(initial_pose, global_pose_lib, args, sym_ops):
    initial_pose = np.asarray(initial_pose, dtype=float).reshape(1, 3)
    if args.pose_window_deg <= 0:
        return initial_pose

    subset, _ = tu.filter_library_by_geodesic_distance(
        global_pose_lib,
        initial_pose[0],
        max_distance=float(args.pose_window_deg),
        symmetry=args.symmetry,
        sym_ops=sym_ops,
        degrees=True,
    )
    if subset.size == 0:
        subset = initial_pose
    elif not np.any(np.all(np.isclose(subset, initial_pose[0], atol=1e-6, rtol=0), axis=1)):
        subset = np.vstack([initial_pose, subset])
    if args.max_pose_count is not None and subset.shape[0] > int(args.max_pose_count):
        distances = tu.euler_geodesic_error(
            subset,
            np.tile(initial_pose[0], (subset.shape[0], 1)),
            symmetry=args.symmetry,
            sym_ops=sym_ops,
            degrees=True,
        )
        keep = np.argsort(distances)[: int(args.max_pose_count)]
        subset = subset[keep]
    return np.asarray(subset, dtype=float)


def build_ctf_grid_for_particle(row, base_ctf_params, args):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)

    center_defocus = float(row[INITIAL_COLUMNS["defocus"]])
    center_phase = float(row[INITIAL_COLUMNS["phase_shift"]])
    center_astig = float(row[INITIAL_COLUMNS["astigmatism"]]) if INITIAL_COLUMNS["astigmatism"] in row and pd.notna(row[INITIAL_COLUMNS["astigmatism"]]) else 0.0
    center_astig_angle = float(row[INITIAL_COLUMNS["astigmatism_angle"]]) if INITIAL_COLUMNS["astigmatism_angle"] in row and pd.notna(row[INITIAL_COLUMNS["astigmatism_angle"]]) else 0.0

    defocus_values = su.parse_float_values(args.defocus_values_A)
    if defocus_values is None:
        defocus_values = st.make_centered_range(center_defocus, args.defocus_half_width_A, args.defocus_step_A)

    phase_values = su.parse_float_values(args.phase_shift_values_deg)
    if phase_values is None:
        phase_values = st.make_centered_range(center_phase, args.phase_shift_half_width_deg, args.phase_shift_step_deg)

    astig_values = su.parse_float_values(args.astigmatism_values_A)
    if astig_values is None:
        astig_values = st.make_centered_range(center_astig, args.astigmatism_half_width_A, args.astigmatism_step_A)

    astig_angle_values = su.parse_float_values(args.astigmatism_angle_values_deg)
    if astig_angle_values is None:
        astig_angle_values = st.make_centered_range(center_astig_angle, args.astigmatism_angle_half_width_deg, args.astigmatism_angle_step_deg)

    ctf_params.defocus = None if len(defocus_values) > 1 else float(defocus_values[0])
    ctf_params.lpp = None if len(phase_values) > 1 else float(phase_values[0])
    ctf_params.A_mag = None if len(astig_values) > 1 else float(astig_values[0])
    ctf_params.A_ang = None if len(astig_angle_values) > 1 else float(astig_angle_values[0])

    dynamic_values = {}
    if ctf_params.defocus is None:
        dynamic_values["defocus"] = np.asarray(defocus_values, dtype=float)
    if ctf_params.lpp is None:
        dynamic_values["lpp"] = np.asarray(phase_values, dtype=float)
    if ctf_params.A_mag is None:
        dynamic_values["A_mag"] = np.asarray(astig_values, dtype=float)
    if ctf_params.A_ang is None:
        dynamic_values["A_ang"] = np.asarray(astig_angle_values, dtype=float)

    grid_summary = {
        "center_defocus": center_defocus,
        "center_phase_shift": center_phase,
        "center_astigmatism": center_astig,
        "center_astigmatism_angle": center_astig_angle,
        "defocus_values": [float(v) for v in defocus_values],
        "phase_shift_values": [float(v) for v in phase_values],
        "astigmatism_values": [float(v) for v in astig_values],
        "astigmatism_angle_values": [float(v) for v in astig_angle_values],
    }
    return ctf_params, dynamic_values, grid_summary


def run_single_particle_refinement(image, image_shape, pose_lib, template, ctf_params, dynamic_values, args):
    ctf_set = tm2d.make_ctf_set(ctf_params, **dynamic_values)
    param_set = tm2d.make_param_set(
        ctf_set,
        rotations=pose_lib,
        pixel_sizes=np.array([float(args.pixel_size)]),
    )
    mics_in = np.asarray([image], dtype=np.float32)
    results = st.run_tm2d_pixels(
        micrographs=mics_in,
        param_set=param_set,
        template=template,
        ctf_params=ctf_params,
        template_batch_size=int(args.template_batch_size),
        enable_progress_bar=False,
    )
    mip = results.get_mip()[0]
    best_index_array = results.get_best_index_array()[0]
    z_score = tu.get_pixel_z_scores(results)[0]
    best_pixel = np.unravel_index(int(np.nanargmax(z_score)), z_score.shape)
    param_index = int(best_index_array[best_pixel])
    best_values = param_set.get_values_at_index(param_index)
    rotation = best_values["rotation"]
    defocus = float(best_values.get("defocus", ctf_params.defocus))
    phase = float(best_values.get("lpp", ctf_params.lpp))
    astig = float(best_values.get("A_mag", ctf_params.A_mag))
    astig_angle = float(best_values.get("A_ang", ctf_params.A_ang))
    defocus_u, defocus_v, defocus_angle = tu.convert_tm2d_C1A1_to_rln(defocus, astig, astig_angle)
    peak_y, peak_x = int(best_pixel[0]), int(best_pixel[1])

    return {
        "tmRefineAngleRot": float(rotation[0]),
        "tmRefineAngleTilt": float(rotation[1]),
        "tmRefineAnglePsi": float(rotation[2]),
        "tmRefineDefocus": defocus,
        "tmRefineDefocusU": float(defocus_u),
        "tmRefineDefocusV": float(defocus_v),
        "tmRefineDefocusAngle": float(defocus_angle),
        "tmRefineAstigmatism": astig,
        "tmRefineAstigmatismAngle": astig_angle,
        "tmRefinePhaseShift": phase,
        "tmRefineZScore": float(z_score[best_pixel]),
        "tmRefineMip": float(mip[best_pixel]),
        "tmRefinePeakX": peak_x,
        "tmRefinePeakY": peak_y,
        "tmRefinePeakOffsetX": float(peak_x - image_shape[1] / 2.0),
        "tmRefinePeakOffsetY": float(peak_y - image_shape[0] / 2.0),
        "tmRefinePixelSize": float(args.pixel_size),
        "tmRefineBFactor": float(args.B_factor),
        "tmRefineParamIndex": param_index,
        "tmRefinePoseCount": int(pose_lib.shape[0]),
        "tmRefineCTFCount": int(ctf_set.get_length()),
    }


def attach_refinement_to_star(df_particles, rows):
    df_out = df_particles.copy()
    for column in REFINE_COLUMNS.values():
        df_out[column] = np.nan

    for row in rows:
        star_index = row["star_index"]
        values = {
            REFINE_COLUMNS["angle_rot"]: row["tmRefineAngleRot"],
            REFINE_COLUMNS["angle_tilt"]: row["tmRefineAngleTilt"],
            REFINE_COLUMNS["angle_psi"]: row["tmRefineAnglePsi"],
            REFINE_COLUMNS["defocus"]: row["tmRefineDefocus"],
            REFINE_COLUMNS["defocus_u"]: row["tmRefineDefocusU"],
            REFINE_COLUMNS["defocus_v"]: row["tmRefineDefocusV"],
            REFINE_COLUMNS["defocus_angle"]: row["tmRefineDefocusAngle"],
            REFINE_COLUMNS["astigmatism"]: row["tmRefineAstigmatism"],
            REFINE_COLUMNS["astigmatism_angle"]: row["tmRefineAstigmatismAngle"],
            REFINE_COLUMNS["phase_shift"]: row["tmRefinePhaseShift"],
            REFINE_COLUMNS["z_score"]: row["tmRefineZScore"],
            REFINE_COLUMNS["mip"]: row["tmRefineMip"],
            REFINE_COLUMNS["peak_x"]: row["tmRefinePeakX"],
            REFINE_COLUMNS["peak_y"]: row["tmRefinePeakY"],
            REFINE_COLUMNS["peak_offset_x"]: row["tmRefinePeakOffsetX"],
            REFINE_COLUMNS["peak_offset_y"]: row["tmRefinePeakOffsetY"],
            REFINE_COLUMNS["pixel_size"]: row["tmRefinePixelSize"],
            REFINE_COLUMNS["B_factor"]: row["tmRefineBFactor"],
            REFINE_COLUMNS["param_index"]: row["tmRefineParamIndex"],
            REFINE_COLUMNS["pose_count"]: row["tmRefinePoseCount"],
            REFINE_COLUMNS["ctf_count"]: row["tmRefineCTFCount"],
            REFINE_COLUMNS["stack_index"]: row["stack_index"],
        }
        for column, value in values.items():
            df_out.at[star_index, column] = value
    return df_out


write_csv = su.write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Refine CTF parameters around a fixed-model TM2D search result, optionally allowing a small pose window.")
    parser.add_argument("--fixed-model-dir", default=None, help="Directory from run_fixed_model_tm2d_search.py.")
    parser.add_argument("--fixed-star", default=None, help="STAR file from run_fixed_model_tm2d_search.py. Overrides --fixed-model-dir lookup.")
    parser.add_argument("--model-fine-dir", default=None, help="Optional model-fine directory for pixel size/B-factor defaults.")
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--B-factor", type=float, default=None)
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--pdb-fpath", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=su.parse_bool, default=None)
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-user", default=None)
    parser.add_argument("--remote-key-filename", default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--session-laser-state", type=su.parse_bool, default=None)
    parser.add_argument("--session-job-type", default=None)
    parser.add_argument("--session-job-num", type=int, default=None)
    parser.add_argument("--particle-limit", type=int, default=None)
    parser.add_argument("--max-particles", type=int, default=None, help="Refine only the first N particles with fixed-model results.")
    parser.add_argument("--pose-window-deg", type=float, default=0.0, help="Symmetry-aware pose window around initial TM2D pose; 0 fixes pose.")
    parser.add_argument("--pose-search-res", type=float, default=None, help="Resolution used to build the pose library when pose window is nonzero.")
    parser.add_argument("--max-pose-count", type=int, default=None, help="Keep only the nearest N poses in each pose window.")
    parser.add_argument("--symmetry", default=su.DEFAULT_SYMMETRY)
    parser.add_argument("--diameter-A", type=float, default=su.DEFAULT_DIAMETER_A, help="Particle diameter in Angstroms for pose-window sampling; defaults to coordinate-derived diameter.")
    parser.add_argument("--defocus-half-width-A", type=float, default=500.0)
    parser.add_argument("--defocus-step-A", type=float, default=10.0)
    parser.add_argument("--defocus-values-A", default=None, help="Comma-separated absolute defocus values; overrides center/half-width/step.")
    parser.add_argument("--phase-shift-half-width-deg", type=float, default=15.0)
    parser.add_argument("--phase-shift-step-deg", type=float, default=1.0)
    parser.add_argument("--phase-shift-values-deg", default=None, help="Comma-separated absolute phase-shift values; overrides center/half-width/step.")
    parser.add_argument("--astigmatism-half-width-A", type=float, default=0.0)
    parser.add_argument("--astigmatism-step-A", type=float, default=50.0)
    parser.add_argument("--astigmatism-values-A", default=None)
    parser.add_argument("--astigmatism-angle-half-width-deg", type=float, default=0.0)
    parser.add_argument("--astigmatism-angle-step-deg", type=float, default=5.0)
    parser.add_argument("--astigmatism-angle-values-deg", default=None)
    parser.add_argument("--template-batch-size", type=int, default=16)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-star", default=None)
    parser.add_argument("--density-helper-fpath", default=None)
    parser.add_argument("--density-helper-python", default=None)
    parser.add_argument("--density-device", default="cpu")
    parser.add_argument("--overwrite-density", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fixed_model_dir is None and args.fixed_star is None:
        raise ValueError("Pass --fixed-model-dir or --fixed-star.")

    fixed_star, star_data, fixed_metadata = load_fixed_inputs(args)
    model_metadata, chosen_peak = st.load_model_fine_defaults(args.model_fine_dir or fixed_metadata.get("model_fine_dir"))
    df_optics = star_data["optics"]
    df_particles = star_data["particles"]
    refine_df = select_refinement_rows(df_particles, args)
    if refine_df.empty:
        raise ValueError("No particles with fixed-model TM2D results found to refine.")

    args.model_type = resolve_arg(args.model_type, (fixed_metadata, "model_type", None), (model_metadata, "model_type", su.DEFAULT_MODEL_TYPE))
    args.workspace_root = resolve_arg(args.workspace_root, (fixed_metadata, "workspace_root", None), (model_metadata, "workspace_root", su.DEFAULT_WORKSPACE_ROOT))
    args.workspace_root_is_remote = resolve_arg(args.workspace_root_is_remote, (fixed_metadata, "workspace_root_is_remote", None), (model_metadata, "workspace_root_is_remote", su.DEFAULT_WORKSPACE_ROOT_IS_REMOTE))
    args.session_name = resolve_arg(args.session_name, (fixed_metadata, "session_name", None), (model_metadata, "session_name", su.DEFAULT_SESSION_NAME))
    args.session_laser_state = resolve_arg(args.session_laser_state, (fixed_metadata, "session_laser_state", None), (model_metadata, "session_laser_state", su.DEFAULT_SESSION_LASER_STATE))
    args.session_job_type = resolve_arg(args.session_job_type, (fixed_metadata, "session_job_type", None), (model_metadata, "session_job_type", su.DEFAULT_SESSION_JOB_TYPE))
    args.session_job_num = int(resolve_arg(args.session_job_num, (fixed_metadata, "session_job_num", None), (model_metadata, "session_job_num", su.DEFAULT_SESSION_JOB_NUM)))
    args.particle_limit = resolve_arg(args.particle_limit, (fixed_metadata, "particle_limit", None), (model_metadata, "particle_limit", None))
    args.devices = resolve_arg(args.devices, (fixed_metadata, "devices", None), (model_metadata, "devices", su.DEFAULT_DEVICES))
    args.pdb_fpath = resolve_arg(args.pdb_fpath, (fixed_metadata, "pdb_fpath", None), (model_metadata, "pdb_fpath", su.DEFAULT_PDB_FPATH))

    if args.pixel_size is None:
        args.pixel_size = float(fixed_metadata.get("pixel_size", chosen_peak.get("pixel_size")))
    if args.B_factor is None:
        args.B_factor = float(fixed_metadata.get("B_factor", chosen_peak.get("B_factor")))

    device_ids = su.parse_device_ids(args.devices)
    vd.make_context(device_ids=device_ids)

    if args.output_dir is None:
        suffix = "fixed_pose" if args.pose_window_deg <= 0 else f"posewin{args.pose_window_deg:g}deg"
        args.output_dir = f"ctf_refinement_tm2d_{suffix}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "ctf_refinement_tm2d_results.star")

    config = st.get_session_config(args)
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=args.session_job_type, n_particles=args.particle_limit)
    image_shape = tuple(stack.im_orig[0].shape)

    pdb_fpath, protein_coords = st.load_protein_coords(args.pdb_fpath)
    template = st.make_template(
        args.model_type,
        image_shape,
        protein_coords,
        pdb_fpath,
        pixel_size=float(args.pixel_size),
        output_dir=args.output_dir,
        overwrite_density=args.overwrite_density,
        density_helper_fpath=args.density_helper_fpath,
        density_helper_python=args.density_helper_python,
        density_device=args.density_device,
    )

    global_pose_lib = None
    if args.pose_window_deg > 0:
        pose_res = args.pose_search_res if args.pose_search_res is not None else float(fixed_metadata.get("search_res", model_metadata.get("search_res", 3.0)))
        global_pose_lib, diameter, ang_step = st.make_pose_library(protein_coords, pose_res, args.symmetry, args.diameter_A)
        print(f"pose library size: {len(global_pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")
    sym_ops = tu.point_group_symmetry_matrices(args.symmetry)

    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = float(stack.Cs_nom_mm) * 1e7
    base_ctf_params.B = float(args.B_factor)

    rows = []
    start_time = time.time()
    total = len(refine_df)
    last_mic = None

    def write_outputs(completed):
        write_csv(output_dir / "ctf_refinement_tm2d_results.csv", rows)
        df_out = attach_refinement_to_star(df_particles, rows)
        starfile.write({"optics": df_optics, "particles": df_out}, args.output_star, overwrite=True)
        metadata = {
            "fixed_star": str(fixed_star),
            "fixed_model_dir": args.fixed_model_dir,
            "model_fine_dir": args.model_fine_dir or fixed_metadata.get("model_fine_dir"),
            "output_star": args.output_star,
            "model_type": args.model_type,
            "pdb_fpath": args.pdb_fpath,
            "pixel_size": float(args.pixel_size),
            "B_factor": float(args.B_factor),
            "pose_window_deg": float(args.pose_window_deg),
            "pose_search_res": args.pose_search_res,
            "max_pose_count": args.max_pose_count,
            "symmetry": args.symmetry,
            "defocus_half_width_A": args.defocus_half_width_A,
            "defocus_step_A": args.defocus_step_A,
            "phase_shift_half_width_deg": args.phase_shift_half_width_deg,
            "phase_shift_step_deg": args.phase_shift_step_deg,
            "astigmatism_half_width_A": args.astigmatism_half_width_A,
            "astigmatism_step_A": args.astigmatism_step_A,
            "astigmatism_angle_half_width_deg": args.astigmatism_angle_half_width_deg,
            "astigmatism_angle_step_deg": args.astigmatism_angle_step_deg,
            "template_batch_size": int(args.template_batch_size),
            "batching_note": "Each particle is refined separately because pose and CTF centers can differ per particle. template_batch_size still batches CTF/pose templates within each particle's local grid.",
            "devices": device_ids,
            "completed_particles": int(completed),
            "total_particles": int(total),
            "elapsed_s": time.time() - start_time,
        }
        with open(output_dir / "ctf_refinement_tm2d_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"updated {args.output_star}: {completed}/{total} refined particles")

    for completed, (star_index, row) in enumerate(refine_df.iterrows(), start=1):
        mic_name = row.get("rlnMicrographName")
        if mic_name != last_mic and last_mic is not None:
            write_outputs(completed - 1)
        last_mic = mic_name

        stack_index = int(row[INITIAL_COLUMNS["stack_index"]])
        if stack_index >= len(stack.im_orig):
            print(f"skipping STAR row {star_index}: stack index {stack_index} not loaded")
            continue
        initial_pose = np.asarray(
            [
                row[INITIAL_COLUMNS["angle_rot"]],
                row[INITIAL_COLUMNS["angle_tilt"]],
                row[INITIAL_COLUMNS["angle_psi"]],
            ],
            dtype=float,
        )
        pose_lib = build_pose_library_for_particle(initial_pose, global_pose_lib, args, sym_ops)
        ctf_params, dynamic_values, ctf_summary = build_ctf_grid_for_particle(row, base_ctf_params, args)
        image, error_msg = st.preprocess_image_for_tm2d(
            stack.im_orig[stack_index],
            expected_shape=image_shape,
            crop_shape=image_shape,
            white_noise_var=0,
            pixel_size=float(args.pixel_size),
            do_hpf=False,
            hpf_cuton_bounds=None,
        )
        if error_msg is not None:
            print(f"skipping STAR row {star_index}: {error_msg}")
            continue

        result = run_single_particle_refinement(image, image_shape, pose_lib, template, ctf_params, dynamic_values, args)
        result.update(
            {
                "star_index": star_index,
                "stack_index": stack_index,
                "rlnMicrographName": row.get("rlnMicrographName"),
                "initialAngleRot": float(initial_pose[0]),
                "initialAngleTilt": float(initial_pose[1]),
                "initialAnglePsi": float(initial_pose[2]),
                "initialDefocus": float(row[INITIAL_COLUMNS["defocus"]]),
                "initialPhaseShift": float(row[INITIAL_COLUMNS["phase_shift"]]),
                **ctf_summary,
            }
        )
        rows.append(result)

        if args.progress:
            print(
                f"{completed}/{total}: row {star_index}, poses={result['tmRefinePoseCount']}, "
                f"ctf={result['tmRefineCTFCount']}, z={result['tmRefineZScore']:.3f}"
            )

    write_outputs(total)


if __name__ == "__main__":
    main()
