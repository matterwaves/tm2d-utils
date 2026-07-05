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

import optimize_model_fine as omf
import optimize_pixel_size as ops
from tm2d_utils import particle_stack as ps


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


def parse_float_values(spec):
    if spec is None or str(spec).strip() == "":
        return None
    return np.asarray([float(value) for value in str(spec).split(",") if value != ""], dtype=float)


def load_model_fine_defaults(model_fine_dir):
    if model_fine_dir is None:
        return {}, {}
    model_fine_dir = Path(model_fine_dir)
    metadata = omf.load_json_if_exists(model_fine_dir / "model_fine_metadata.json")
    peaks = omf.load_json_if_exists(model_fine_dir / "model_fine_peak_estimates.json")
    best = omf.load_json_if_exists(model_fine_dir / "model_fine_best.json")
    chosen = peaks.get("spline") or peaks.get("sampled") or best
    return metadata, chosen


def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    return default


def select_micrographs(mics_and_inds, num_micrographs, particles_per_micrograph, micrograph_start_index):
    start = int(micrograph_start_index)
    stop = len(mics_and_inds) if num_micrographs is None else min(len(mics_and_inds), start + int(num_micrographs))
    selected = []
    for mic_fpath, mic_abs_inds in mics_and_inds[start:stop]:
        inds = list(mic_abs_inds if particles_per_micrograph is None else mic_abs_inds[: int(particles_per_micrograph)])
        selected.append((mic_fpath, inds))
    return selected


def build_ctf_grid(mic_stack, base_ctf_params, args):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)

    defocus_u = np.asarray(mic_stack.defocus_u_in, dtype=float)
    defocus_v = np.asarray(mic_stack.defocus_v_in, dtype=float)
    defocus_ang = np.asarray(mic_stack.defocus_ang_in, dtype=float)
    phase_shift = np.asarray(mic_stack.phase_shift_in, dtype=float)
    defocus_tm2d, A_mag, A_ang = tu.convert_rln_C1A1_to_tm2d(defocus_u, defocus_v, defocus_ang)

    base_defocus = float(np.nanmean(defocus_tm2d) if args.ctf_defocus is None else args.ctf_defocus)
    base_A_mag = float(np.nanmean(A_mag) if args.ctf_astigmatism is None else args.ctf_astigmatism)
    base_A_ang = float(np.nanmean(A_ang) if args.ctf_astigmatism_angle is None else args.ctf_astigmatism_angle)
    base_lpp = float(np.nanmean(phase_shift) if args.ctf_phase_shift is None else args.ctf_phase_shift)

    defocus_values = parse_float_values(args.defocus_values_A)
    if defocus_values is None:
        offsets = np.asarray(args.defocus_offsets_A, dtype=float)
        defocus_values = base_defocus + offsets
    phase_values = parse_float_values(args.phase_shift_values_deg)
    if phase_values is None:
        offsets = np.asarray(args.phase_shift_offsets_deg, dtype=float)
        phase_values = base_lpp + offsets

    ctf_params.B = float(args.B_factor)
    ctf_params.defocus = None if len(defocus_values) > 1 else float(defocus_values[0])
    ctf_params.A_mag = base_A_mag
    ctf_params.A_ang = base_A_ang
    ctf_params.lpp = None if len(phase_values) > 1 else float(phase_values[0])

    dynamic_values = {}
    if ctf_params.defocus is None:
        dynamic_values["defocus"] = np.asarray(defocus_values, dtype=float)
    if ctf_params.lpp is None:
        dynamic_values["lpp"] = np.asarray(phase_values, dtype=float)

    return ctf_params, dynamic_values, {
        "base_defocus": base_defocus,
        "base_A_mag": base_A_mag,
        "base_A_ang": base_A_ang,
        "base_lpp": base_lpp,
        "defocus_values": [float(v) for v in defocus_values],
        "phase_shift_values": [float(v) for v in phase_values],
    }


def run_search(selected_mics, stack, image_shape, pose_lib, template, base_ctf_params, args, partial_writer=None):
    rows = []
    ctf_summaries = []
    for mic_ind, (mic_fpath, mic_abs_inds) in enumerate(selected_mics):
        t0 = time.time()
        mic_stack = stack.get_substack(mic_abs_inds)
        ctf_params, dynamic_values, ctf_summary = build_ctf_grid(mic_stack, base_ctf_params, args)
        ctf_summaries.append({"mic_fpath": mic_fpath, **ctf_summary})
        ctf_set = tm2d.make_ctf_set(ctf_params, **dynamic_values)
        param_set = tm2d.make_param_set(ctf_set, rotations=pose_lib, pixel_sizes=np.array([float(args.pixel_size)]))
        print(
            f"micrograph {mic_ind + 1}/{len(selected_mics)}: {len(mic_abs_inds)} particles, "
            f"{len(pose_lib)} poses x {ctf_set.get_length()} CTF combinations"
        )

        for batch_number, p_rel_inds in enumerate(ops.get_particle_batches(len(mic_abs_inds), max_batch_size=args.batch_size), start=1):
            mics_in, good_rel_inds = ops.build_tm2d_input_batch(
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

            results = ops.run_tm2d_pixels(
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
                best_values = param_set.get_values_at_index(param_index)
                rotation = best_values["rotation"]
                defocus = float(best_values.get("defocus", ctf_params.defocus))
                phase = float(best_values.get("lpp", ctf_params.lpp))
                defocus_u, defocus_v, defocus_angle = tu.convert_tm2d_C1A1_to_rln(defocus, ctf_params.A_mag, ctf_params.A_ang)
                peak_y, peak_x = int(best_pixel[0]), int(best_pixel[1])
                rows.append(
                    {
                        "stack_index": int(mic_abs_inds[p_rel_ind]),
                        "mic_ind": int(mic_ind),
                        "mic_fpath": mic_fpath,
                        "particle_rel_ind": int(p_rel_ind),
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
                        "tmZScore": float(z_score[particle_batch_ind][best_pixel]),
                        "tmMip": float(mip[particle_batch_ind][best_pixel]),
                        "tmPeakX": peak_x,
                        "tmPeakY": peak_y,
                        "tmPeakOffsetX": float(peak_x - image_shape[1] / 2.0),
                        "tmPeakOffsetY": float(peak_y - image_shape[0] / 2.0),
                        "tmPixelSize": float(args.pixel_size),
                        "tmBFactor": float(args.B_factor),
                        "tmParamIndex": param_index,
                    }
                )
        print(f"  finished in {time.time() - t0:.1f} s")
        if partial_writer is not None:
            partial_writer(rows, ctf_summaries, mic_ind + 1, len(selected_mics))
    return rows, ctf_summaries


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


def attach_results_to_star(df_particles, rows, stack_to_star):
    df_out = df_particles.copy()
    for column in TM_COLUMNS.values():
        df_out[column] = np.nan

    for row in rows:
        star_index = stack_to_star.get(row["stack_index"])
        if star_index is None:
            continue
        values = {
            TM_COLUMNS["angle_rot"]: row["tmAngleRot"],
            TM_COLUMNS["angle_tilt"]: row["tmAngleTilt"],
            TM_COLUMNS["angle_psi"]: row["tmAnglePsi"],
            TM_COLUMNS["defocus"]: row["tmDefocus"],
            TM_COLUMNS["defocus_u"]: row["tmDefocusU"],
            TM_COLUMNS["defocus_v"]: row["tmDefocusV"],
            TM_COLUMNS["defocus_angle"]: row["tmDefocusAngle"],
            TM_COLUMNS["astigmatism"]: row["tmAstigmatism"],
            TM_COLUMNS["astigmatism_angle"]: row["tmAstigmatismAngle"],
            TM_COLUMNS["phase_shift"]: row["tmPhaseShift"],
            TM_COLUMNS["z_score"]: row["tmZScore"],
            TM_COLUMNS["mip"]: row["tmMip"],
            TM_COLUMNS["peak_x"]: row["tmPeakX"],
            TM_COLUMNS["peak_y"]: row["tmPeakY"],
            TM_COLUMNS["peak_offset_x"]: row["tmPeakOffsetX"],
            TM_COLUMNS["peak_offset_y"]: row["tmPeakOffsetY"],
            TM_COLUMNS["pixel_size"]: row["tmPixelSize"],
            TM_COLUMNS["B_factor"]: row["tmBFactor"],
            TM_COLUMNS["param_index"]: row["tmParamIndex"],
            TM_COLUMNS["mic_index"]: row["mic_ind"],
            TM_COLUMNS["particle_rel_index"]: row["particle_rel_ind"],
            TM_COLUMNS["stack_index"]: row["stack_index"],
        }
        for column, value in values.items():
            df_out.at[star_index, column] = value
    return df_out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TM2D pose search with fixed optimized pixel size and B factor, writing a STAR with TM2D result columns.")
    parser.add_argument("--model-fine-dir", default=None, help="Directory from optimize_model_fine.py; used for metadata defaults and spline/sample peak values.")
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--B-factor", type=float, default=None)
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--pdb-fpath", default=None, help="Path to the PDB used for atomic templates.")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=ops.parse_bool, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--session-laser-state", type=ops.parse_bool, default=None)
    parser.add_argument("--session-job-type", default=None)
    parser.add_argument("--session-job-num", type=int, default=None)
    parser.add_argument("--particle-limit", type=int, default=None)
    parser.add_argument("--num-micrographs", type=int, default=None)
    parser.add_argument("--micrograph-start-index", type=int, default=None)
    parser.add_argument("--particles-per-micrograph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--template-batch-size", type=int, default=None)
    parser.add_argument("--defocus-offsets-A", type=ops.parse_float_list, default=None, help="Offsets from each micrograph mean TM2D defocus. Used unless --defocus-values-A is given.")
    parser.add_argument("--defocus-values-A", default=None, help="Comma-separated absolute TM2D defocus values in Angstroms.")
    parser.add_argument("--phase-shift-offsets-deg", type=ops.parse_float_list, default=None, help="Offsets from each micrograph mean phase shift. Used unless --phase-shift-values-deg is given.")
    parser.add_argument("--phase-shift-values-deg", default=None, help="Comma-separated absolute phase shifts in degrees.")
    parser.add_argument("--ctf-defocus", type=float, default=None)
    parser.add_argument("--ctf-astigmatism", type=float, default=None)
    parser.add_argument("--ctf-astigmatism-angle", type=float, default=None)
    parser.add_argument("--ctf-phase-shift", type=float, default=None)
    parser.add_argument("--search-res", type=float, default=None)
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
    model_metadata, chosen_peak = load_model_fine_defaults(args.model_fine_dir)

    args.model_type = resolve_arg(args.model_type, model_metadata, "model_type", ops.model_type)
    args.workspace_root = resolve_arg(args.workspace_root, model_metadata, "workspace_root", ops.default_workspace_root)
    args.workspace_root_is_remote = resolve_arg(args.workspace_root_is_remote, model_metadata, "workspace_root_is_remote", ops.default_workspace_root_is_remote)
    args.session_name = resolve_arg(args.session_name, model_metadata, "session_name", ops.default_session_name)
    args.session_laser_state = resolve_arg(args.session_laser_state, model_metadata, "session_laser_state", ops.default_session_laser_state)
    args.session_job_type = resolve_arg(args.session_job_type, model_metadata, "session_job_type", ops.default_session_job_type)
    args.session_job_num = int(resolve_arg(args.session_job_num, model_metadata, "session_job_num", ops.default_session_job_num))
    args.particle_limit = resolve_arg(args.particle_limit, model_metadata, "particle_limit", None)
    args.micrograph_start_index = int(resolve_arg(args.micrograph_start_index, model_metadata, "micrograph_start_index", 0))
    args.search_res = float(resolve_arg(args.search_res, model_metadata, "search_res", 3.0))
    args.devices = resolve_arg(args.devices, model_metadata, "devices", "0,1,2,3")
    args.pdb_fpath = resolve_arg(args.pdb_fpath, model_metadata, "pdb_fpath", ops.DEFAULT_PDB_FPATH)
    args.defocus_offsets_A = resolve_arg(args.defocus_offsets_A, model_metadata, "defocus_offsets_A", [0.0])
    args.phase_shift_offsets_deg = [0.0] if args.phase_shift_offsets_deg is None else args.phase_shift_offsets_deg
    args.ctf_defocus = resolve_arg(args.ctf_defocus, model_metadata, "ctf_defocus", None)
    args.ctf_astigmatism = resolve_arg(args.ctf_astigmatism, model_metadata, "ctf_astigmatism", None)
    args.ctf_astigmatism_angle = resolve_arg(args.ctf_astigmatism_angle, model_metadata, "ctf_astigmatism_angle", None)
    args.ctf_phase_shift = resolve_arg(args.ctf_phase_shift, model_metadata, "ctf_phase_shift", None)

    if args.pixel_size is None:
        if "pixel_size" not in chosen_peak:
            raise ValueError("Pass --pixel-size or provide --model-fine-dir with model_fine peak JSON.")
        args.pixel_size = float(chosen_peak["pixel_size"])
    if args.B_factor is None:
        if "B_factor" not in chosen_peak:
            raise ValueError("Pass --B-factor or provide --model-fine-dir with model_fine peak JSON.")
        args.B_factor = float(chosen_peak["B_factor"])

    defocus_grid_values = parse_float_values(args.defocus_values_A)
    phase_grid_values = parse_float_values(args.phase_shift_values_deg)
    ctf_grid_size = (
        (len(defocus_grid_values) if defocus_grid_values is not None else len(args.defocus_offsets_A))
        * (len(phase_grid_values) if phase_grid_values is not None else len(args.phase_shift_offsets_deg))
    )
    if args.template_batch_size is None:
        args.template_batch_size = 4 if ctf_grid_size > 1 else 1

    device_ids = omf.parse_device_ids(args.devices)
    vd.make_context(device_ids=device_ids)

    if args.output_dir is None:
        args.output_dir = ops.make_default_output_dir("fixed_model_tm2d", args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "fixed_model_tm2d_results.star")

    config = ops.get_session_config(args)
    df_optics, df_particles, source_star_fpath = ps.get_dfs_from_session(config.session, job_type=config.job_type)
    print(f"source STAR: {source_star_fpath}")
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=config.job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = select_micrographs(mics_and_inds, args.num_micrographs, args.particles_per_micrograph, args.micrograph_start_index)
    print(f"loaded stack with {len(stack.im_orig)} particles from {len(mics_and_inds)} micrographs")
    print(f"selected {sum(len(inds) for _, inds in selected_mics)} particles from {len(selected_mics)} micrographs")

    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = ops.load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = ops.make_pose_library(protein_coords, args.search_res, ops.symmetry, ops.diameter_A)
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")

    template = ops.make_template(
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

    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = stack.Cs_nom_mm * 1e7
    base_ctf_params.B = float(args.B_factor)

    star_pixel_size = float(df_optics["rlnMicrographOriginalPixelSize"].iloc[0])
    stack_to_star = map_stack_indices_to_star_rows(stack, df_particles, config.session.workspace.root_dir, star_pixel_size)

    def write_outputs(rows, ctf_summaries, completed_micrographs, total_micrographs):
        write_csv(output_dir / "fixed_model_tm2d_results.csv", rows)
        write_csv(output_dir / "fixed_model_tm2d_ctf_grid.csv", ctf_summaries)
        df_out = attach_results_to_star(df_particles, rows, stack_to_star)
        starfile.write({"optics": df_optics, "particles": df_out}, args.output_star, overwrite=True)
        metadata = {
            "model_fine_dir": args.model_fine_dir,
            "source_star_fpath": source_star_fpath,
            "output_star": args.output_star,
            "model_type": args.model_type,
            "pdb_fpath": args.pdb_fpath,
            "pixel_size": float(args.pixel_size),
            "B_factor": float(args.B_factor),
            "search_res": args.search_res,
            "pose_library_size": int(len(pose_lib)),
            "defocus_offsets_A": args.defocus_offsets_A,
            "defocus_values_A": None if args.defocus_values_A is None else [float(v) for v in parse_float_values(args.defocus_values_A)],
            "phase_shift_offsets_deg": args.phase_shift_offsets_deg,
            "phase_shift_values_deg": None if args.phase_shift_values_deg is None else [float(v) for v in parse_float_values(args.phase_shift_values_deg)],
            "template_batch_size": int(args.template_batch_size),
            "template_batching_note": "CTF grids are passed through tm2d.make_ctf_set by setting searched CTF fields to None; Plan batches those CTF combinations across template_batch_size. A single fixed CTF combination uses template_batch_size=1 by default.",
            "workspace_root": args.workspace_root,
            "session_job_type": args.session_job_type,
            "session_job_num": args.session_job_num,
            "particle_limit": args.particle_limit,
            "num_micrographs": args.num_micrographs,
            "micrograph_start_index": args.micrograph_start_index,
            "particles_per_micrograph": args.particles_per_micrograph,
            "devices": device_ids,
            "completed_micrographs": int(completed_micrographs),
            "total_micrographs": int(total_micrographs),
            "n_result_rows": len(rows),
            "n_star_rows_with_results": int(df_out[TM_COLUMNS["z_score"]].notna().sum()),
        }
        with open(output_dir / "fixed_model_tm2d_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        print(
            f"updated {args.output_star}: {metadata['n_star_rows_with_results']} STAR rows with TM2D results "
            f"after {completed_micrographs}/{total_micrographs} micrographs"
        )

    rows, ctf_summaries = run_search(
        selected_mics,
        stack,
        image_shape,
        pose_lib,
        template,
        base_ctf_params,
        args,
        partial_writer=write_outputs,
    )
    if not selected_mics:
        write_outputs(rows, ctf_summaries, 0, 0)


if __name__ == "__main__":
    main()
