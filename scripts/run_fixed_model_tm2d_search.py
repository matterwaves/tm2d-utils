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


TM_COLUMNS = st.TM_COLUMNS


def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    return default



write_csv = su.write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run TM2D pose search with fixed optimized pixel size and B factor, writing a STAR with TM2D result columns.")
    parser.add_argument("--model-fine-dir", default=None, help="Directory from optimize_model_fine.py; used for metadata defaults and spline/sample peak values.")
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--B-factor", type=float, default=None)
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--symmetry", default=su.DEFAULT_SYMMETRY, help="Point-group symmetry used to reduce/search orientations, e.g. C1, C2, O.")
    parser.add_argument("--diameter-A", type=float, default=su.DEFAULT_DIAMETER_A, help="Particle diameter in Angstroms for Crowther angular sampling; defaults to coordinate-derived diameter.")
    parser.add_argument("--pdb-fpath", default=None, help="Path to the PDB used for atomic templates.")
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
    parser.add_argument("--num-micrographs", type=int, default=None)
    parser.add_argument("--micrograph-start-index", type=int, default=None)
    parser.add_argument("--particles-per-micrograph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--template-batch-size", type=int, default=None)
    parser.add_argument("--defocus-offsets-A", type=su.parse_float_list, default=None, help="Offsets from each micrograph mean TM2D defocus. Used unless --defocus-values-A is given.")
    parser.add_argument("--defocus-values-A", default=None, help="Comma-separated absolute TM2D defocus values in Angstroms.")
    parser.add_argument("--phase-shift-offsets-deg", type=su.parse_float_list, default=None, help="Offsets from each micrograph mean phase shift. Used unless --phase-shift-values-deg is given.")
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
    model_metadata, chosen_peak = st.load_model_fine_defaults(args.model_fine_dir)

    args.model_type = resolve_arg(args.model_type, model_metadata, "model_type", su.DEFAULT_MODEL_TYPE)
    args.workspace_root = resolve_arg(args.workspace_root, model_metadata, "workspace_root", su.DEFAULT_WORKSPACE_ROOT)
    args.workspace_root_is_remote = resolve_arg(args.workspace_root_is_remote, model_metadata, "workspace_root_is_remote", su.DEFAULT_WORKSPACE_ROOT_IS_REMOTE)
    args.session_name = resolve_arg(args.session_name, model_metadata, "session_name", su.DEFAULT_SESSION_NAME)
    args.session_laser_state = resolve_arg(args.session_laser_state, model_metadata, "session_laser_state", su.DEFAULT_SESSION_LASER_STATE)
    args.session_job_type = resolve_arg(args.session_job_type, model_metadata, "session_job_type", su.DEFAULT_SESSION_JOB_TYPE)
    args.session_job_num = int(resolve_arg(args.session_job_num, model_metadata, "session_job_num", su.DEFAULT_SESSION_JOB_NUM))
    args.particle_limit = resolve_arg(args.particle_limit, model_metadata, "particle_limit", None)
    args.micrograph_start_index = int(resolve_arg(args.micrograph_start_index, model_metadata, "micrograph_start_index", 0))
    args.search_res = float(resolve_arg(args.search_res, model_metadata, "search_res", 3.0))
    args.devices = resolve_arg(args.devices, model_metadata, "devices", su.DEFAULT_DEVICES)
    args.pdb_fpath = resolve_arg(args.pdb_fpath, model_metadata, "pdb_fpath", su.DEFAULT_PDB_FPATH)
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

    defocus_grid_values = su.parse_float_values(args.defocus_values_A)
    phase_grid_values = su.parse_float_values(args.phase_shift_values_deg)
    ctf_grid_size = (
        (len(defocus_grid_values) if defocus_grid_values is not None else len(args.defocus_offsets_A))
        * (len(phase_grid_values) if phase_grid_values is not None else len(args.phase_shift_offsets_deg))
    )
    if args.template_batch_size is None:
        args.template_batch_size = 4 if ctf_grid_size > 1 else 1

    device_ids = su.parse_device_ids(args.devices)
    vd.make_context(device_ids=device_ids)

    if args.output_dir is None:
        args.output_dir = su.make_default_output_dir("fixed_model_tm2d", args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "fixed_model_tm2d_results.star")

    config = st.get_session_config(args)
    df_optics, df_particles, source_star_fpath = ps.get_dfs_from_session(config.session, job_type=config.job_type)
    print(f"source STAR: {source_star_fpath}")
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=config.job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = st.select_micrographs(mics_and_inds, args.num_micrographs, args.particles_per_micrograph, args.micrograph_start_index)
    print(f"loaded stack with {len(stack.im_orig)} particles from {len(mics_and_inds)} micrographs")
    print(f"selected {sum(len(inds) for _, inds in selected_mics)} particles from {len(selected_mics)} micrographs")

    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = st.load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = st.make_pose_library(protein_coords, args.search_res, args.symmetry, args.diameter_A)
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")

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

    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = stack.Cs_nom_mm * 1e7
    base_ctf_params.B = float(args.B_factor)

    star_pixel_size = float(df_optics["rlnMicrographOriginalPixelSize"].iloc[0])
    stack_to_star = st.map_stack_indices_to_star_rows(stack, df_particles, config.session.workspace.root_dir, star_pixel_size)

    def write_outputs(rows, ctf_summaries, completed_micrographs, total_micrographs):
        write_csv(output_dir / "fixed_model_tm2d_results.csv", rows)
        write_csv(output_dir / "fixed_model_tm2d_ctf_grid.csv", ctf_summaries)
        df_out = st.attach_tm_rows_to_star(df_particles, rows, TM_COLUMNS, row_to_star_index=stack_to_star)
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
            "defocus_values_A": None if args.defocus_values_A is None else [float(v) for v in su.parse_float_values(args.defocus_values_A)],
            "phase_shift_offsets_deg": args.phase_shift_offsets_deg,
            "phase_shift_values_deg": None if args.phase_shift_values_deg is None else [float(v) for v in su.parse_float_values(args.phase_shift_values_deg)],
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

    rows, ctf_summaries = st.run_pixel_pose_search(
        selected_mics=selected_mics,
        stack=stack,
        image_shape=image_shape,
        pose_lib=pose_lib,
        template=template,
        base_ctf_params=base_ctf_params,
        args=args,
        ctf_builder=st.make_micrograph_ctf_grid,
        partial_writer=write_outputs,
    )
    if not selected_mics:
        write_outputs(rows, ctf_summaries, 0, 0)


if __name__ == "__main__":
    main()
