import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import starfile

import tm2d
import tm2d_utils as tu
import vkdispatch as vd

import script_utils as su
import script_tm2d as st
from tm2d_utils import particle_stack as ps


ITER_COLUMNS = st.ITER_PIXEL_COLUMNS


write_csv = su.write_csv



def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    return default


def plot_pixel_diagnostic(mip, z_score, mic_name, pose, relion_pose, output_path, symmetry):
    peak_z_pix = np.unravel_index(int(np.nanargmax(z_score)), z_score.shape)
    peak_mip_pix = np.unravel_index(int(np.nanargmax(mip)), mip.shape)
    geodesic_error = tu.euler_geodesic_error(
        np.asarray(relion_pose, dtype=float),
        np.asarray(pose, dtype=float),
        symmetry=symmetry,
        degrees=True,
    )

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4.5))
    im0 = axes[0].imshow(mip, origin="lower", cmap=cmc.lipari)
    axes[0].scatter([peak_mip_pix[1]], [peak_mip_pix[0]], marker="o", facecolors="none", edgecolors="white", s=80, linewidths=1.4)
    axes[0].set_title(f"MIP, peak={np.nanmax(mip):.3f}")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(z_score, origin="lower", cmap=cmc.lipari)
    axes[1].scatter([peak_z_pix[1]], [peak_z_pix[0]], marker="o", facecolors="none", edgecolors="white", s=80, linewidths=1.4)
    axes[1].set_title(f"Z score, peak={np.nanmax(z_score):.3f}")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle(
        f"{os.path.basename(str(mic_name))}\n"
        f"pose guess=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}); "
        f"RELION pose=({relion_pose[0]:.2f}, {relion_pose[1]:.2f}, {relion_pose[2]:.2f}); "
        f"geodesic error={geodesic_error:.2f} deg"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative step 1: per-pixel TM2D search with fixed pixel size, B factor, and CTF settings.")
    parser.add_argument("--model-fine-dir", default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--B-factor", type=float, default=None)
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--symmetry", default=su.DEFAULT_SYMMETRY, help="Point-group symmetry used to reduce/search orientations, e.g. C1, C2, O.")
    parser.add_argument("--diameter-A", type=float, default=su.DEFAULT_DIAMETER_A, help="Particle diameter in Angstroms for Crowther angular sampling; defaults to coordinate-derived diameter.")
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
    parser.add_argument("--num-micrographs", type=int, default=None)
    parser.add_argument("--micrograph-start-index", type=int, default=0)
    parser.add_argument("--particles-per-micrograph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=su.DEFAULT_BATCH_SIZE)
    parser.add_argument("--template-batch-size", type=int, default=su.DEFAULT_PIXEL_TEMPLATE_BATCH_SIZE)
    parser.add_argument("--search-res", type=float, default=su.DEFAULT_SEARCH_RES_A)
    parser.add_argument("--ctf-defocus", type=float, default=None)
    parser.add_argument("--ctf-astigmatism", type=float, default=None)
    parser.add_argument("--ctf-astigmatism-angle", type=float, default=None)
    parser.add_argument("--ctf-phase-shift", type=float, default=None)
    parser.add_argument("--diagnostic-micrographs", type=int, default=10)
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
    args.search_res = float(args.search_res)
    args.devices = resolve_arg(args.devices, model_metadata, "devices", su.DEFAULT_DEVICES)
    args.pdb_fpath = resolve_arg(args.pdb_fpath, model_metadata, "pdb_fpath", su.DEFAULT_PDB_FPATH)
    if args.pixel_size is None:
        args.pixel_size = float(chosen_peak.get("pixel_size"))
    if args.B_factor is None:
        args.B_factor = float(chosen_peak.get("B_factor"))

    vd.make_context(device_ids=su.parse_device_ids(args.devices))
    if args.output_dir is None:
        args.output_dir = su.make_default_output_dir("iter_pixel_tm2d", args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "iter_pixel_tm2d_results.star")

    config = st.get_session_config(args)
    df_optics, df_particles, source_star_fpath = ps.get_dfs_from_session(config.session, job_type=args.session_job_type)
    print(f"source STAR: {source_star_fpath}")
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=args.session_job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = st.select_micrographs(mics_and_inds, args.num_micrographs, args.particles_per_micrograph, args.micrograph_start_index)
    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = st.load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = st.make_pose_library(protein_coords, args.search_res, args.symmetry, args.diameter_A)
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")
    template = st.make_template(args.model_type, image_shape, protein_coords, pdb_fpath, float(args.pixel_size), args.output_dir, args.overwrite_density, args.density_helper_fpath, args.density_helper_python, args.density_device)
    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = stack.Cs_nom_mm * 1e7
    base_ctf_params.B = float(args.B_factor)
    star_pixel_size = float(df_optics["rlnMicrographOriginalPixelSize"].iloc[0])
    stack_to_star = st.map_stack_indices_to_star_rows(stack, df_particles, config.session.workspace.root_dir, star_pixel_size)

    diagnostic_count = 0
    start_time = time.time()

    def row_callback(**payload):
        nonlocal diagnostic_count
        row = payload["row"]
        stack_index = row["stack_index"]
        star_index = stack_to_star.get(stack_index)
        if star_index is None:
            return None
        row["star_index"] = star_index

        mic_ind = payload["mic_ind"]
        p_rel_ind = payload["p_rel_ind"]
        if p_rel_ind == 0 and mic_ind < args.diagnostic_micrographs and diagnostic_count <= mic_ind:
            mic_stack = payload["mic_stack"]
            relion_pose = np.asarray([
                mic_stack.phi_in[p_rel_ind],
                mic_stack.theta_in[p_rel_ind],
                mic_stack.psi_in[p_rel_ind],
            ], dtype=float)
            pose = np.asarray([row["tmAngleRot"], row["tmAngleTilt"], row["tmAnglePsi"]], dtype=float)
            particle_batch_ind = payload["particle_batch_ind"]
            plot_pixel_diagnostic(
                payload["mip"][particle_batch_ind],
                payload["z_score"][particle_batch_ind],
                payload["mic_fpath"],
                pose,
                relion_pose,
                diag_dir / f"pixel_search_mic{mic_ind:03d}.png",
                symmetry=args.symmetry,
            )
            diagnostic_count += 1
        return row

    def write_outputs(rows, ctf_summaries, completed_micrographs, total_micrographs):
        write_csv(output_dir / "iter_pixel_tm2d_results.csv", rows)
        df_out = st.attach_tm_rows_to_star(df_particles, rows, ITER_COLUMNS)
        starfile.write({"optics": df_optics, "particles": df_out}, args.output_star, overwrite=True)
        metadata = vars(args).copy()
        metadata.update({
            "source_star_fpath": source_star_fpath,
            "completed_micrographs": completed_micrographs,
            "total_micrographs": total_micrographs,
            "n_rows": len(rows),
            "elapsed_s": time.time() - start_time,
        })
        with open(output_dir / "iter_pixel_tm2d_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"updated {args.output_star}: {len(rows)} rows after {completed_micrographs}/{total_micrographs} micrographs")

    rows, ctf_summaries = st.run_pixel_pose_search(
        selected_mics=selected_mics,
        stack=stack,
        image_shape=image_shape,
        pose_lib=pose_lib,
        template=template,
        base_ctf_params=base_ctf_params,
        args=args,
        ctf_builder=lambda mic_stack, base, a: st.make_micrograph_ctf_grid(mic_stack, base, a, allow_dynamic=False),
        row_callback=row_callback,
        partial_writer=write_outputs,
    )
    if not selected_mics:
        write_outputs(rows, ctf_summaries, 0, 0)


if __name__ == "__main__":
    main()
