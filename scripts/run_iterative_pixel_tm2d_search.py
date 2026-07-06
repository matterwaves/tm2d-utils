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

import optimize_model_fine as omf
import optimize_pixel_size as ops
import run_fixed_model_tm2d_search as fixed
from tm2d_utils import particle_stack as ps


ITER_COLUMNS = {
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


def load_model_fine_defaults(model_fine_dir):
    if model_fine_dir is None:
        return {}, {}
    return fixed.load_model_fine_defaults(model_fine_dir)


def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    return default


def build_fixed_ctf_params(mic_stack, base_ctf_params, args):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)

    defocus_u = np.asarray(mic_stack.defocus_u_in, dtype=float)
    defocus_v = np.asarray(mic_stack.defocus_v_in, dtype=float)
    defocus_ang = np.asarray(mic_stack.defocus_ang_in, dtype=float)
    phase_shift = np.asarray(mic_stack.phase_shift_in, dtype=float)
    defocus_tm2d, A_mag, A_ang = tu.convert_rln_C1A1_to_tm2d(defocus_u, defocus_v, defocus_ang)

    ctf_params.defocus = float(np.nanmean(defocus_tm2d) if args.ctf_defocus is None else args.ctf_defocus)
    ctf_params.A_mag = float(np.nanmean(A_mag) if args.ctf_astigmatism is None else args.ctf_astigmatism)
    ctf_params.A_ang = float(np.nanmean(A_ang) if args.ctf_astigmatism_angle is None else args.ctf_astigmatism_angle)
    ctf_params.lpp = float(np.nanmean(phase_shift) if args.ctf_phase_shift is None else args.ctf_phase_shift)
    ctf_params.B = float(args.B_factor)
    return ctf_params


def plot_pixel_diagnostic(mip, z_score, mic_name, pose, relion_pose, output_path, symmetry=ops.symmetry):
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


def attach_results_to_star(df_particles, rows):
    df_out = df_particles.copy()
    for column in ITER_COLUMNS.values():
        df_out[column] = np.nan
    for row in rows:
        values = {
            ITER_COLUMNS["angle_rot"]: row["tmAngleRot"],
            ITER_COLUMNS["angle_tilt"]: row["tmAngleTilt"],
            ITER_COLUMNS["angle_psi"]: row["tmAnglePsi"],
            ITER_COLUMNS["defocus"]: row["tmDefocus"],
            ITER_COLUMNS["defocus_u"]: row["tmDefocusU"],
            ITER_COLUMNS["defocus_v"]: row["tmDefocusV"],
            ITER_COLUMNS["defocus_angle"]: row["tmDefocusAngle"],
            ITER_COLUMNS["astigmatism"]: row["tmAstigmatism"],
            ITER_COLUMNS["astigmatism_angle"]: row["tmAstigmatismAngle"],
            ITER_COLUMNS["phase_shift"]: row["tmPhaseShift"],
            ITER_COLUMNS["z_score"]: row["tmZScore"],
            ITER_COLUMNS["mip"]: row["tmMip"],
            ITER_COLUMNS["peak_x"]: row["tmPeakX"],
            ITER_COLUMNS["peak_y"]: row["tmPeakY"],
            ITER_COLUMNS["peak_offset_x"]: row["tmPeakOffsetX"],
            ITER_COLUMNS["peak_offset_y"]: row["tmPeakOffsetY"],
            ITER_COLUMNS["pixel_size"]: row["tmPixelSize"],
            ITER_COLUMNS["B_factor"]: row["tmBFactor"],
            ITER_COLUMNS["param_index"]: row["tmParamIndex"],
            ITER_COLUMNS["mic_index"]: row["mic_ind"],
            ITER_COLUMNS["particle_rel_index"]: row["particle_rel_ind"],
            ITER_COLUMNS["stack_index"]: row["stack_index"],
        }
        for column, value in values.items():
            df_out.at[row["star_index"], column] = value
    return df_out


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative step 1: per-pixel TM2D search with fixed pixel size, B factor, and CTF settings.")
    parser.add_argument("--model-fine-dir", default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--B-factor", type=float, default=None)
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--pdb-fpath", default=None)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=ops.parse_bool, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--session-laser-state", type=ops.parse_bool, default=None)
    parser.add_argument("--session-job-type", default=None)
    parser.add_argument("--session-job-num", type=int, default=None)
    parser.add_argument("--particle-limit", type=int, default=None)
    parser.add_argument("--num-micrographs", type=int, default=None)
    parser.add_argument("--micrograph-start-index", type=int, default=0)
    parser.add_argument("--particles-per-micrograph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--template-batch-size", type=int, default=1)
    parser.add_argument("--search-res", type=float, default=None)
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
    model_metadata, chosen_peak = load_model_fine_defaults(args.model_fine_dir)
    args.model_type = resolve_arg(args.model_type, model_metadata, "model_type", ops.model_type)
    args.workspace_root = resolve_arg(args.workspace_root, model_metadata, "workspace_root", ops.default_workspace_root)
    args.workspace_root_is_remote = resolve_arg(args.workspace_root_is_remote, model_metadata, "workspace_root_is_remote", ops.default_workspace_root_is_remote)
    args.session_name = resolve_arg(args.session_name, model_metadata, "session_name", ops.default_session_name)
    args.session_laser_state = resolve_arg(args.session_laser_state, model_metadata, "session_laser_state", ops.default_session_laser_state)
    args.session_job_type = resolve_arg(args.session_job_type, model_metadata, "session_job_type", ops.default_session_job_type)
    args.session_job_num = int(resolve_arg(args.session_job_num, model_metadata, "session_job_num", ops.default_session_job_num))
    args.search_res = float(resolve_arg(args.search_res, model_metadata, "search_res", 3.0))
    args.devices = resolve_arg(args.devices, model_metadata, "devices", "0,1,2,3")
    args.pdb_fpath = resolve_arg(args.pdb_fpath, model_metadata, "pdb_fpath", ops.DEFAULT_PDB_FPATH)
    if args.pixel_size is None:
        args.pixel_size = float(chosen_peak.get("pixel_size"))
    if args.B_factor is None:
        args.B_factor = float(chosen_peak.get("B_factor"))

    vd.make_context(device_ids=omf.parse_device_ids(args.devices))
    if args.output_dir is None:
        args.output_dir = ops.make_default_output_dir("iter_pixel_tm2d", args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "iter_pixel_tm2d_results.star")

    config = ops.get_session_config(args)
    df_optics, df_particles, source_star_fpath = ps.get_dfs_from_session(config.session, job_type=args.session_job_type)
    print(f"source STAR: {source_star_fpath}")
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=args.session_job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = fixed.select_micrographs(mics_and_inds, args.num_micrographs, args.particles_per_micrograph, args.micrograph_start_index)
    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = ops.load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = ops.make_pose_library(protein_coords, args.search_res, ops.symmetry, ops.diameter_A)
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")
    template = ops.make_template(args.model_type, image_shape, protein_coords, pdb_fpath, float(args.pixel_size), args.output_dir, args.overwrite_density, args.density_helper_fpath, args.density_helper_python, args.density_device)
    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = stack.Cs_nom_mm * 1e7
    base_ctf_params.B = float(args.B_factor)
    star_pixel_size = float(df_optics["rlnMicrographOriginalPixelSize"].iloc[0])
    stack_to_star = fixed.map_stack_indices_to_star_rows(stack, df_particles, config.session.workspace.root_dir, star_pixel_size)

    rows = []
    diagnostic_count = 0
    start_time = time.time()

    def write_outputs(done_mics):
        write_csv(output_dir / "iter_pixel_tm2d_results.csv", rows)
        df_out = attach_results_to_star(df_particles, rows)
        starfile.write({"optics": df_optics, "particles": df_out}, args.output_star, overwrite=True)
        metadata = vars(args).copy()
        metadata.update({"source_star_fpath": source_star_fpath, "completed_micrographs": done_mics, "n_rows": len(rows), "elapsed_s": time.time() - start_time})
        with open(output_dir / "iter_pixel_tm2d_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"updated {args.output_star}: {len(rows)} rows after {done_mics}/{len(selected_mics)} micrographs")

    for mic_ind, (mic_fpath, mic_abs_inds) in enumerate(selected_mics):
        mic_stack = stack.get_substack(mic_abs_inds)
        ctf_params = build_fixed_ctf_params(mic_stack, base_ctf_params, args)
        ctf_set = tm2d.make_ctf_set(ctf_params)
        param_set = tm2d.make_param_set(ctf_set, rotations=pose_lib, pixel_sizes=np.array([float(args.pixel_size)]))
        print(f"micrograph {mic_ind + 1}/{len(selected_mics)}: {len(mic_abs_inds)} particles, {len(pose_lib)} poses")
        for batch_number, p_rel_inds in enumerate(ops.get_particle_batches(len(mic_abs_inds), args.batch_size), start=1):
            mics_in, good_rel_inds = ops.build_tm2d_input_batch(mic_stack, p_rel_inds, image_shape, image_shape, mic_ind, batch_number, 0, float(args.pixel_size), False, None)
            if mics_in is None:
                continue
            results = ops.run_tm2d_pixels(mics_in, param_set, template, ctf_params, int(args.template_batch_size), args.progress)
            mip = results.get_mip()
            best_index_array = results.get_best_index_array()
            z_score = tu.get_pixel_z_scores(results)
            for particle_batch_ind, p_rel_ind in enumerate(good_rel_inds):
                best_pixel = np.unravel_index(int(np.nanargmax(z_score[particle_batch_ind])), z_score.shape[1:])
                param_index = int(best_index_array[particle_batch_ind][best_pixel])
                best_values = param_set.get_values_at_index(param_index)
                rotation = best_values["rotation"]
                defocus_u, defocus_v, defocus_angle = tu.convert_tm2d_C1A1_to_rln(ctf_params.defocus, ctf_params.A_mag, ctf_params.A_ang)
                stack_index = int(mic_abs_inds[p_rel_ind])
                star_index = stack_to_star.get(stack_index)
                if star_index is None:
                    continue
                peak_y, peak_x = int(best_pixel[0]), int(best_pixel[1])
                rows.append({"star_index": star_index, "stack_index": stack_index, "mic_ind": mic_ind, "mic_fpath": mic_fpath, "particle_rel_ind": int(p_rel_ind), "tmAngleRot": float(rotation[0]), "tmAngleTilt": float(rotation[1]), "tmAnglePsi": float(rotation[2]), "tmDefocus": float(ctf_params.defocus), "tmDefocusU": float(defocus_u), "tmDefocusV": float(defocus_v), "tmDefocusAngle": float(defocus_angle), "tmAstigmatism": float(ctf_params.A_mag), "tmAstigmatismAngle": float(ctf_params.A_ang), "tmPhaseShift": float(ctf_params.lpp), "tmZScore": float(z_score[particle_batch_ind][best_pixel]), "tmMip": float(mip[particle_batch_ind][best_pixel]), "tmPeakX": peak_x, "tmPeakY": peak_y, "tmPeakOffsetX": float(peak_x - image_shape[1] / 2.0), "tmPeakOffsetY": float(peak_y - image_shape[0] / 2.0), "tmPixelSize": float(args.pixel_size), "tmBFactor": float(args.B_factor), "tmParamIndex": param_index})
                if p_rel_ind == 0 and mic_ind < args.diagnostic_micrographs and diagnostic_count <= mic_ind:
                    relion_pose = np.asarray([
                        mic_stack.phi_in[p_rel_ind],
                        mic_stack.theta_in[p_rel_ind],
                        mic_stack.psi_in[p_rel_ind],
                    ], dtype=float)
                    plot_pixel_diagnostic(
                        mip[particle_batch_ind],
                        z_score[particle_batch_ind],
                        mic_fpath,
                        rotation,
                        relion_pose,
                        diag_dir / f"pixel_search_mic{mic_ind:03d}.png",
                    )
                    diagnostic_count += 1
        write_outputs(mic_ind + 1)


if __name__ == "__main__":
    main()
