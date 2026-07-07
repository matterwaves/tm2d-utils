import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tm2d
import tm2d_utils as tu
import script_utils as su
import script_tm2d as st

from tm2d_utils import particle_stack as ps

import vkdispatch as vd


model_type = su.DEFAULT_MODEL_TYPE
particle_limit = su.DEFAULT_PARTICLE_LIMIT

pixel_size_guess_0 = su.DEFAULT_PIXEL_SIZE_GUESS_A  # [A]
B_factor_guess_0 = su.DEFAULT_B_FACTOR_A2  # [A^2]
default_workspace_root = su.DEFAULT_WORKSPACE_ROOT
default_workspace_root_is_remote = su.DEFAULT_WORKSPACE_ROOT_IS_REMOTE
default_remote_host = su.DEFAULT_REMOTE_HOST
default_remote_user = su.DEFAULT_REMOTE_USER
default_remote_key_filename = su.DEFAULT_REMOTE_KEY_FILENAME
default_session_name = su.DEFAULT_SESSION_NAME
default_session_laser_state = su.DEFAULT_SESSION_LASER_STATE
default_session_job_type = su.DEFAULT_SESSION_JOB_TYPE
default_session_job_num = su.DEFAULT_SESSION_JOB_NUM
density_padding_factor = su.DEFAULT_DENSITY_PADDING_FACTOR
symmetry = su.DEFAULT_APOF_SYMMETRY
diameter_A = su.DEFAULT_APOF_DIAMETER_A  # [A]
template_batch_size = su.DEFAULT_PIXEL_TEMPLATE_BATCH_SIZE
DEFAULT_PDB_FPATH = su.DEFAULT_PDB_FPATH
DEFAULT_DENSITY_HELPER_FPATHS = su.DEFAULT_DENSITY_HELPER_FPATHS

parse_bool = su.parse_bool
sanitize_label = su.sanitize_label
make_default_output_dir = su.make_default_output_dir


get_session_config = st.get_session_config
parse_float_list = su.parse_float_list
make_pixel_grid = st.make_pixel_grid
make_pose_library = st.make_pose_library
get_particle_batches = st.get_particle_batches
build_tm2d_input_batch = st.build_tm2d_input_batch
preprocess_image_for_tm2d = st.preprocess_image_for_tm2d
run_tm2d_pixels = st.run_tm2d_pixels
load_protein_coords = st.load_protein_coords
resolve_density_helper_fpath = st.resolve_density_helper_fpath
make_density_mrc = st.make_density_mrc
make_template = st.make_template
select_micrographs = st.select_micrographs
update_ctf_for_micrograph = st.update_ctf_for_micrograph


get_best_rows_by_particle = st.get_best_rows_by_particle
summarize_scores = st.summarize_scores
run_pixel_size_candidate = st.run_pixel_size_candidate
write_csv = su.write_csv


def fit_fine_spline(rows):
    finite_rows = [
        row for row in rows
        if np.isfinite(row["pixel_size"]) and np.isfinite(row["objective_z_score"])
    ]
    if len(finite_rows) < 3:
        return None

    x = np.asarray([row["pixel_size"] for row in finite_rows], dtype=float)
    y = np.asarray([row["objective_z_score"] for row in finite_rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_inds = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_inds]
    if x.size < 3 or float(np.ptp(x)) == 0:
        return None

    spline_order = min(3, x.size - 1)
    try:
        from scipy.interpolate import InterpolatedUnivariateSpline

        spline = InterpolatedUnivariateSpline(x, y, k=spline_order)
    except Exception as exc:
        print(f"Could not fit fine pixel-size spline: {exc}")
        return None

    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 1000)
    y_fit = spline(x_fit)
    peak_ind = int(np.nanargmax(y_fit))
    peak_pixel_size = float(x_fit[peak_ind])
    peak_objective = float(y_fit[peak_ind])

    return {
        "spline_order": int(spline_order),
        "peak_pixel_size": peak_pixel_size,
        "peak_objective_z_score": peak_objective,
        "sampled_peak_pixel_size": float(x[int(np.nanargmax(y))]),
        "sampled_peak_objective_z_score": float(np.nanmax(y)),
        "x_fit": x_fit,
        "y_fit": y_fit,
    }

def format_plot_result_text(model_type, summary_rows):
    best = max(summary_rows, key=lambda row: row["objective_z_score"])
    lines = [
        f"Model: {model_type}",
        (
            f"Sampled peak: p={best['pixel_size']:.5f} A "
            f"({best['stage']}), objective z={best['objective_z_score']:.3f}"
        ),
    ]

    fine_rows = [row for row in summary_rows if row["stage"] == "fine"]
    fit = fit_fine_spline(fine_rows) if fine_rows else None
    if fit is not None:
        lines.append(
            f"Interpolated fine peak: p={fit['peak_pixel_size']:.5f} A, "
            f"objective z={fit['peak_objective_z_score']:.3f}"
        )
    return "\n".join(lines)


def save_plot(summary_rows, output_dir, model_type=None):
    if not summary_rows:
        return

    fig, axes = plt.subplots(ncols=2, figsize=(11, 4), sharey=False)
    if model_type is not None:
        fig.suptitle(format_plot_result_text(model_type, summary_rows), fontsize=11)
    for ax, stage in zip(axes, ["coarse", "fine"]):
        rows = [row for row in summary_rows if row["stage"] == stage]
        ax.set_title(stage.capitalize())
        ax.set_xlabel("Pixel size [A]")
        ax.grid(alpha=0.25)
        if not rows:
            ax.text(0.5, 0.5, "No results", ha="center", va="center", transform=ax.transAxes)
            continue

        rows.sort(key=lambda row: row["pixel_size"])
        ax.plot(
            [row["pixel_size"] for row in rows],
            [row["objective_z_score"] for row in rows],
            marker="o",
            label="data",
        )
        sampled_peak = max(rows, key=lambda row: row["objective_z_score"])
        ax.scatter(
            [sampled_peak["pixel_size"]],
            [sampled_peak["objective_z_score"]],
            marker="x",
            s=70,
            color="black",
            linewidths=1.8,
            label="Sampled peak",
            zorder=5,
        )

        if stage == "fine":
            fit = fit_fine_spline(rows)
            if fit is not None:
                ax.plot(fit["x_fit"], fit["y_fit"], color="tab:red", label="Spline interpolation")
                ax.axvline(fit["peak_pixel_size"], color="tab:red", linestyle="--", linewidth=1)
                ax.scatter(
                    [fit["peak_pixel_size"]],
                    [fit["peak_objective_z_score"]],
                    marker="*",
                    s=110,
                    color="tab:red",
                    edgecolors="white",
                    linewidths=0.6,
                    label="Interpolated peak",
                    zorder=6,
                )
                ax.text(
                    0.03,
                    0.97,
                    f"peak = {fit['peak_pixel_size']:.5f} A",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )
                fit_out = dict(fit)
                fit_out.pop("x_fit")
                fit_out.pop("y_fit")
                with open(Path(output_dir) / "pixel_size_optimization_fine_spline_fit.json", "w") as f:
                    json.dump(fit_out, f, indent=4)

        ax.legend()

    axes[0].set_ylabel("Objective z score")
    fig.tight_layout(rect=[0, 0, 1, 0.86] if model_type is not None else None)
    fig.savefig(Path(output_dir) / "pixel_size_optimization.png", dpi=200)
    plt.close(fig)



def run_stage(
    stage,
    pixel_sizes,
    selected_mics,
    stack,
    image_shape,
    pose_lib,
    model_type,
    protein_coords,
    pdb_fpath,
    ctf_params,
    args,
):
    raw_rows_all = []
    summary_rows = []
    atomic_template = None
    if model_type == "atomic":
        atomic_template = make_template(
            model_type,
            image_shape,
            protein_coords,
            pdb_fpath,
            pixel_size=float(pixel_sizes[0]),
            output_dir=args.output_dir,
            overwrite_density=args.overwrite_density,
            density_helper_fpath=args.density_helper_fpath,
            density_helper_python=args.density_helper_python,
            density_device=args.density_device,
        )

    for pix_ind, pixel_size in enumerate(pixel_sizes, start=1):
        print(f"working on {stage} pixel size {pix_ind}/{len(pixel_sizes)}: {pixel_size:.5f} A")
        template = atomic_template
        if model_type == "density":
            template = make_template(
                model_type,
                image_shape,
                protein_coords,
                pdb_fpath,
                pixel_size=float(pixel_size),
                output_dir=args.output_dir,
                overwrite_density=args.overwrite_density,
                density_helper_fpath=args.density_helper_fpath,
                density_helper_python=args.density_helper_python,
                density_device=args.density_device,
            )

        raw_rows = run_pixel_size_candidate(
            stage=stage,
            pixel_size=float(pixel_size),
            selected_mics=selected_mics,
            stack=stack,
            image_shape=image_shape,
            pose_lib=pose_lib,
            template=template,
            ctf_params=ctf_params,
            defocus_offsets=args.defocus_offsets_A,
            defocus_override=args.ctf_defocus,
            astigmatism_override=args.ctf_astigmatism,
            astigmatism_angle_override=args.ctf_astigmatism_angle,
            phase_shift_override=args.ctf_phase_shift,
            batch_size=args.batch_size,
            enable_progress_bar=args.progress,
        )
        raw_rows_all.extend(raw_rows)
        summary = summarize_scores(
            pixel_size=float(pixel_size),
            stage=stage,
            raw_rows=raw_rows,
            top_n=args.top_n,
            score_stat=args.score_stat,
        )
        summary_rows.append(summary)
        print(f"... {stage} {pixel_size:.5f} A: objective z = {summary['objective_z_score']:.3f}")

    return summary_rows, raw_rows_all


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize 2DTM pixel size using ensemble peak z scores.")
    parser.add_argument("--model-type", default=model_type, choices=["atomic", "density"])
    parser.add_argument("--workspace-root", default=default_workspace_root)
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=parse_bool, default=default_workspace_root_is_remote)
    parser.add_argument("--remote-host", default=default_remote_host)
    parser.add_argument("--remote-user", default=default_remote_user)
    parser.add_argument("--remote-key-filename", default=default_remote_key_filename)
    parser.add_argument("--session-name", default=default_session_name)
    parser.add_argument("--session-laser-state", type=parse_bool, default=default_session_laser_state)
    parser.add_argument("--session-job-type", default=default_session_job_type)
    parser.add_argument("--session-job-num", type=int, default=default_session_job_num)
    parser.add_argument("--particle-limit", type=int, default=particle_limit)
    parser.add_argument("--num-micrographs", type=int, default=4)
    parser.add_argument("--micrograph-start-index", type=int, default=0, help="Zero-based micrograph index to start deterministic sampling from.")
    parser.add_argument("--particles-per-micrograph", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pixel-size-guess", type=float, default=pixel_size_guess_0)
    parser.add_argument("--coarse-half-width", type=float, default=0.02)
    parser.add_argument("--coarse-step", type=float, default=0.005)
    parser.add_argument("--fine-half-width", type=float, default=0.004)
    parser.add_argument("--fine-step", type=float, default=0.001)
    parser.add_argument("--skip-fine", action="store_true")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--score-stat", default="mean", choices=["mean", "median"])
    parser.add_argument("--defocus-offsets-A", type=parse_float_list, default=[0.0], help="Comma-separated defocus offsets in Angstroms.")
    parser.add_argument("--ctf-defocus", type=float, default=None, help="Override STAR defocus with a constant value in Angstroms.")
    parser.add_argument("--ctf-astigmatism", type=float, default=None, help="Override STAR astigmatism magnitude with a constant value in Angstroms.")
    parser.add_argument("--ctf-astigmatism-angle", type=float, default=None, help="Override STAR astigmatism angle with a constant value in degrees.")
    parser.add_argument("--ctf-phase-shift", type=float, default=None, help="Override STAR phase shift with a constant value in degrees.")
    parser.add_argument("--search-res", type=float, default=3.0)
    parser.add_argument("--B-factor", type=float, default=B_factor_guess_0)
    parser.add_argument("--pdb-fpath", default=DEFAULT_PDB_FPATH, help="Path to the PDB used for atomic templates.")
    parser.add_argument("--devices", default=su.DEFAULT_DEVICES, help="Comma-separated vkdispatch device IDs for tm2d matching.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--density-helper-fpath", default=None, help="Path to simulate_tt_mrc.py; defaults can be set in tm2d_utils/local_parameters.py.")
    parser.add_argument("--density-helper-python", default=None, help="Python used to launch the density helper; defaults to the current tm2d Python.")
    parser.add_argument("--density-device", default="cpu", help="Device passed to the density-template generator only.")
    parser.add_argument("--overwrite-density", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = [int(device) for device in args.devices.split(",") if device != ""]
    vd.make_context(device_ids=device_ids)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = make_default_output_dir("pixel_size_optimization", args)
    args.output_dir = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = get_session_config(args)
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=config.job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = select_micrographs(
        mics_and_inds,
        args.num_micrographs,
        args.particles_per_micrograph,
        micrograph_start_index=args.micrograph_start_index,
    )
    print(f"loaded stack with {len(stack.im_orig)} particles from {len(mics_and_inds)} micrographs")
    print(f"selected {sum(len(inds) for _, inds in selected_mics)} particles from {len(selected_mics)} micrographs")

    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = make_pose_library(
        protein_coords=protein_coords,
        search_res=args.search_res,
        symmetry=symmetry,
        diameter_a=diameter_A,
    )
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")

    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.Cs = stack.Cs_nom_mm * 1e7
    ctf_params.B = args.B_factor

    metadata = {
        "model_type": args.model_type,
        "workspace_root": args.workspace_root,
        "workspace_root_is_remote": args.workspace_root_is_remote,
        "session_name": args.session_name,
        "session_laser_state": args.session_laser_state,
        "session_job_type": args.session_job_type,
        "session_job_num": args.session_job_num,
        "particle_limit": args.particle_limit,
        "num_micrographs": args.num_micrographs,
        "micrograph_start_index": args.micrograph_start_index,
        "particles_per_micrograph": args.particles_per_micrograph,
        "selected_micrographs": [{"mic_fpath": mic_fpath, "n_particles": len(inds)} for mic_fpath, inds in selected_mics],
        "pixel_size_guess": args.pixel_size_guess,
        "defocus_offsets_A": args.defocus_offsets_A,
        "ctf_defocus": args.ctf_defocus,
        "ctf_astigmatism": args.ctf_astigmatism,
        "ctf_astigmatism_angle": args.ctf_astigmatism_angle,
        "ctf_phase_shift": args.ctf_phase_shift,
        "top_n": args.top_n,
        "score_stat": args.score_stat,
        "B_factor": args.B_factor,
        "pdb_fpath": args.pdb_fpath,
        "search_res": args.search_res,
        "pose_library_size": int(len(pose_lib)),
        "devices": device_ids,
        "density_helper_fpath": args.density_helper_fpath,
        "density_helper_python": args.density_helper_python,
        "density_device": args.density_device,
    }
    with open(Path(output_dir) / "pixel_size_optimization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    all_summary_rows = []
    all_raw_rows = []
    coarse_pixel_sizes = make_pixel_grid(args.pixel_size_guess, args.coarse_half_width, args.coarse_step)
    coarse_summary, coarse_raw = run_stage(
        stage="coarse",
        pixel_sizes=coarse_pixel_sizes,
        selected_mics=selected_mics,
        stack=stack,
        image_shape=image_shape,
        pose_lib=pose_lib,
        model_type=args.model_type,
        protein_coords=protein_coords,
        pdb_fpath=pdb_fpath,
        ctf_params=ctf_params,
        args=args,
    )
    all_summary_rows.extend(coarse_summary)
    all_raw_rows.extend(coarse_raw)
    write_csv(Path(output_dir) / "pixel_size_optimization_summary.csv", all_summary_rows)
    write_csv(Path(output_dir) / "pixel_size_optimization_particles.csv", all_raw_rows)
    save_plot(all_summary_rows, output_dir, model_type=args.model_type)

    best_coarse = max(coarse_summary, key=lambda row: row["objective_z_score"])
    print(f"best coarse pixel size: {best_coarse['pixel_size']:.5f} A")

    if not args.skip_fine:
        fine_pixel_sizes = make_pixel_grid(best_coarse["pixel_size"], args.fine_half_width, args.fine_step)
        fine_summary, fine_raw = run_stage(
            stage="fine",
            pixel_sizes=fine_pixel_sizes,
            selected_mics=selected_mics,
            stack=stack,
            image_shape=image_shape,
            pose_lib=pose_lib,
            model_type=args.model_type,
            protein_coords=protein_coords,
            pdb_fpath=pdb_fpath,
            ctf_params=ctf_params,
            args=args,
        )
        all_summary_rows.extend(fine_summary)
        all_raw_rows.extend(fine_raw)
        write_csv(Path(output_dir) / "pixel_size_optimization_summary.csv", all_summary_rows)
        write_csv(Path(output_dir) / "pixel_size_optimization_particles.csv", all_raw_rows)
        save_plot(all_summary_rows, output_dir, model_type=args.model_type)

    best = max(all_summary_rows, key=lambda row: row["objective_z_score"])
    with open(Path(output_dir) / "pixel_size_optimization_best.json", "w") as f:
        json.dump(best, f, indent=4)
    print(f"best pixel size: {best['pixel_size']:.5f} A ({best['stage']}), objective z = {best['objective_z_score']:.3f}")


if __name__ == "__main__":
    main()
