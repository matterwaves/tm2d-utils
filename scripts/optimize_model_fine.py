import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tm2d_utils as tu
import vkdispatch as vd

import optimize_pixel_size as ops


B_factor_guess_0 = 50.0


def parse_range(center, half_width, step):
    n_steps = int(round((2.0 * half_width) / step))
    values = center - half_width + step * np.arange(n_steps + 1)
    return np.round(values, 6)


def parse_float_values(spec):
    if spec is None or spec == "":
        return None
    return np.asarray([float(value) for value in spec.split(",")], dtype=float)


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_pixel_size_metadata(pixel_size_dir):
    return load_json_if_exists(Path(pixel_size_dir) / "pixel_size_optimization_metadata.json")


def load_best_fine_pixel_size(pixel_size_dir, summary_fpath=None):
    summary_path = Path(summary_fpath) if summary_fpath is not None else Path(pixel_size_dir) / "pixel_size_optimization_summary.csv"
    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage") != "fine":
                continue
            row["pixel_size"] = float(row["pixel_size"])
            row["objective_z_score"] = float(row["objective_z_score"])
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No fine-stage rows found in {summary_path}")

    return max(rows, key=lambda row: row["objective_z_score"])


def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if key in metadata and metadata[key] is not None:
        return metadata[key]
    return default


def parse_device_ids(devices):
    if isinstance(devices, str):
        return [int(device) for device in devices.split(",") if device != ""]
    return [int(device) for device in devices]


def build_objective_grid(summary_rows, pixel_sizes, B_factors):
    objective_by_key = {
        (float(row["B_factor"]), float(row["pixel_size"])): float(row["objective_z_score"])
        for row in summary_rows
    }
    z_grid = np.full((len(B_factors), len(pixel_sizes)), np.nan, dtype=float)
    for b_ind, B_factor in enumerate(B_factors):
        for p_ind, pixel_size in enumerate(pixel_sizes):
            z_grid[b_ind, p_ind] = objective_by_key.get((float(B_factor), float(pixel_size)), np.nan)
    return z_grid


def estimate_model_peaks(summary_rows, pixel_sizes, B_factors):
    sampled_best = max(summary_rows, key=lambda row: row["objective_z_score"])
    estimates = {
        "sampled": {
            "pixel_size": float(sampled_best["pixel_size"]),
            "B_factor": float(sampled_best["B_factor"]),
            "objective_z_score": float(sampled_best["objective_z_score"]),
        },
        "spline": None,
    }

    z_grid = build_objective_grid(summary_rows, pixel_sizes, B_factors)
    if len(pixel_sizes) < 2 or len(B_factors) < 2 or not np.isfinite(z_grid).all():
        return estimates

    try:
        from scipy.interpolate import RectBivariateSpline

        kx = min(3, len(B_factors) - 1)
        ky = min(3, len(pixel_sizes) - 1)
        spline = RectBivariateSpline(B_factors, pixel_sizes, z_grid, kx=kx, ky=ky)
        B_fit = np.linspace(float(np.min(B_factors)), float(np.max(B_factors)), 400)
        p_fit = np.linspace(float(np.min(pixel_sizes)), float(np.max(pixel_sizes)), 400)
        z_fit = spline(B_fit, p_fit)
        best_ind = np.unravel_index(int(np.nanargmax(z_fit)), z_fit.shape)
        estimates["spline"] = {
            "pixel_size": float(p_fit[best_ind[1]]),
            "B_factor": float(B_fit[best_ind[0]]),
            "objective_z_score": float(z_fit[best_ind]),
            "spline_order_B": int(kx),
            "spline_order_pixel_size": int(ky),
        }
    except Exception as exc:
        print(f"Could not estimate model-fine spline peak: {exc}")

    return estimates


def format_model_fine_result_text(model_type, summary_rows, pixel_sizes, B_factors):
    estimates = estimate_model_peaks(summary_rows, pixel_sizes, B_factors)
    sampled = estimates["sampled"]
    lines = [
        f"Model: {model_type}",
        (
            f"Sampled peak: p={sampled['pixel_size']:.5f} A, "
            f"B={sampled['B_factor']:.2f} A^2, "
            f"objective z={sampled['objective_z_score']:.3f}"
        ),
    ]
    spline = estimates.get("spline")
    if spline is not None:
        lines.append(
            f"Interpolated peak: p={spline['pixel_size']:.5f} A, "
            f"B={spline['B_factor']:.2f} A^2, "
            f"objective z={spline['objective_z_score']:.3f}"
        )
    return "\n".join(lines)


def save_model_fine_plot(summary_rows, pixel_sizes, B_factors, output_dir, model_type=None):
    if not summary_rows:
        return

    z_grid = build_objective_grid(summary_rows, pixel_sizes, B_factors)
    peak_estimates = estimate_model_peaks(summary_rows, pixel_sizes, B_factors)
    sampled_peak = peak_estimates["sampled"]
    spline_peak = peak_estimates.get("spline")

    fig, axes = plt.subplots(ncols=3, figsize=(16, 4.5))
    if model_type is not None:
        fig.suptitle(format_model_fine_result_text(model_type, summary_rows, pixel_sizes, B_factors), fontsize=11)

    im = axes[0].imshow(
        z_grid,
        origin="lower",
        aspect="auto",
        extent=[
            float(np.min(pixel_sizes)),
            float(np.max(pixel_sizes)),
            float(np.min(B_factors)),
            float(np.max(B_factors)),
        ],
    )
    axes[0].scatter(
        [sampled_peak["pixel_size"]],
        [sampled_peak["B_factor"]],
        marker="x",
        s=80,
        color="black",
        linewidths=1.8,
        label="Sampled peak",
        zorder=5,
    )
    if spline_peak is not None:
        axes[0].scatter(
            [spline_peak["pixel_size"]],
            [spline_peak["B_factor"]],
            marker="*",
            s=130,
            color="tab:red",
            edgecolors="white",
            linewidths=0.6,
            label="Interpolated peak",
            zorder=6,
        )
    axes[0].set_title("Objective grid")
    axes[0].set_xlabel("Pixel size [A]")
    axes[0].set_ylabel("B factor [A^2]")
    axes[0].legend(fontsize="small", loc="best")
    fig.colorbar(im, ax=axes[0], label="Objective z score")

    row_colors = plt.cm.viridis(np.linspace(0, 1, len(B_factors)))
    for b_ind, B_factor in enumerate(B_factors):
        axes[1].plot(pixel_sizes, z_grid[b_ind, :], marker="o", color=row_colors[b_ind], label=f"B={B_factor:g}")
    sampled_b_ind = int(np.argmin(np.abs(np.asarray(B_factors, dtype=float) - sampled_peak["B_factor"])))
    axes[1].scatter(
        [sampled_peak["pixel_size"]],
        [sampled_peak["objective_z_score"]],
        marker="x",
        s=70,
        color="black",
        linewidths=1.8,
        label="Sampled peak",
        zorder=5,
    )
    if spline_peak is not None:
        axes[1].axvline(spline_peak["pixel_size"], color="tab:red", linestyle="--", linewidth=1)
        axes[1].scatter(
            [spline_peak["pixel_size"]],
            [spline_peak["objective_z_score"]],
            marker="*",
            s=110,
            color="tab:red",
            edgecolors="white",
            linewidths=0.6,
            label="Interpolated peak",
            zorder=6,
        )
    axes[1].set_title(f"Rows: fixed B; sampled row B={B_factors[sampled_b_ind]:g}")
    axes[1].set_xlabel("Pixel size [A]")
    axes[1].set_ylabel("Objective z score")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize="small")

    col_colors = plt.cm.plasma(np.linspace(0, 1, len(pixel_sizes)))
    for p_ind, pixel_size in enumerate(pixel_sizes):
        axes[2].plot(B_factors, z_grid[:, p_ind], marker="o", color=col_colors[p_ind], label=f"p={pixel_size:.4f}")
    sampled_p_ind = int(np.argmin(np.abs(np.asarray(pixel_sizes, dtype=float) - sampled_peak["pixel_size"])))
    axes[2].scatter(
        [sampled_peak["B_factor"]],
        [sampled_peak["objective_z_score"]],
        marker="x",
        s=70,
        color="black",
        linewidths=1.8,
        label="Sampled peak",
        zorder=5,
    )
    if spline_peak is not None:
        axes[2].axvline(spline_peak["B_factor"], color="tab:red", linestyle="--", linewidth=1)
        axes[2].scatter(
            [spline_peak["B_factor"]],
            [spline_peak["objective_z_score"]],
            marker="*",
            s=110,
            color="tab:red",
            edgecolors="white",
            linewidths=0.6,
            label="Interpolated peak",
            zorder=6,
        )
    axes[2].set_title(f"Columns: fixed pixel; sampled p={pixel_sizes[sampled_p_ind]:.5f}")
    axes[2].set_xlabel("B factor [A^2]")
    axes[2].set_ylabel("Objective z score")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize="small")

    fig.tight_layout(rect=[0, 0, 1, 0.84] if model_type is not None else None)
    fig.savefig(Path(output_dir) / "model_fine_pixel_size_B_factor_grid.png", dpi=200)
    plt.close(fig)


def make_template_for_pixel(pixel_size, image_shape, protein_coords, pdb_fpath, args):
    return ops.make_template(
        args.model_type,
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


def run_model_fine_grid(pixel_sizes, B_factors, selected_mics, stack, image_shape, pose_lib, protein_coords, pdb_fpath, ctf_params, args):
    raw_rows_all = []
    summary_rows = []

    for pix_ind, pixel_size in enumerate(pixel_sizes, start=1):
        print(f"working on pixel size {pix_ind}/{len(pixel_sizes)}: {pixel_size:.5f} A")
        template = make_template_for_pixel(
            pixel_size,
            image_shape=image_shape,
            protein_coords=protein_coords,
            pdb_fpath=pdb_fpath,
            args=args,
        )

        for b_ind, B_factor in enumerate(B_factors, start=1):
            print(f"... working on B factor {b_ind}/{len(B_factors)}: {B_factor:.2f} A^2")
            ctf_params.B = float(B_factor)
            raw_rows = ops.run_pixel_size_candidate(
                stage="model_fine",
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
            for row in raw_rows:
                row["B_factor"] = float(B_factor)
            raw_rows_all.extend(raw_rows)

            summary = ops.summarize_scores(
                pixel_size=float(pixel_size),
                stage="model_fine",
                raw_rows=raw_rows,
                top_n=args.top_n,
                score_stat=args.score_stat,
            )
            summary["B_factor"] = float(B_factor)
            summary_rows.append(summary)
            print(
                f"... p={pixel_size:.5f} A, B={B_factor:.2f} A^2: "
                f"objective z = {summary['objective_z_score']:.3f}"
            )

            ops.write_csv(Path(args.output_dir) / "model_fine_summary.csv", summary_rows)
            ops.write_csv(Path(args.output_dir) / "model_fine_particles.csv", raw_rows_all)
            save_model_fine_plot(summary_rows, pixel_sizes, B_factors, args.output_dir, model_type=args.model_type)

    best = max(summary_rows, key=lambda row: row["objective_z_score"])
    peak_estimates = estimate_model_peaks(summary_rows, pixel_sizes, B_factors)
    with open(Path(args.output_dir) / "model_fine_best.json", "w") as f:
        json.dump(best, f, indent=4)
    with open(Path(args.output_dir) / "model_fine_peak_estimates.json", "w") as f:
        json.dump(peak_estimates, f, indent=4)
    save_model_fine_plot(summary_rows, pixel_sizes, B_factors, args.output_dir, model_type=args.model_type)
    return summary_rows, raw_rows_all, best


def parse_args():
    parser = argparse.ArgumentParser(description="Fine 2D model search over pixel size and B factor after optimize_pixel_size.py.")
    parser.add_argument("--pixel-size-dir", default="pixel_size_optimization_atomic_out")
    parser.add_argument("--pixel-size-summary", default=None)
    parser.add_argument("--pixel-size", type=float, default=None, help="Override the fine-search best sampled pixel size.")
    parser.add_argument("--pixel-size-half-width", type=float, default=0.002)
    parser.add_argument("--pixel-size-step", type=float, default=0.0005)
    parser.add_argument("--B-factor-estimate", type=float, default=B_factor_guess_0)
    parser.add_argument("--B-factor-half-width", type=float, default=50.0)
    parser.add_argument("--B-factor-step", type=float, default=10.0)
    parser.add_argument("--B-factor-values", default="", help="Comma-separated B factors; overrides estimate/half-width/step.")
    parser.add_argument("--pdb-fpath", default=None, help="Path to the PDB used for atomic templates.")
    parser.add_argument("--model-type", default=None, choices=["atomic", "density"])
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=ops.parse_bool, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--session-laser-state", type=ops.parse_bool, default=None)
    parser.add_argument("--session-job-type", default=None)
    parser.add_argument("--session-job-num", type=int, default=None)
    parser.add_argument("--particle-limit", type=int, default=None)
    parser.add_argument("--num-micrographs", type=int, default=None)
    parser.add_argument("--micrograph-start-index", type=int, default=None, help="Zero-based micrograph index to start deterministic sampling from.")
    parser.add_argument("--particles-per-micrograph", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--score-stat", default=None, choices=["mean", "median"])
    parser.add_argument("--defocus-offsets-A", type=ops.parse_float_list, default=None, help="Comma-separated defocus offsets in Angstroms.")
    parser.add_argument("--ctf-defocus", type=float, default=None, help="Override STAR defocus with a constant value in Angstroms.")
    parser.add_argument("--ctf-astigmatism", type=float, default=None, help="Override STAR astigmatism magnitude with a constant value in Angstroms.")
    parser.add_argument("--ctf-astigmatism-angle", type=float, default=None, help="Override STAR astigmatism angle with a constant value in degrees.")
    parser.add_argument("--ctf-phase-shift", type=float, default=None, help="Override STAR phase shift with a constant value in degrees.")
    parser.add_argument("--search-res", type=float, default=None)
    parser.add_argument("--devices", default=None, help="Comma-separated vkdispatch device IDs for tm2d matching.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--density-helper-fpath", default=None)
    parser.add_argument("--density-helper-python", default=None)
    parser.add_argument("--density-device", default="cpu")
    parser.add_argument("--overwrite-density", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = load_pixel_size_metadata(args.pixel_size_dir)
    best_fine_row = load_best_fine_pixel_size(args.pixel_size_dir, args.pixel_size_summary)

    args.model_type = resolve_arg(args.model_type, metadata, "model_type", ops.model_type)
    args.workspace_root = resolve_arg(args.workspace_root, metadata, "workspace_root", ops.default_workspace_root)
    args.workspace_root_is_remote = resolve_arg(args.workspace_root_is_remote, metadata, "workspace_root_is_remote", ops.default_workspace_root_is_remote)
    args.session_name = resolve_arg(args.session_name, metadata, "session_name", ops.default_session_name)
    args.session_laser_state = resolve_arg(args.session_laser_state, metadata, "session_laser_state", ops.default_session_laser_state)
    args.session_job_type = resolve_arg(args.session_job_type, metadata, "session_job_type", ops.default_session_job_type)
    args.session_job_num = int(resolve_arg(args.session_job_num, metadata, "session_job_num", ops.default_session_job_num))
    args.particle_limit = int(resolve_arg(args.particle_limit, metadata, "particle_limit", ops.particle_limit))
    args.num_micrographs = int(resolve_arg(args.num_micrographs, metadata, "num_micrographs", 4))
    args.micrograph_start_index = int(resolve_arg(args.micrograph_start_index, metadata, "micrograph_start_index", 0))
    args.particles_per_micrograph = int(resolve_arg(args.particles_per_micrograph, metadata, "particles_per_micrograph", 50))
    args.top_n = int(resolve_arg(args.top_n, metadata, "top_n", 100))
    args.score_stat = resolve_arg(args.score_stat, metadata, "score_stat", "mean")
    args.search_res = float(resolve_arg(args.search_res, metadata, "search_res", 3.0))
    args.defocus_offsets_A = resolve_arg(args.defocus_offsets_A, metadata, "defocus_offsets_A", [0.0])
    args.ctf_defocus = resolve_arg(args.ctf_defocus, metadata, "ctf_defocus", None)
    args.ctf_astigmatism = resolve_arg(args.ctf_astigmatism, metadata, "ctf_astigmatism", None)
    args.ctf_astigmatism_angle = resolve_arg(args.ctf_astigmatism_angle, metadata, "ctf_astigmatism_angle", None)
    args.ctf_phase_shift = resolve_arg(args.ctf_phase_shift, metadata, "ctf_phase_shift", None)
    args.devices = resolve_arg(args.devices, metadata, "devices", "0,1,2,3")
    args.pdb_fpath = resolve_arg(args.pdb_fpath, metadata, "pdb_fpath", ops.DEFAULT_PDB_FPATH)

    device_ids = parse_device_ids(args.devices)
    vd.make_context(device_ids=device_ids)

    center_pixel_size = float(args.pixel_size) if args.pixel_size is not None else float(best_fine_row["pixel_size"])
    pixel_sizes = parse_range(center_pixel_size, args.pixel_size_half_width, args.pixel_size_step)
    B_factor_values = parse_float_values(args.B_factor_values)
    if B_factor_values is None:
        B_factor_values = parse_range(args.B_factor_estimate, args.B_factor_half_width, args.B_factor_step)
    B_factor_values = np.sort(np.asarray(B_factor_values, dtype=float))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ops.make_default_output_dir("model_fine", args) + f"_B{args.B_factor_estimate:g}"
    args.output_dir = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = ops.get_session_config(args)
    print("loading stack...")
    stack = ops.ps.read_stack_from_session(config.session, job_type=config.job_type, n_particles=args.particle_limit)
    mics_and_inds = stack.get_substacks_inds_by_field("mic_fpath")
    selected_mics = ops.select_micrographs(
        mics_and_inds,
        args.num_micrographs,
        args.particles_per_micrograph,
        micrograph_start_index=args.micrograph_start_index,
    )
    print(f"loaded stack with {len(stack.im_orig)} particles from {len(mics_and_inds)} micrographs")
    print(f"selected {sum(len(inds) for _, inds in selected_mics)} particles from {len(selected_mics)} micrographs")

    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = ops.load_protein_coords(args.pdb_fpath)
    pose_lib, diameter, ang_step = ops.make_pose_library(
        protein_coords=protein_coords,
        search_res=args.search_res,
        symmetry=ops.symmetry,
        diameter_a=ops.diameter_A,
    )
    print(f"pose library size: {len(pose_lib)} orientations, diameter: {diameter:.2f} A, angular step: {ang_step:.2f} deg")

    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.Cs = stack.Cs_nom_mm * 1e7
    ctf_params.B = float(args.B_factor_estimate)

    run_metadata = {
        "pixel_size_dir": args.pixel_size_dir,
        "pixel_size_summary": args.pixel_size_summary,
        "starting_fine_pixel_size": center_pixel_size,
        "starting_fine_objective_z_score": float(best_fine_row["objective_z_score"]),
        "B_factor_estimate": args.B_factor_estimate,
        "pixel_size_half_width": args.pixel_size_half_width,
        "pixel_size_step": args.pixel_size_step,
        "B_factor_half_width": args.B_factor_half_width,
        "B_factor_step": args.B_factor_step,
        "B_factor_values": [float(v) for v in B_factor_values],
        "pdb_fpath": args.pdb_fpath,
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
        "top_n": args.top_n,
        "score_stat": args.score_stat,
        "defocus_offsets_A": args.defocus_offsets_A,
        "ctf_defocus": args.ctf_defocus,
        "ctf_astigmatism": args.ctf_astigmatism,
        "ctf_astigmatism_angle": args.ctf_astigmatism_angle,
        "ctf_phase_shift": args.ctf_phase_shift,
        "search_res": args.search_res,
        "pose_library_size": int(len(pose_lib)),
        "devices": device_ids,
    }
    with open(Path(output_dir) / "model_fine_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=4)

    _, _, best = run_model_fine_grid(
        pixel_sizes=pixel_sizes,
        B_factors=B_factor_values,
        selected_mics=selected_mics,
        stack=stack,
        image_shape=image_shape,
        pose_lib=pose_lib,
        protein_coords=protein_coords,
        pdb_fpath=pdb_fpath,
        ctf_params=ctf_params,
        args=args,
    )
    print(
        f"best model: pixel size={best['pixel_size']:.5f} A, "
        f"B={best['B_factor']:.2f} A^2, objective z={best['objective_z_score']:.3f}"
    )


if __name__ == "__main__":
    main()
