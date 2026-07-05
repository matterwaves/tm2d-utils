import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tm2d
import tm2d_utils as tu

import spa
import file_handling as fh
from tm2d_utils import particle_stack as ps

import vkdispatch as vd


model_type = "atomic"  # "atomic" or "density"
particle_limit = 1000

pixel_size_guess_0 = 0.725  # [A]
B_factor_guess_0 = 50  # [A^2]
default_workspace_root = "/data/26_01_07-Apo"
default_workspace_root_is_remote = True
default_session_name = None
default_session_laser_state = None
default_session_job_type = "Refine3D"
default_session_job_num = 193
density_padding_factor = 2
symmetry = "O"
diameter_A = 130  # [A]
template_batch_size = 1
DEFAULT_DENSITY_HELPER_FPATHS = (
    "/home/ppetrov/GitHub/tm2d-utils/volume/simulate_tt_mrc.py",
    "/home/ppetrov/GitHub/theia_processing/volume/simulate_tt_mrc.py",
)


@dataclass
class SessionConfig:
    session: spa.RelionSession
    job_type: str
    color: str


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def sanitize_label(value):
    label = str(value).strip().replace("/", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label).strip("_")


def make_default_output_dir(prefix, args):
    root_label = sanitize_label(Path(args.workspace_root).name or "workspace")
    name_label = sanitize_label(args.session_name) if args.session_name is not None else "session"
    return (
        f"{prefix}_{args.model_type}_{root_label}_{name_label}_"
        f"{args.session_job_type}{args.session_job_num}_mic{args.micrograph_start_index}_n{args.num_micrographs}"
    )


def get_session_config(args):
    session_kwargs = {
        "workspace": fh.RelionWorkspace(root_dir=args.workspace_root, root_is_remote=args.workspace_root_is_remote),
        args.session_job_type: args.session_job_num,
    }
    if args.session_name not in {None, ""}:
        session_kwargs["name"] = args.session_name
    if args.session_laser_state is not None:
        session_kwargs["laser"] = args.session_laser_state

    session = spa.RelionSession(**session_kwargs)
    return SessionConfig(session=session, job_type=args.session_job_type, color="tab:blue")


def parse_float_list(spec):
    if spec is None or spec == "":
        return []
    return [float(value) for value in spec.split(",")]


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


def load_protein_coords():
    fname = "6z6u_apoferritin.pdb"
    pdb_dpath = "/home/ppetrov/GitHub/TEM_LPP_Image_Simulator/PDBs"
    pdb_fpath = os.path.join(pdb_dpath, fname)
    protein_coords = tu.load_coords_from_pdb(pdb_fpath)
    print(len(protein_coords), "atoms loaded from", pdb_fpath)
    return pdb_fpath, protein_coords


def resolve_density_helper_fpath(helper_fpath):
    if helper_fpath is not None:
        helper_path = Path(helper_fpath).expanduser()
        if not helper_path.exists():
            raise FileNotFoundError(f"Density helper not found: {helper_path}")
        return helper_path

    for candidate in DEFAULT_DENSITY_HELPER_FPATHS:
        helper_path = Path(candidate).expanduser()
        if helper_path.exists():
            return helper_path

    searched = ", ".join(DEFAULT_DENSITY_HELPER_FPATHS)
    raise FileNotFoundError(f"Could not find density helper. Searched: {searched}")


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
            padding_factor=density_padding_factor,
        )

    raise ValueError(f"Unknown model type: {model_type}")


def select_micrographs(mics_and_inds, num_micrographs, particles_per_micrograph, micrograph_start_index=0):
    start = int(micrograph_start_index)
    if start < 0:
        raise ValueError("micrograph_start_index must be non-negative")

    selected = []
    for mic_fpath, mic_abs_inds in mics_and_inds[start:start + num_micrographs]:
        selected.append((mic_fpath, list(mic_abs_inds[:particles_per_micrograph])))
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
):
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
                    best_params_index = best_index_array[particle_batch_ind][best_pixel]
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
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated vkdispatch device IDs for tm2d matching.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--density-helper-fpath", default=None, help="Path to simulate_tt_mrc.py; defaults to tm2d-utils/volume if present, then theia_processing/volume.")
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
    pdb_fpath, protein_coords = load_protein_coords()
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
