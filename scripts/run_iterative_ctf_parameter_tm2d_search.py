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
from tm2d_utils import whitener


INCOL = st.ITER_PIXEL_COLUMNS
OUTCOL = {
    "defocus": "rlnTM2DIterCTFDefocus",
    "defocus_u": "rlnTM2DIterCTFDefocusU",
    "defocus_v": "rlnTM2DIterCTFDefocusV",
    "defocus_angle": "rlnTM2DIterCTFDefocusAngle",
    "phase_shift": "rlnTM2DIterCTFPhaseShift",
    "z_score": "rlnTM2DIterCTFZScore",
    "mip": "rlnTM2DIterCTFMIP",
    "param_index": "rlnTM2DIterCTFParamIndex",
    "pixel_size": "rlnTM2DIterCTFPixelSize",
    "B_factor": "rlnTM2DIterCTFBFactor",
    "stack_index": "rlnTM2DIterCTFStackIndex",
}


write_csv = su.write_csv


def apply_spectral_filters(image, pixel_size, args):
    image = np.asarray(image, dtype=np.float32)
    if args.high_pass_cuton_start is not None or args.high_pass_cuton_end is not None:
        if args.high_pass_cuton_start is None or args.high_pass_cuton_end is None:
            raise ValueError("Pass both --high-pass-cuton-start and --high-pass-cuton-end")
        image = whitener.high_pass_filter_image(
            image,
            pixel_size=float(pixel_size),
            cuton_start=float(args.high_pass_cuton_start),
            cuton_end=float(args.high_pass_cuton_end),
        ).astype(np.float32, copy=False)
    if args.low_pass_cuton_start is not None or args.low_pass_cuton_end is not None:
        if args.low_pass_cuton_start is None or args.low_pass_cuton_end is None:
            raise ValueError("Pass both --low-pass-cuton-start and --low-pass-cuton-end")
        image = whitener.low_pass_filter_image(
            image,
            pixel_size=float(pixel_size),
            cuton_start=float(args.low_pass_cuton_start),
            cuton_end=float(args.low_pass_cuton_end),
        ).astype(np.float32, copy=False)
    return image


def build_ctf_grid(row, base_ctf_params, args):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)
    center_defocus = float(row[INCOL["defocus"]])
    center_phase = float(row[INCOL["phase_shift"]])
    astig = float(row[INCOL["astigmatism"]])
    astig_angle = float(row[INCOL["astigmatism_angle"]])

    if args.fixed_defocus_A is not None:
        defocus_values = np.asarray([float(args.fixed_defocus_A)], dtype=float)
    else:
        defocus_values = su.parse_float_values(args.defocus_values_A)
        if defocus_values is None:
            defocus_values = st.make_centered_range(center_defocus, args.defocus_half_width_A, args.defocus_step_A)

    if args.fixed_phase_shift_deg is not None:
        phase_values = np.asarray([float(args.fixed_phase_shift_deg)], dtype=float)
    else:
        phase_values = su.parse_float_values(args.phase_shift_values_deg)
        if phase_values is None:
            phase_values = st.make_centered_range(center_phase, args.phase_shift_half_width_deg, args.phase_shift_step_deg)

    ctf_params.defocus = None if len(defocus_values) > 1 else float(defocus_values[0])
    ctf_params.lpp = None if len(phase_values) > 1 else float(phase_values[0])
    ctf_params.A_mag = astig
    ctf_params.A_ang = astig_angle
    dynamic_values = {}
    if ctf_params.defocus is None:
        dynamic_values["defocus"] = defocus_values
    if ctf_params.lpp is None:
        dynamic_values["lpp"] = phase_values
    return ctf_params, dynamic_values, defocus_values, phase_values, center_defocus, center_phase


def orient_param_grid(values, defocus_values, phase_values):
    values = np.asarray(values, dtype=float).squeeze()
    n_defocus = len(defocus_values)
    n_phase = len(phase_values)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.shape == (n_defocus, n_phase):
        return values
    if values.shape == (n_phase, n_defocus):
        return values.T
    if values.size == n_defocus * n_phase:
        # tm2d CTF dynamic axes follow CTFParams field order: lpp before defocus.
        return values.reshape(n_phase, n_defocus).T
    if n_phase == 1 and values.size == n_defocus:
        return values.reshape(n_defocus, 1)
    if n_defocus == 1 and values.size == n_phase:
        return values.reshape(1, n_phase)
    raise ValueError(f"Could not orient parameter grid with shape {values.shape}; expected ({n_defocus}, {n_phase})")


def plot_1d_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path):
    phase_varies = len(phase_values) > 1
    if phase_varies:
        x = phase_values
        mip_line = mip_grid[0, :]
        z_line = z_grid[0, :]
        xlabel = "Phase shift [deg]"
        guess_x = center_phase
        fixed_text = f"defocus={float(defocus_values[0]):.1f} A"
    else:
        x = defocus_values
        mip_line = mip_grid[:, 0]
        z_line = z_grid[:, 0]
        xlabel = "Defocus [A]"
        guess_x = center_defocus
        fixed_text = f"phase={float(phase_values[0]):.2f} deg"

    fig, axes = plt.subplots(ncols=2, figsize=(11, 4.0))
    for ax, y, ylabel in [(axes[0], mip_line, "MIP"), (axes[1], z_line, "Z score")]:
        peak_ind = int(np.nanargmax(y))
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.1, color="C0")
        ax.scatter([x[peak_ind]], [y[peak_ind]], marker="o", facecolors="none", edgecolors="tab:red", s=95, linewidths=1.6, label="peak")
        if np.nanmin(x) <= guess_x <= np.nanmax(x):
            ax.axvline(guess_x, color="0.25", linestyle="--", linewidth=1.0, alpha=0.75, label="guess")
        ax.set_title(f"{ylabel}, peak={float(y[peak_ind]):.3f} at {x[peak_ind]:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")

    fig.suptitle(f"{os.path.basename(str(mic_name))}\n{fixed_text}; defocus guess={center_defocus:.1f} A, phase guess={center_phase:.2f} deg")
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_point_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path):
    fig, axes = plt.subplots(ncols=2, figsize=(9, 4.0))
    values = [(float(mip_grid[0, 0]), "MIP"), (float(z_grid[0, 0]), "Z score")]
    for ax, (value, ylabel) in zip(axes, values):
        ax.scatter([0], [value], marker="o", facecolors="none", edgecolors="tab:red", s=95, linewidths=1.6)
        ax.set_title(f"{ylabel}={value:.3f}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0])
        ax.set_xticklabels(["fixed"])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{os.path.basename(str(mic_name))}\ndefocus={float(defocus_values[0]):.1f} A, phase={float(phase_values[0]):.2f} deg")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path):
    mip_grid = orient_param_grid(mip_grid, defocus_values, phase_values)
    z_grid = orient_param_grid(z_grid, defocus_values, phase_values)
    if len(defocus_values) == 1 and len(phase_values) == 1:
        plot_single_point_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path)
        return
    if len(defocus_values) == 1 or len(phase_values) == 1:
        plot_1d_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path)
        return

    fig, axes = plt.subplots(ncols=2, figsize=(11, 4.5))
    extent = [float(np.min(phase_values)), float(np.max(phase_values)), float(np.min(defocus_values)), float(np.max(defocus_values))]
    im0 = axes[0].imshow(mip_grid, origin="lower", aspect="auto", extent=extent, cmap=cmc.lipari)
    axes[0].set_title(f"MIP, peak={np.nanmax(mip_grid):.3f}")
    axes[0].set_xlabel("Phase shift [deg]")
    axes[0].set_ylabel("Defocus [A]")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(z_grid, origin="lower", aspect="auto", extent=extent, cmap=cmc.lipari)
    axes[1].set_title(f"Z score, peak={np.nanmax(z_grid):.3f}")
    axes[1].set_xlabel("Phase shift [deg]")
    axes[1].set_ylabel("Defocus [A]")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    best_ind = np.unravel_index(int(np.nanargmax(z_grid)), z_grid.shape)
    best_defocus = defocus_values[best_ind[0]]
    best_phase = phase_values[best_ind[1]]
    for ax in axes:
        ax.scatter([center_phase], [center_defocus], marker="o", facecolors="none", edgecolors="white", s=80, linewidths=1.4, label="guess")
        ax.scatter([best_phase], [best_defocus], marker="o", facecolors="none", edgecolors="tab:red", s=95, linewidths=1.6, label="peak z")
        ax.legend(fontsize="small")
    fig.suptitle(f"{os.path.basename(str(mic_name))}\ndefocus guess={center_defocus:.1f} A, phase guess={center_phase:.2f} deg")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def attach_results(df_particles, rows):
    df_out = df_particles.copy()
    for c in OUTCOL.values():
        df_out[c] = np.nan
    for row in rows:
        values = {OUTCOL["defocus"]: row["defocus"], OUTCOL["defocus_u"]: row["defocus_u"], OUTCOL["defocus_v"]: row["defocus_v"], OUTCOL["defocus_angle"]: row["defocus_angle"], OUTCOL["phase_shift"]: row["phase_shift"], OUTCOL["z_score"]: row["z_score"], OUTCOL["mip"]: row["mip"], OUTCOL["param_index"]: row["param_index"], OUTCOL["pixel_size"]: row["pixel_size"], OUTCOL["B_factor"]: row["B_factor"], OUTCOL["stack_index"]: row["stack_index"]}
        for c, v in values.items():
            df_out.at[row["star_index"], c] = v
    return df_out


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative step 2: per-parameter defocus/phase search with pose and astigmatism fixed.")
    parser.add_argument("--iter-pixel-dir", default=None)
    parser.add_argument("--iter-pixel-star", default=None)
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
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--defocus-half-width-A", type=float, default=500.0)
    parser.add_argument("--defocus-step-A", type=float, default=10.0)
    parser.add_argument("--defocus-values-A", default=None)
    parser.add_argument("--fixed-defocus-A", type=float, default=None, help="Use one fixed defocus value for every particle; useful for 1D phase searches.")
    parser.add_argument("--phase-shift-half-width-deg", type=float, default=15.0)
    parser.add_argument("--phase-shift-step-deg", type=float, default=1.0)
    parser.add_argument("--phase-shift-values-deg", default=None)
    parser.add_argument("--fixed-phase-shift-deg", type=float, default=None, help="Use one fixed phase-shift value for every particle; useful for 1D defocus searches.")
    parser.add_argument("--template-batch-size", type=int, default=16)
    parser.add_argument("--output-radius", type=int, default=None)
    parser.add_argument("--high-pass-cuton-start", type=float, default=None, help="Raised-cosine high-pass start frequency in 1/A.")
    parser.add_argument("--high-pass-cuton-end", type=float, default=None, help="Raised-cosine high-pass end frequency in 1/A.")
    parser.add_argument("--low-pass-cuton-start", type=float, default=None, help="Raised-cosine low-pass start frequency in 1/A.")
    parser.add_argument("--low-pass-cuton-end", type=float, default=None, help="Raised-cosine low-pass end frequency in 1/A.")
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
    if args.iter_pixel_star is None:
        if args.iter_pixel_dir is None:
            raise ValueError("Pass --iter-pixel-dir or --iter-pixel-star")
        args.iter_pixel_star = str(st.find_star_file(args.iter_pixel_dir, "iter_pixel_tm2d_results.star"))
    metadata = su.load_json_if_exists(Path(args.iter_pixel_star).parent / "iter_pixel_tm2d_metadata.json")
    data = starfile.read(args.iter_pixel_star)
    df_optics = data["optics"]
    df_particles = data["particles"]
    st.require_columns(df_particles, [INCOL["angle_rot"], INCOL["angle_tilt"], INCOL["angle_psi"], INCOL["defocus"], INCOL["phase_shift"], INCOL["astigmatism"], INCOL["astigmatism_angle"], INCOL["z_score"], INCOL["stack_index"]])
    refine_df = df_particles[df_particles[INCOL["z_score"]].notna()].copy()
    if args.max_particles is not None:
        refine_df = refine_df.iloc[:int(args.max_particles)].copy()

    for attr, key, default in [("model_type", "model_type", su.DEFAULT_MODEL_TYPE), ("workspace_root", "workspace_root", su.DEFAULT_WORKSPACE_ROOT), ("workspace_root_is_remote", "workspace_root_is_remote", su.DEFAULT_WORKSPACE_ROOT_IS_REMOTE), ("session_name", "session_name", su.DEFAULT_SESSION_NAME), ("session_laser_state", "session_laser_state", su.DEFAULT_SESSION_LASER_STATE), ("session_job_type", "session_job_type", su.DEFAULT_SESSION_JOB_TYPE), ("particle_limit", "particle_limit", None), ("devices", "devices", su.DEFAULT_DEVICES), ("pdb_fpath", "pdb_fpath", su.DEFAULT_PDB_FPATH)]:
        if getattr(args, attr) is None:
            setattr(args, attr, metadata.get(key, default))
    if args.session_job_num is None:
        args.session_job_num = int(metadata.get("session_job_num", su.DEFAULT_SESSION_JOB_NUM))
    if args.pixel_size is None:
        args.pixel_size = float(metadata.get("pixel_size"))
    if args.B_factor is None:
        args.B_factor = float(metadata.get("B_factor"))

    vd.make_context(device_ids=su.parse_device_ids(args.devices))
    if args.output_dir is None:
        args.output_dir = "iter_ctf_param_tm2d"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "iter_ctf_param_tm2d_results.star")

    config = st.get_session_config(args)
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=args.session_job_type, n_particles=args.particle_limit)
    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = st.load_protein_coords(args.pdb_fpath)
    template = st.make_template(args.model_type, image_shape, protein_coords, pdb_fpath, float(args.pixel_size), args.output_dir, args.overwrite_density, args.density_helper_fpath, args.density_helper_python, args.density_device)
    base_ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    base_ctf_params.Cs = stack.Cs_nom_mm * 1e7
    base_ctf_params.B = float(args.B_factor)

    rows = []
    last_mic = None
    start_time = time.time()

    def write_outputs(done):
        write_csv(output_dir / "iter_ctf_param_tm2d_results.csv", rows)
        df_out = attach_results(df_particles, rows)
        starfile.write({"optics": df_optics, "particles": df_out}, args.output_star, overwrite=True)
        meta = vars(args).copy()
        meta.update({"completed_particles": done, "total_particles": len(refine_df), "elapsed_s": time.time() - start_time})
        with open(output_dir / "iter_ctf_param_tm2d_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
        print(f"updated {args.output_star}: {done}/{len(refine_df)} particles")

    diagnostic_mics = set()
    for done, (star_index, row) in enumerate(refine_df.iterrows(), start=1):
        mic_name = row.get("rlnMicrographName")
        if mic_name != last_mic and last_mic is not None:
            write_outputs(done - 1)
        last_mic = mic_name
        stack_index = int(row[INCOL["stack_index"]])
        if stack_index >= len(stack.im_orig):
            continue
        image, error_msg = st.preprocess_image_for_tm2d(stack.im_orig[stack_index], image_shape, image_shape, 0, float(args.pixel_size), False, None)
        if error_msg is not None:
            continue
        image = apply_spectral_filters(image, float(args.pixel_size), args)
        pose = np.asarray([row[INCOL["angle_rot"]], row[INCOL["angle_tilt"]], row[INCOL["angle_psi"]]], dtype=float)
        ctf_params, dynamic_values, defocus_values, phase_values, center_defocus, center_phase = build_ctf_grid(row, base_ctf_params, args)
        z_grid, mip_grid, param_index, defocus, phase, ctf_set = st.run_single_pose_ctf_param_search(image, image_shape, pose, template, ctf_params, dynamic_values, args.pixel_size, args.template_batch_size, output_radius=args.output_radius, progress=False)
        z_flat = np.asarray(z_grid, dtype=float).ravel()
        mip_flat = np.asarray(mip_grid, dtype=float).ravel()
        best_flat = int(np.nanargmax(z_flat))
        best_z = float(z_flat[best_flat])
        best_mip = float(mip_flat[best_flat])
        defocus_u, defocus_v, defocus_angle = tu.convert_tm2d_C1A1_to_rln(defocus, float(row[INCOL["astigmatism"]]), float(row[INCOL["astigmatism_angle"]]))
        rows.append({"star_index": star_index, "stack_index": stack_index, "rlnMicrographName": mic_name, "defocus": defocus, "defocus_u": float(defocus_u), "defocus_v": float(defocus_v), "defocus_angle": float(defocus_angle), "phase_shift": phase, "z_score": best_z, "mip": best_mip, "param_index": param_index, "pixel_size": float(args.pixel_size), "B_factor": float(args.B_factor)})
        if mic_name not in diagnostic_mics and len(diagnostic_mics) < args.diagnostic_micrographs:
            plot_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, diag_dir / f"ctf_param_search_mic{len(diagnostic_mics):03d}.png")
            diagnostic_mics.add(mic_name)
        if args.progress:
            print(f"{done}/{len(refine_df)} row {star_index}: ctf={ctf_set.get_length()}, z={best_z:.3f}")
    write_outputs(len(refine_df))


if __name__ == "__main__":
    main()
