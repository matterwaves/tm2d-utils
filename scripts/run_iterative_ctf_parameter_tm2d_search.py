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
import run_iterative_pixel_tm2d_search as iterpix
from tm2d_utils import particle_stack as ps


INCOL = iterpix.ITER_COLUMNS
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


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def find_iter_pixel_star(iter_pixel_dir):
    p = Path(iter_pixel_dir)
    candidate = p / "iter_pixel_tm2d_results.star"
    if candidate.exists():
        return candidate
    matches = sorted(p.glob("*.star"))
    if not matches:
        raise FileNotFoundError(f"No STAR file found in {p}")
    return matches[0]


def parse_float_values(spec):
    if spec is None or str(spec).strip() == "":
        return None
    return np.asarray([float(v) for v in str(spec).split(",") if v != ""], dtype=float)


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


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required STAR columns: {', '.join(missing)}")


def build_ctf_grid(row, base_ctf_params, args):
    ctf_params = tu.ctf_like_theia(defocus=0, Cs=0)
    ctf_params.__dict__.update(base_ctf_params.__dict__)
    center_defocus = float(row[INCOL["defocus"]])
    center_phase = float(row[INCOL["phase_shift"]])
    astig = float(row[INCOL["astigmatism"]])
    astig_angle = float(row[INCOL["astigmatism_angle"]])
    defocus_values = parse_float_values(args.defocus_values_A)
    if defocus_values is None:
        defocus_values = make_centered_range(center_defocus, args.defocus_half_width_A, args.defocus_step_A)
    phase_values = parse_float_values(args.phase_shift_values_deg)
    if phase_values is None:
        phase_values = make_centered_range(center_phase, args.phase_shift_half_width_deg, args.phase_shift_step_deg)
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


def run_param_search(image, image_shape, pose, template, ctf_params, dynamic_values, args):
    ctf_set = tm2d.make_ctf_set(ctf_params, **dynamic_values)
    param_set = tm2d.make_param_set(ctf_set, rotations=np.asarray([pose], dtype=float), pixel_sizes=np.array([float(args.pixel_size)]))
    results = tm2d.ResultsParam(1, param_set.get_total_count(), output_radius=args.output_radius)
    comparator = tm2d.ComparatorCrossCorrelation(shape=(1, *image_shape), template_shape=template.get_shape())
    plan = tm2d.Plan(template=template, comparator=comparator, results=results, ctf_params=ctf_params, template_batch_size=int(args.template_batch_size), output_radius=args.output_radius, enable_rotation_weights=False)
    plan.set_data(np.asarray([image], dtype=np.float32))
    plan.run(param_set, enable_progress_bar=False)
    z_values = np.squeeze(results.get_zscore_list(param_set))
    mip_values = np.squeeze(results.get_mip_list(param_set))
    flat_best = int(np.nanargmax(z_values))
    param_index = int(np.ravel_multi_index(np.unravel_index(flat_best, z_values.shape), z_values.shape)) if z_values.ndim > 0 else 0
    best_values = param_set.get_values_at_index(param_index)
    defocus = float(best_values.get("defocus", ctf_params.defocus))
    phase = float(best_values.get("lpp", ctf_params.lpp))
    return z_values, mip_values, param_index, defocus, phase, ctf_set


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


def plot_param_diagnostic(mip_grid, z_grid, defocus_values, phase_values, mic_name, center_defocus, center_phase, output_path):
    mip_grid = orient_param_grid(mip_grid, defocus_values, phase_values)
    z_grid = orient_param_grid(z_grid, defocus_values, phase_values)
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
    parser.add_argument("--workspace-root-is-remote", "--remote-is-true", dest="workspace_root_is_remote", type=ops.parse_bool, default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--session-laser-state", type=ops.parse_bool, default=None)
    parser.add_argument("--session-job-type", default=None)
    parser.add_argument("--session-job-num", type=int, default=None)
    parser.add_argument("--particle-limit", type=int, default=None)
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--defocus-half-width-A", type=float, default=500.0)
    parser.add_argument("--defocus-step-A", type=float, default=10.0)
    parser.add_argument("--defocus-values-A", default=None)
    parser.add_argument("--phase-shift-half-width-deg", type=float, default=15.0)
    parser.add_argument("--phase-shift-step-deg", type=float, default=1.0)
    parser.add_argument("--phase-shift-values-deg", default=None)
    parser.add_argument("--template-batch-size", type=int, default=16)
    parser.add_argument("--output-radius", type=int, default=None)
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
        args.iter_pixel_star = str(find_iter_pixel_star(args.iter_pixel_dir))
    metadata = load_json_if_exists(Path(args.iter_pixel_star).parent / "iter_pixel_tm2d_metadata.json")
    data = starfile.read(args.iter_pixel_star)
    df_optics = data["optics"]
    df_particles = data["particles"]
    require_columns(df_particles, [INCOL["angle_rot"], INCOL["angle_tilt"], INCOL["angle_psi"], INCOL["defocus"], INCOL["phase_shift"], INCOL["astigmatism"], INCOL["astigmatism_angle"], INCOL["z_score"], INCOL["stack_index"]])
    refine_df = df_particles[df_particles[INCOL["z_score"]].notna()].copy()
    if args.max_particles is not None:
        refine_df = refine_df.iloc[:int(args.max_particles)].copy()

    for attr, key, default in [("model_type", "model_type", ops.model_type), ("workspace_root", "workspace_root", ops.default_workspace_root), ("workspace_root_is_remote", "workspace_root_is_remote", ops.default_workspace_root_is_remote), ("session_name", "session_name", ops.default_session_name), ("session_laser_state", "session_laser_state", ops.default_session_laser_state), ("session_job_type", "session_job_type", ops.default_session_job_type), ("particle_limit", "particle_limit", None), ("devices", "devices", "0,1,2,3"), ("pdb_fpath", "pdb_fpath", ops.DEFAULT_PDB_FPATH)]:
        if getattr(args, attr) is None:
            setattr(args, attr, metadata.get(key, default))
    if args.session_job_num is None:
        args.session_job_num = int(metadata.get("session_job_num", ops.default_session_job_num))
    if args.pixel_size is None:
        args.pixel_size = float(metadata.get("pixel_size"))
    if args.B_factor is None:
        args.B_factor = float(metadata.get("B_factor"))

    vd.make_context(device_ids=omf.parse_device_ids(args.devices))
    if args.output_dir is None:
        args.output_dir = "iter_ctf_param_tm2d"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    if args.output_star is None:
        args.output_star = str(output_dir / "iter_ctf_param_tm2d_results.star")

    config = ops.get_session_config(args)
    print("loading stack...")
    stack = ps.read_stack_from_session(config.session, job_type=args.session_job_type, n_particles=args.particle_limit)
    image_shape = tuple(stack.im_orig[0].shape)
    pdb_fpath, protein_coords = ops.load_protein_coords(args.pdb_fpath)
    template = ops.make_template(args.model_type, image_shape, protein_coords, pdb_fpath, float(args.pixel_size), args.output_dir, args.overwrite_density, args.density_helper_fpath, args.density_helper_python, args.density_device)
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
        image, error_msg = ops.preprocess_image_for_tm2d(stack.im_orig[stack_index], image_shape, image_shape, 0, float(args.pixel_size), False, None)
        if error_msg is not None:
            continue
        pose = np.asarray([row[INCOL["angle_rot"]], row[INCOL["angle_tilt"]], row[INCOL["angle_psi"]]], dtype=float)
        ctf_params, dynamic_values, defocus_values, phase_values, center_defocus, center_phase = build_ctf_grid(row, base_ctf_params, args)
        z_grid, mip_grid, param_index, defocus, phase, ctf_set = run_param_search(image, image_shape, pose, template, ctf_params, dynamic_values, args)
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


def find_iter_pixel_star(iter_pixel_dir):
    p = Path(iter_pixel_dir)
    candidate = p / "iter_pixel_tm2d_results.star"
    if candidate.exists():
        return candidate
    matches = sorted(p.glob("*.star"))
    if not matches:
        raise FileNotFoundError(f"No STAR file found in {p}")
    return matches[0]


if __name__ == "__main__":
    main()
