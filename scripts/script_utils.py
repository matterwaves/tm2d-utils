"""Shared defaults and small helpers for tm2d-utils command-line scripts."""

import argparse
import csv
import json
from pathlib import Path

try:
    from tm2d_utils import local_parameters as local_params
except ImportError:
    from tm2d_utils import local_parameters_template as local_params


DEFAULT_MODEL_TYPE = "atomic"
DEFAULT_PARTICLE_LIMIT = 1000
DEFAULT_PIXEL_SIZE_GUESS_A = 0.725
DEFAULT_B_FACTOR_A2 = 50.0
DEFAULT_WORKSPACE_ROOT = local_params.DEFAULT_WORKSPACE_ROOT
DEFAULT_WORKSPACE_ROOT_IS_REMOTE = local_params.DEFAULT_WORKSPACE_ROOT_IS_REMOTE
DEFAULT_REMOTE_HOST = getattr(local_params, "DEFAULT_REMOTE_HOST", None)
DEFAULT_REMOTE_USER = getattr(local_params, "DEFAULT_REMOTE_USER", None)
DEFAULT_REMOTE_KEY_FILENAME = getattr(local_params, "DEFAULT_REMOTE_KEY_FILENAME", None)
DEFAULT_SESSION_NAME = None
DEFAULT_SESSION_LASER_STATE = None
DEFAULT_SESSION_JOB_TYPE = "Refine3D"
DEFAULT_SESSION_JOB_NUM = 193
DEFAULT_DENSITY_PADDING_FACTOR = 2
DEFAULT_SYMMETRY = "C1"
DEFAULT_DIAMETER_A = None
DEFAULT_APOF_SYMMETRY = "O"
DEFAULT_APOF_DIAMETER_A = 130.0
DEFAULT_SEARCH_RES_A = 3.0
DEFAULT_DEVICES = "0,1,2,3"
DEFAULT_BATCH_SIZE = 24
DEFAULT_PIXEL_TEMPLATE_BATCH_SIZE = 1
DEFAULT_PDB_FPATH = local_params.DEFAULT_PDB_FPATH
DEFAULT_DENSITY_HELPER_FPATHS = tuple(local_params.DEFAULT_DENSITY_HELPER_FPATHS)


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
    root_label = sanitize_label(Path(args.workspace_root).name if args.workspace_root is not None else "workspace")
    name_label = sanitize_label(args.session_name) if args.session_name is not None else "session"
    return (
        f"{prefix}_{args.model_type}_{root_label}_{name_label}_"
        f"{args.session_job_type}{args.session_job_num}_mic{args.micrograph_start_index}_n{args.num_micrographs}"
    )


def resolve_arg(value, metadata, key, default):
    if value is not None:
        return value
    if metadata is not None and key in metadata and metadata[key] is not None:
        return metadata[key]
    return default


def session_workspace_kwargs(args):
    workspace_kwargs = {}
    remote_host = getattr(args, "remote_host", DEFAULT_REMOTE_HOST)
    remote_user = getattr(args, "remote_user", DEFAULT_REMOTE_USER)
    remote_key_filename = getattr(args, "remote_key_filename", DEFAULT_REMOTE_KEY_FILENAME)
    if remote_host is not None:
        workspace_kwargs["remote_host"] = remote_host
    if remote_user is not None:
        workspace_kwargs["remote_user"] = remote_user
    if remote_key_filename is not None:
        workspace_kwargs["key_filename"] = remote_key_filename
    return workspace_kwargs


def parse_float_list(spec):
    if spec is None or spec == "":
        return []
    return [float(value) for value in str(spec).split(",") if value != ""]


def parse_range(center, half_width, step):
    n_steps = int(round((2.0 * half_width) / step))
    return [round(center - half_width + step * ind, 6) for ind in range(n_steps + 1)]


def parse_float_values(spec):
    if spec is None or spec == "":
        return None
    return [float(value) for value in str(spec).split(",") if value != ""]


def parse_device_ids(devices):
    if isinstance(devices, str):
        return [int(device) for device in devices.split(",") if device != ""]
    return [int(device) for device in devices]


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


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
