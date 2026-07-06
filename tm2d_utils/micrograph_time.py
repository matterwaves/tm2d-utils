import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


def fix_timestamp_in_path(path_str):
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})(.*)$")
    dirname, basename = os.path.split(str(path_str))
    match = pattern.match(basename)
    if not match:
        return path_str

    date, hour, minute, second, rest = match.groups()
    new_basename = f"{date}_{hour}.{minute}.{second}{rest}"
    if dirname:
        return os.path.join(dirname, new_basename)
    return new_basename


def get_micrograph_datetime(fpath, relion=None, apply_tzinfo=True, timezone="America/Los_Angeles"):
    fname = os.path.basename(str(fpath))
    patterns = []
    if relion is True:
        patterns.append(r"^(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})")
    elif relion is False:
        patterns.append(r"^(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})")
    else:
        patterns.extend([
            r"^(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})",
            r"^(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})",
        ])

    for pattern in patterns:
        match = re.match(pattern, fname)
        if match:
            date_str, hour, minute, second = match.groups()
            dt = datetime.strptime(f"{date_str} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
            if apply_tzinfo:
                dt = dt.replace(tzinfo=ZoneInfo(timezone))
            return dt

    fixed = fix_timestamp_in_path(fname)
    if fixed != fname:
        return get_micrograph_datetime(fixed, relion=False, apply_tzinfo=apply_tzinfo, timezone=timezone)

    raise ValueError(f"No valid timestamp found in filename: {fname}")


def micrograph_timestamp_table(micrograph_paths, relion=True, timezone="America/Los_Angeles"):
    rows = []
    for mic_path in pd.unique(pd.Series(micrograph_paths).dropna()):
        rows.append({
            "rlnMicrographName": mic_path,
            "micrograph_datetime": get_micrograph_datetime(mic_path, relion=relion, timezone=timezone),
        })
    table = pd.DataFrame(rows).sort_values("micrograph_datetime").reset_index(drop=True)
    table["micrograph_time_order"] = range(len(table))
    return table


def add_micrograph_timestamps(df, micrograph_col="rlnMicrographName", relion=True, timezone="America/Los_Angeles"):
    table = micrograph_timestamp_table(df[micrograph_col], relion=relion, timezone=timezone)
    return df.merge(table, left_on=micrograph_col, right_on="rlnMicrographName", how="left", suffixes=("", "_timestamp"))
