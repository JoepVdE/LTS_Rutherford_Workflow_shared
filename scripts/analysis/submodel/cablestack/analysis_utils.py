"""Shared helpers for the cablestack analysis scripts in this directory.

Canonical home for logic that was previously copy-pasted across
analyse_pressure.py, plot_fd_good.py, compare_cables.py, and the
ad-hoc diagnostic plotters:

  * locating an apdl_runfolder from a run-folder path,
  * finding the latest run folder under data/runs/,
  * extracting the cable label from an apdl_runfolder path,
  * parsing the whitespace float-table dumps written by the APDL PP decks
    (fd_good_*.txt, fd_pressure_*.txt, uy_top_*.txt, ...),
  * parsing the postprocessed pp/*_stress_strain.txt exports.

Import style: plain same-directory import (``from analysis_utils import ...``).
This works both when a script in this directory is run standalone (the script's
own directory is sys.path[0]) and when main.py inserts this directory on
sys.path before importing analyse_pressure.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd


# Repo root: this file lives at scripts/analysis/submodel/cablestack/.
REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = REPO_ROOT / "data" / "runs"


# ── Run-folder resolution ─────────────────────────────────────────────────────

def find_apdl_runfolder(path) -> str:
    """Return the apdl_runfolder for a top-level run folder, or the folder itself.

    Accepts either a top-level run folder (containing APDL/submodel/apdl_runfolder)
    or a path that already IS an apdl_runfolder (detected via 0-start.inp).
    Raises FileNotFoundError when neither matches.
    """
    path = os.fspath(path)
    candidate = os.path.join(path, "APDL", "submodel", "apdl_runfolder")
    if os.path.isdir(candidate):
        return candidate
    if os.path.isfile(os.path.join(path, "0-start.inp")):
        return path
    raise FileNotFoundError(f"Cannot find apdl_runfolder under {path}")


def iter_apdl_runfolders_latest_first() -> Iterator[str]:
    """Yield candidate apdl_runfolder paths under data/runs/, newest run first.

    Yields ``<run>/APDL/submodel/apdl_runfolder`` for every entry in data/runs/
    (run folders are named YYYYMMDD_HHMMSS_<CABLE>..., so reverse name sort is
    reverse chronological).  The yielded path is NOT checked for existence --
    callers filter on whatever marker file they need.
    Raises FileNotFoundError when data/runs/ itself is missing.
    """
    if not RUNS_ROOT.is_dir():
        raise FileNotFoundError(f"No data/runs/ directory at {RUNS_ROOT}")
    for run_name in sorted(os.listdir(RUNS_ROOT), reverse=True):
        yield os.path.join(str(RUNS_ROOT), run_name, "APDL", "submodel", "apdl_runfolder")


def find_latest_apdl_runfolder(marker: str = "loading_cycle.json") -> str:
    """Most recent apdl_runfolder under data/runs/ that contains `marker`.

    Raises FileNotFoundError when no run folder qualifies (or data/runs/ is
    missing entirely).
    """
    for apdl_rf in iter_apdl_runfolders_latest_first():
        if os.path.isfile(os.path.join(apdl_rf, marker)):
            return apdl_rf
    raise FileNotFoundError(f"No run folder with {marker} under data/runs/")


def cable_label_from_runfolder(apdl_runfolder) -> str:
    """Pull '<CABLE>' out of a '<run>/APDL/submodel/apdl_runfolder' path.

    Run folders are named ``YYYYMMDD_HHMMSS_<CABLE>[_apdl_rerun[_N]]`` -- the
    first two underscore-separated fields are the timestamp, the rest is the
    cable label with any ``_apdl_rerun`` suffix stripped.
    """
    run_folder = Path(apdl_runfolder).parents[2].name
    parts = run_folder.split("_", 2)
    if len(parts) < 3:
        return run_folder
    label = parts[2]
    for suffix in ("_apdl_rerun",):
        if suffix in label:
            label = label.split(suffix)[0]
    return label


# ── File readers ──────────────────────────────────────────────────────────────

def read_float_table(filepath, ncols: int, columns: List[str]) -> pd.DataFrame:
    """Whitespace-separated float table reader for the APDL PP dumps.

    Keeps only lines that split into exactly `ncols` fields, all parseable as
    float -- header lines, blank lines and partial writes are skipped silently.
    Returns a DataFrame with the given column names (empty, but with columns,
    when no data row parses).
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) != ncols:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=columns)


def read_node_mean(filepath, value_name: str) -> pd.DataFrame:
    """Mean nodal value per (LoadStep, SubStep) from a 4-column nodal dump.

    Input columns: LoadStep, SubStep, NodeID, <value> (e.g. uy_top_*.txt /
    ux_left_*.txt written by 8-PP-pressure.inp / 8-PP-radial.inp).  Returns a
    DataFrame [LoadStep, SubStep, <value_name>] with the per-substep mean.
    """
    df = read_float_table(filepath, 4, ["LoadStep", "SubStep", "NodeID", value_name])
    return df.groupby(["LoadStep", "SubStep"], as_index=False)[value_name].mean()


def read_stress_strain_curve(path,
                             strain_name: str = "strain_load",
                             sigma_name: str = "sigma_load_MPa") -> Optional[np.ndarray]:
    """Parse a postprocessed *_stress_strain.txt into an (n, 2) [strain, sigma] array.

    Format (written by analyse_pressure): a variable-depth ``#`` comment header,
    then one column-name line, then whitespace-separated float rows.  The two
    requested columns are located by name on the header line, so the reader is
    robust to the differing column sets of the displacement vs pressure exports
    (and, with explicit names, to the OLD plane-strain exports).

    Returns None when the file is missing, the named columns are absent from
    the header, or fewer than 2 data rows parse.  Never raises on bad input.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        return None
    strain_col = sigma_col = None
    data_rows: List[List[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if strain_col is None:
                names = line.split()
                try:
                    strain_col = names.index(strain_name)
                    sigma_col = names.index(sigma_name)
                except ValueError:
                    return None
                continue
            try:
                parts = [float(x) for x in line.split()]
            except ValueError:
                continue
            if max(strain_col, sigma_col) < len(parts):
                data_rows.append([parts[strain_col], parts[sigma_col]])
    if len(data_rows) < 2:
        return None
    return np.asarray(data_rows)
