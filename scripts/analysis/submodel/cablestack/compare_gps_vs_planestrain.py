"""Overlay new GPS stress-strain curves against the old plane-strain results.

Reads the OLD plane-strain data from
    <OLD_ROOT>/Data/                 (precomputed *_stress_strain.txt and raw fd_good_*.txt)
    <OLD_ROOT>/PressureData/         (raw fd_pressure_*, uy_top_*, fd_lateral_*, ux_left_*)
    <OLD_ROOT>/cable_parameters_user.json  (cable geometry: stack_height_mm, n_stacks)

Reads the NEW GPS data from each `<run>/APDL/submodel/apdl_runfolder/pp/<cable><suffix>_stress_strain.txt`,
written by `analyse_pressure.analyse`.

For each cable and each loading case where BOTH datasets exist, emits a side-by-side
overlay SVG to `<out_dir>/<cable>_<case>_GPSvsPS.svg`. Cases where only GPS data exists
are still emitted (single curve, title says "PS data unavailable").

Geometry conventions:
  total_height_m = n_stacks * stack_height_mm * 1e-3   (same OLD & NEW)
  total_width_m  = (cable_width_mm + 2*x_cab_margin_mm) * 1e-3

UY/UX in the OLD fd_good_*.txt header is labelled "(mm)" but values are in metres
(consistent with the OLD analyse_pressure.py reader). FY_total / FX_total are N per
unit out-of-plane depth.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OLD_ROOT = Path(r"C:/Users/vanden_j/OneDrive - ETH Zurich/Documents/ANSYS/0 - 10-stack/data analysis")
NEW_RUNS = {
    "R2D2_HF": Path(r"C:/LTS_Rutherford_Workflow/data/runs/20260504_232855_R2D2_HF_apdl_rerun_51/APDL/submodel/apdl_runfolder"),
    "R2D2_LF": Path(r"C:/LTS_Rutherford_Workflow/data/runs/20260511_171204_R2D2_LF_apdl_rerun_15/APDL/submodel/apdl_runfolder"),
    "CD1":     Path(r"C:/LTS_Rutherford_Workflow/data/runs/20260504_232855_CD1_apdl_rerun_52/APDL/submodel/apdl_runfolder"),
}

# (case_key, NEW pp suffix, OLD precomputed name pattern, friendly title)
CASES = [
    ("disp_transverse",     "",            "fd_good_{cable}.txt",                 "Displacement-controlled, transverse (Y)"),
    ("disp_radial",         "_disp_radial", None,                                  "Displacement-controlled, radial (X)"),
    ("pressure_transverse", "_pressure",   "{cable}_pressure_stress_strain.txt",  "Pressure-controlled, transverse (Y)"),
    ("pressure_radial",     "_radial",     "{cable}_lateral_stress_strain.txt",   "Pressure-controlled, radial (X)"),
]


from analysis_utils import read_stress_strain_curve


def _read_new_stress_strain(path: Path) -> Optional[pd.DataFrame]:
    """Read a NEW pp/*_stress_strain.txt into a DataFrame with strain + sigma columns."""
    arr = read_stress_strain_curve(path)
    if arr is None:
        return None
    return pd.DataFrame(arr, columns=["strain_load", "sigma_load_MPa"])


def _read_old_precomputed(path: Path, strain_name: str, sigma_name: str) -> Optional[pd.DataFrame]:
    """Read an OLD pre-computed *_stress_strain.txt (different header depth + column names)."""
    arr = read_stress_strain_curve(path, strain_name=strain_name, sigma_name=sigma_name)
    if arr is None:
        return None
    return pd.DataFrame(arr, columns=["strain_load", "sigma_load_MPa"])


def _read_old_fd_good(path: Path, total_width_m: float, total_height_m: float) -> Optional[pd.DataFrame]:
    """Read raw OLD fd_good_*.txt and convert to (strain_load, sigma_load_MPa) for disp_transverse."""
    if not path.is_file():
        return None
    rows: List[List[float]] = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows, columns=["Set", "Time", "UY", "FY_total", "UX", "FX_total"])
    df["strain_load"] = -df["UY"] / total_height_m
    df["sigma_load_MPa"] = df["FY_total"] / (total_width_m * 1e6)
    # Prepend origin
    df = pd.concat([pd.DataFrame([{c: 0.0 for c in df.columns}]), df], ignore_index=True)
    return df[["strain_load", "sigma_load_MPa"]]


def _old_geometry(cable: str) -> Tuple[float, float]:
    """(total_width_m, total_height_m) from OLD cable_parameters_user.json."""
    with (OLD_ROOT / "cable_parameters_user.json").open() as f:
        cfg = json.load(f)
    c = cfg["cables"][cable]
    margin_mm = cfg["cablestack"].get("x_cab_margin_mm", 0.0)
    total_width_m = (c["cable_width"] + 2.0 * margin_mm) * 1e-3
    total_height_m = c["n_stacks"] * c["stack_height_mm"] * 1e-3
    return total_width_m, total_height_m


def _load_old(cable: str, case_key: str, old_filename: Optional[str]) -> Optional[pd.DataFrame]:
    """Return OLD plane-strain curve for one (cable, case) or None if unavailable."""
    if case_key == "disp_radial":
        return None  # old project never ran radial-displacement

    if case_key == "disp_transverse":
        # use raw fd_good_<cable>.txt (no pre-computed exists)
        path = OLD_ROOT / "Data" / f"fd_good_{cable}.txt"
        tw, th = _old_geometry(cable)
        return _read_old_fd_good(path, tw, th)

    if case_key == "pressure_transverse":
        path = OLD_ROOT / "Data" / (old_filename.format(cable=cable))
        return _read_old_precomputed(path, "epsilon_y", "sigma_y_MPa")

    if case_key == "pressure_radial":
        # OLD pre-computed file uses "_lateral_" naming
        path = OLD_ROOT / "Data" / (old_filename.format(cable=cable))
        return _read_old_precomputed(path, "epsilon_x", "sigma_x_MPa")

    return None


def _load_new(cable: str, new_pp_suffix: str) -> Optional[pd.DataFrame]:
    """Return NEW GPS curve for one (cable, case) or None if unavailable."""
    apdl_rf = NEW_RUNS.get(cable)
    if apdl_rf is None:
        return None
    path = apdl_rf / "pp" / f"{cable}{new_pp_suffix}_stress_strain.txt"
    return _read_new_stress_strain(path)


def compare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Tuple[str, str, str, Optional[float], Optional[float]]] = []

    for cable in NEW_RUNS:
        for case_key, new_suffix, old_filename, descr in CASES:
            new_df = _load_new(cable, new_suffix)
            old_df = _load_old(cable, case_key, old_filename)

            if new_df is None and old_df is None:
                summary.append((cable, case_key, "missing", None, None))
                continue

            fig, ax = plt.subplots(figsize=(8.5, 6.0))
            old_peak = new_peak = None
            if old_df is not None:
                ax.plot(old_df["strain_load"] * 100.0, old_df["sigma_load_MPa"],
                        "k--o", markersize=4, linewidth=1.5, alpha=0.8,
                        label=f"Plane strain (old, n={len(old_df)})")
                old_peak = float(np.abs(old_df["sigma_load_MPa"]).max())
            if new_df is not None:
                ax.plot(new_df["strain_load"] * 100.0, new_df["sigma_load_MPa"],
                        "C0-o", markersize=4, linewidth=1.8,
                        label=f"GPS + mixed u-P (new, n={len(new_df)})")
                new_peak = float(np.abs(new_df["sigma_load_MPa"]).max())

            title = f"{cable} - {descr}"
            note_parts = []
            if old_df is None:
                note_parts.append("no plane-strain data")
            if new_df is None:
                note_parts.append("no GPS data")
            if note_parts:
                title += f"  [{', '.join(note_parts)}]"
            ax.set_title(title)
            ax.set_xlabel("strain_load (%)")
            ax.set_ylabel("sigma_load (MPa)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            out_path = out_dir / f"{cable}_{case_key}_GPSvsPS.svg"
            fig.savefig(out_path)
            plt.close(fig)
            summary.append((cable, case_key, "ok", old_peak, new_peak))
            print(f"-> {out_path}")

    # Summary table
    print("\nSummary: peak |sigma_load_MPa|")
    print(f"  {'cable':<10} {'case':<22} {'status':<10} {'PS_peak':>10}  {'GPS_peak':>10}  GPS/PS")
    for cable, case, status, op, np_ in summary:
        ratio = (np_ / op) if (op and np_) else None
        op_s  = f"{op:.2f}" if op is not None else "    -"
        np_s_ = f"{np_:.2f}" if np_ is not None else "    -"
        ratio_s = f"{ratio:.3f}" if ratio is not None else "  -"
        print(f"  {cable:<10} {case:<22} {status:<10} {op_s:>10}  {np_s_:>10}  {ratio_s}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:/LTS_Rutherford_Workflow/data/runs/comparison_GPSvsPS_2026-05-15"
    )
    compare(out)
